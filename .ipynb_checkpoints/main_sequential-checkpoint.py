import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from sequential_adapter import SequentialEvidenceAdapter, extract_spatial_tokens
from utils import build_cache_model, clip_classifier, cls_acc, search_hp


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='yaml config path')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_loader_with_tokens(loader, clip_model, device, upsample_size=14):
    global_features, patch_features, labels = [], [], []
    for images, target in loader:
        images = images.to(device)
        target = target.to(device)
        g, p = extract_spatial_tokens(clip_model, images, upsample_size=upsample_size)
        global_features.append(g)
        patch_features.append(p)
        labels.append(target)

    return torch.cat(global_features), torch.cat(patch_features), torch.cat(labels)


def build_train_batches(global_features, patch_features, labels, batch_size=1):
    batches = []
    n = labels.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((global_features[start:end], patch_features[start:end], labels[start:end]))
    return batches


@torch.no_grad()
def eval_zero_shot(global_features, labels, clip_weights):
    logits = 100.0 * global_features @ clip_weights
    return cls_acc(logits, labels)


@torch.no_grad()
def eval_tip_adapter(global_features, labels, clip_weights, cache_keys, cache_values, beta, alpha):
    affinity = global_features @ cache_keys
    cache_values = cache_values.to(dtype=affinity.dtype, device=affinity.device)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    logits = 100.0 * global_features @ clip_weights + alpha * cache_logits
    return cls_acc(logits, labels)


@torch.no_grad()
def eval_fixed_two_stage(global_features, patch_features, labels, clip_weights, cache_keys, cache_values, beta, alpha, top_r=5, patch_k=8):
    # deterministic refinement baseline (no policy learning)
    clip_logits = 100.0 * global_features @ clip_weights
    affinity = global_features @ cache_keys
    cache_values = cache_values.to(dtype=affinity.dtype, device=affinity.device)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    fused = clip_logits + alpha * cache_logits
    top_r_idx = fused.topk(k=top_r, dim=-1).indices

    # stage 1: discriminative patch selection against top-r candidates
    class_weights = clip_weights.t()[top_r_idx]  # [B, r, D]
    patch_scores = torch.einsum('bpd,bcd->bpc', patch_features, class_weights)
    disc = patch_scores[..., 0] - patch_scores[..., 1:].max(dim=-1).values
    idx = disc.topk(k=min(patch_k, disc.shape[-1]), dim=-1).indices
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, patch_features.shape[-1])
    selected = torch.gather(patch_features, 1, gather_idx)
    local = selected.mean(dim=1)
    patch_logits = 100.0 * local @ clip_weights

    # stage 2: top-r class-filtered cache
    class_mask = torch.nn.functional.one_hot(top_r_idx, num_classes=cache_values.shape[1]).sum(dim=1).clamp(max=1)
    masked_cache_values = cache_values.unsqueeze(0) * class_mask[:, None, :].float()
    restricted_cache_logits = ((-1) * (beta - beta * affinity[:, :, None])).exp() * masked_cache_values
    restricted_cache_logits = restricted_cache_logits.sum(dim=1)

    logits = 0.4 * clip_logits + 0.4 * restricted_cache_logits + 0.2 * patch_logits
    return cls_acc(logits, labels)


def evaluate_sequential(model, global_features, patch_features, labels):
    model.eval()
    with torch.no_grad():
        logits = model.predict(global_features, patch_features)
    return cls_acc(logits, labels)


def save_checkpoint(model, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        'policy': model.policy.state_dict(),
        'verifier': model.verifier.state_dict(),
    }
    path = os.path.join(out_dir, f'{tag}.pt')
    torch.save(ckpt, path)
    return path


def main():
    args = get_arguments()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    seed = cfg.get('seed', 1)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    clip_model, preprocess = clip.load(cfg['backbone'], device=device)
    clip_model.eval()

    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg.get('eval_batch_size', 64), is_train=False, tfm=preprocess, shuffle=False)
    val_loader = build_data_loader(data_source=dataset.val, batch_size=cfg.get('eval_batch_size', 64), is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=cfg.get('eval_batch_size', 64), is_train=False, tfm=preprocess, shuffle=False)

    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader)

    tr_global, tr_patch, tr_labels = encode_loader_with_tokens(train_loader, clip_model, device, upsample_size=cfg.get('seq_upsample_size', 14))
    va_global, va_patch, va_labels = encode_loader_with_tokens(val_loader, clip_model, device, upsample_size=cfg.get('seq_upsample_size', 14))
    te_global, te_patch, te_labels = encode_loader_with_tokens(test_loader, clip_model, device, upsample_size=cfg.get('seq_upsample_size', 14))

    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, va_global, va_labels, clip_weights)
    if best_beta == 0:
        best_beta, best_alpha = cfg['init_beta'], cfg['init_alpha']

    # Baselines
    results = {
        'zero_shot_val': eval_zero_shot(va_global, va_labels, clip_weights),
        'zero_shot_test': eval_zero_shot(te_global, te_labels, clip_weights),
        'tip_adapter_val': eval_tip_adapter(va_global, va_labels, clip_weights, cache_keys, cache_values, best_beta, best_alpha),
        'tip_adapter_test': eval_tip_adapter(te_global, te_labels, clip_weights, cache_keys, cache_values, best_beta, best_alpha),
        'fixed_two_stage_val': eval_fixed_two_stage(va_global, va_patch, va_labels, clip_weights, cache_keys, cache_values, best_beta, best_alpha,
                                                    top_r=cfg.get('seq_top_r', 5), patch_k=cfg.get('seq_patch_k', 8)),
        'fixed_two_stage_test': eval_fixed_two_stage(te_global, te_patch, te_labels, clip_weights, cache_keys, cache_values, best_beta, best_alpha,
                                                     top_r=cfg.get('seq_top_r', 5), patch_k=cfg.get('seq_patch_k', 8)),
    }

    # Supervised router baseline (imitation only)
    supervised_router = SequentialEvidenceAdapter(
        clip_weights=clip_weights,
        cache_keys=cache_keys,
        cache_values=cache_values,
        beta=best_beta,
        alpha=best_alpha,
        top_r=cfg.get('seq_top_r', 5),
        patch_k=cfg.get('seq_patch_k', 8),
        max_steps=cfg.get('seq_max_steps', 3),
    ).to(device)

    train_batches = build_train_batches(tr_global, tr_patch, tr_labels, batch_size=cfg.get('seq_batch_size', 1))
    supervised_router.train_imitation(train_batches, epochs=cfg.get('seq_imitation_epochs', 5), lr=cfg.get('seq_imitation_lr', 1e-3))
    results['supervised_router_val'] = evaluate_sequential(supervised_router, va_global, va_patch, va_labels)
    results['supervised_router_test'] = evaluate_sequential(supervised_router, te_global, te_patch, te_labels)
    sup_ckpt = save_checkpoint(supervised_router, cache_dir, f'seq_supervised_{cfg["shots"]}shot')

    # RL fine-tuned policy
    rl_router = SequentialEvidenceAdapter(
        clip_weights=clip_weights,
        cache_keys=cache_keys,
        cache_values=cache_values,
        beta=best_beta,
        alpha=best_alpha,
        top_r=cfg.get('seq_top_r', 5),
        patch_k=cfg.get('seq_patch_k', 8),
        max_steps=cfg.get('seq_max_steps', 3),
    ).to(device)

    rl_router.load_state_dict(supervised_router.state_dict())
    rl_router.train_reinforce(
        train_batches,
        epochs=cfg.get('seq_rl_epochs', 3),
        lr=cfg.get('seq_rl_lr', 1e-4),
        reward_alpha=cfg.get('seq_reward_alpha', 0.1),
        reward_beta=cfg.get('seq_reward_beta', 0.1),
        reward_gamma=cfg.get('seq_reward_gamma', 0.002),
    )

    results['sequential_rl_val'] = evaluate_sequential(rl_router, va_global, va_patch, va_labels)
    results['sequential_rl_test'] = evaluate_sequential(rl_router, te_global, te_patch, te_labels)
    rl_ckpt = save_checkpoint(rl_router, cache_dir, f'seq_rl_{cfg["shots"]}shot')

    payload = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'dataset': cfg['dataset'],
        'backbone': cfg['backbone'],
        'shots': cfg['shots'],
        'seed': seed,
        'best_beta': best_beta,
        'best_alpha': best_alpha,
        'supervised_checkpoint': sup_ckpt,
        'rl_checkpoint': rl_ckpt,
        'results': results,
    }

    result_path = os.path.join(cache_dir, f'seq_results_{cfg["shots"]}shot_seed{seed}.json')
    with open(result_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print('\n=== Sequential Evidence Experiment Summary ===')
    for k, v in results.items():
        print(f'{k}: {v:.2f}')
    print(f'Results saved to: {result_path}')


if __name__ == '__main__':
    main()

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_STOP = 0
ACTION_INSPECT_DISCRIMINATIVE = 1
ACTION_INSPECT_AMBIGUOUS = 2
ACTION_RETRIEVE_TOPR_SUPPORT = 3
ACTION_RETRIEVE_ALIGNED_SUPPORT = 4
ACTION_REFINE_FUSION = 5


@dataclass
class EvidenceState:
    state_vector: torch.Tensor
    clip_logits: torch.Tensor
    cache_logits: torch.Tensor
    patch_logits: torch.Tensor
    top_candidates: torch.Tensor
    verifier_score: torch.Tensor
    margin: torch.Tensor
    entropy: torch.Tensor
    agreement: torch.Tensor


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_actions: int = 6, use_gru: bool = False):
        super().__init__()
        self.use_gru = use_gru
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        if use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        state = torch.nan_to_num(state, nan=0.0, posinf=1e4, neginf=-1e4)
        x = self.backbone(state)
        if self.use_gru:
            x, next_hidden = self.gru(x.unsqueeze(1), hidden)
            x = x[:, 0]
        else:
            next_hidden = hidden

        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value, next_hidden


class VerifierNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor):
        state = torch.nan_to_num(state, nan=0.0, posinf=1e4, neginf=-1e4)
        return self.net(state).squeeze(-1)


class RefinementHead(nn.Module):
    """Discrete fusion templates keep RL stable at initialization."""

    def __init__(self):
        super().__init__()
        self.templates = {
            "global_heavy": (0.75, 0.20, 0.05),
            "cache_heavy": (0.20, 0.75, 0.05),
            "local_heavy": (0.20, 0.10, 0.70),
            "balanced": (0.40, 0.40, 0.20),
        }
        self.template_names = list(self.templates.keys())

    def forward(self, clip_logits: torch.Tensor, cache_logits: torch.Tensor, patch_logits: torch.Tensor, template_id: int):
        name = self.template_names[int(template_id) % len(self.template_names)]
        w0, w1, w2 = self.templates[name]
        return w0 * clip_logits + w1 * cache_logits + w2 * patch_logits


def _entropy(logits: torch.Tensor):
    probs = logits.softmax(dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def extract_vit_patch_tokens(clip_model, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns normalized CLS feature and normalized projected patch features for ViT-based CLIP."""
    visual = clip_model.visual
    if not hasattr(visual, "transformer"):
        raise ValueError("Sequential evidence adapter currently supports ViT CLIP backbones only.")

    x = images.type(clip_model.dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    x_ln = visual.ln_post(x)
    cls_tokens = x_ln[:, 0]
    patch_tokens = x_ln[:, 1:]

    if visual.proj is not None:
        cls_tokens = cls_tokens @ visual.proj
        patch_tokens = patch_tokens @ visual.proj

    cls_tokens = F.normalize(cls_tokens, dim=-1)
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    return cls_tokens, patch_tokens


def extract_resnet_spatial_tokens(
    clip_model, images: torch.Tensor, upsample_size: int = 14
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns normalized global feature and projected spatial tokens for ResNet-based CLIP.
    Spatial tokens are upsampled from 7x7 to 14x14 by default.
    """
    visual = clip_model.visual
    if not hasattr(visual, "attnpool"):
        raise ValueError("extract_resnet_spatial_tokens expects a ResNet CLIP visual backbone.")

    x = images.type(visual.conv1.weight.dtype)
    for conv, bn in [(visual.conv1, visual.bn1), (visual.conv2, visual.bn2), (visual.conv3, visual.bn3)]:
        x = visual.relu(bn(conv(x)))
    x = visual.avgpool(x)
    x = visual.layer1(x)
    x = visual.layer2(x)
    x = visual.layer3(x)
    x = visual.layer4(x)

    attnpool = visual.attnpool
    tokens = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
    tokens = torch.cat([tokens.mean(dim=0, keepdim=True), tokens], dim=0)
    tokens = tokens + attnpool.positional_embedding[:, None, :].to(tokens.dtype)
    tokens, _ = F.multi_head_attention_forward(
        query=tokens,
        key=tokens,
        value=tokens,
        embed_dim_to_check=tokens.shape[-1],
        num_heads=attnpool.num_heads,
        q_proj_weight=attnpool.q_proj.weight,
        k_proj_weight=attnpool.k_proj.weight,
        v_proj_weight=attnpool.v_proj.weight,
        in_proj_weight=None,
        in_proj_bias=torch.cat([attnpool.q_proj.bias, attnpool.k_proj.bias, attnpool.v_proj.bias]),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0,
        out_proj_weight=attnpool.c_proj.weight,
        out_proj_bias=attnpool.c_proj.bias,
        use_separate_proj_weight=True,
        training=attnpool.training,
        need_weights=False,
    )

    global_tokens = F.normalize(tokens[0], dim=-1)
    spatial_tokens = tokens[1:].permute(1, 2, 0)
    side = int(spatial_tokens.shape[-1] ** 0.5)
    feature_map = spatial_tokens.reshape(spatial_tokens.shape[0], spatial_tokens.shape[1], side, side)
    if upsample_size is not None and upsample_size > side:
        feature_map = F.interpolate(feature_map, size=(upsample_size, upsample_size), mode="bicubic", align_corners=False)
    spatial_tokens = feature_map.flatten(2).permute(0, 2, 1)
    spatial_tokens = F.normalize(spatial_tokens, dim=-1)
    return global_tokens, spatial_tokens


def extract_spatial_tokens(clip_model, images: torch.Tensor, upsample_size: int = 14):
    """Backbone-agnostic extraction helper: ViT patches or RN spatial tokens."""
    visual = clip_model.visual
    if hasattr(visual, "transformer"):
        return extract_vit_patch_tokens(clip_model, images)
    if hasattr(visual, "attnpool"):
        return extract_resnet_spatial_tokens(clip_model, images, upsample_size=upsample_size)
    raise ValueError("Unsupported CLIP visual backbone for spatial token extraction.")


class SequentialEvidenceAdapter(nn.Module):
    def __init__(
        self,
        clip_weights: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        beta: float,
        alpha: float,
        top_r: int = 5,
        patch_k: int = 8,
        max_steps: int = 3,
        state_scale: float = 50.0,
        state_dim: Optional[int] = None,
    ):
        super().__init__()
        self.register_buffer("clip_weights", clip_weights)
        self.register_buffer("cache_keys", cache_keys)
        self.register_buffer("cache_values", cache_values)
        self.beta = beta
        self.alpha = alpha
        self.top_r = top_r
        self.patch_k = patch_k
        self.max_steps = max_steps
        self.state_scale = state_scale

        inferred_state_dim = 4 * top_r + 6
        self.state_dim = state_dim or inferred_state_dim

        self.policy = PolicyNetwork(self.state_dim)
        self.verifier = VerifierNetwork(self.state_dim)
        self.refine = RefinementHead()

    def _sanitize_state_vector(self, vector: torch.Tensor):
        vector = torch.nan_to_num(vector, nan=0.0, posinf=1e4, neginf=-1e4)
        # Keep optimization numerically stable for policy/verifier MLPs.
        vector = torch.clamp(vector / self.state_scale, min=-20.0, max=20.0)
        return vector

    def _cache_logits(self, image_features: torch.Tensor, class_mask: Optional[torch.Tensor] = None):
        affinity = image_features @ self.cache_keys
        cache_values = self.cache_values.to(dtype=affinity.dtype, device=affinity.device)
        if class_mask is not None:
            one_hot_mask = class_mask[:, None, :].float()
            masked_cache_values = cache_values.unsqueeze(0) * one_hot_mask
            cache_logits = ((-1) * (self.beta - self.beta * affinity[:, :, None])).exp() * masked_cache_values
            return cache_logits.sum(dim=1)

        return ((-1) * (self.beta - self.beta * affinity)).exp() @ cache_values

    def _patch_logits(self, patch_features: torch.Tensor):
        patch_to_text = torch.einsum("bpd,dc->bpc", patch_features, self.clip_weights)
        return patch_to_text.mean(dim=1)

    def _build_state(self, clip_logits: torch.Tensor, cache_logits: torch.Tensor, patch_logits: torch.Tensor):
        fused = clip_logits + self.alpha * cache_logits
        probs = fused.softmax(dim=-1)
        top2_vals, top2_idx = fused.topk(k=2, dim=-1)
        margin = top2_vals[:, 0] - top2_vals[:, 1]
        entropy = _entropy(fused)

        cache_top = cache_logits.argmax(dim=-1)
        clip_top = clip_logits.argmax(dim=-1)
        agreement = (cache_top == clip_top).float()

        top_r_vals, top_r_idx = fused.topk(k=self.top_r, dim=-1)
        top_r_cache = cache_logits.gather(1, top_r_idx)
        top_r_patch = patch_logits.gather(1, top_r_idx)
        top_r_probs = probs.gather(1, top_r_idx)

        raw_vector = torch.cat(
            [
                top_r_vals,
                top_r_cache,
                top_r_patch,
                top_r_probs,
                margin[:, None],
                entropy[:, None],
                agreement[:, None],
                (top2_idx[:, 0] - top2_idx[:, 1]).float().unsqueeze(1),
                top2_vals,
            ],
            dim=-1,
        )
        vector = self._sanitize_state_vector(raw_vector)

        verifier_score = torch.sigmoid(self.verifier(vector))
        return EvidenceState(
            state_vector=vector,
            clip_logits=clip_logits,
            cache_logits=cache_logits,
            patch_logits=patch_logits,
            top_candidates=top_r_idx,
            verifier_score=verifier_score,
            margin=margin,
            entropy=entropy,
            agreement=agreement,
        )

    def _discriminative_patch_score(self, patch_features: torch.Tensor, candidate_idx: torch.Tensor):
        candidate_weights = self.clip_weights.t()[candidate_idx]
        patch_scores = torch.einsum("bpd,bcd->bpc", patch_features, candidate_weights)
        top = patch_scores[..., 0]
        rivals = patch_scores[..., 1:].max(dim=-1).values
        return top - rivals

    def _ambiguous_patch_score(self, patch_features: torch.Tensor, candidate_idx: torch.Tensor):
        candidate_weights = self.clip_weights.t()[candidate_idx[:, :2]]
        patch_scores = torch.einsum("bpd,bcd->bpc", patch_features, candidate_weights)
        return -torch.abs(patch_scores[..., 0] - patch_scores[..., 1])

    def _select_patch_features(self, patch_features: torch.Tensor, scores: torch.Tensor):
        idx = scores.topk(k=min(self.patch_k, scores.shape[-1]), dim=-1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, patch_features.shape[-1])
        selected = torch.gather(patch_features, 1, gather_idx)
        weights = torch.softmax(scores.gather(1, idx), dim=-1).unsqueeze(-1)
        return (selected * weights).sum(dim=1)

    def heuristic_action(self, state: EvidenceState):
        if state.margin.mean().item() > 1.5 and state.verifier_score.mean().item() > 0.7:
            return ACTION_STOP
        if state.agreement.mean().item() < 0.5:
            return ACTION_RETRIEVE_TOPR_SUPPORT
        if state.margin.mean().item() < 0.3:
            return ACTION_INSPECT_AMBIGUOUS
        return ACTION_INSPECT_DISCRIMINATIVE

    def forward_episode(self, image_features: torch.Tensor, patch_features: torch.Tensor, training: bool = False):
        if image_features.shape[0] != 1:
            raise ValueError("forward_episode expects batch size 1. Use predict() for batched inference.")
        clip_logits = 100.0 * image_features @ self.clip_weights
        cache_logits = self._cache_logits(image_features)
        patch_logits = self._patch_logits(patch_features)

        hidden = None
        trajectory = []
        final_logits = clip_logits + self.alpha * cache_logits
        compute_cost = 0.0

        for step in range(self.max_steps):
            state = self._build_state(clip_logits, cache_logits, patch_logits)

            logits, value, hidden = self.policy(state.state_vector, hidden)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            value = torch.nan_to_num(value, nan=0.0, posinf=50.0, neginf=-50.0)
            if training:
                if not torch.isfinite(logits).all():
                    action = torch.zeros((1,), dtype=torch.long, device=logits.device)
                    log_prob = torch.zeros((1,), dtype=state.state_vector.dtype, device=logits.device)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
            else:
                action = logits.argmax(dim=-1)
                log_prob = torch.zeros_like(action, dtype=state.state_vector.dtype)

            action_id = int(action[0].item())
            template_id = step % len(self.refine.template_names)

            if action_id == ACTION_STOP:
                final_logits = self.refine(clip_logits, cache_logits, patch_logits, template_id)
                trajectory.append((log_prob, value, action, state))
                break
            elif action_id == ACTION_INSPECT_DISCRIMINATIVE:
                score = self._discriminative_patch_score(patch_features, state.top_candidates)
                local_feature = self._select_patch_features(patch_features, score)
                patch_logits = 100.0 * local_feature @ self.clip_weights
                compute_cost += float(self.patch_k)
            elif action_id == ACTION_INSPECT_AMBIGUOUS:
                score = self._ambiguous_patch_score(patch_features, state.top_candidates)
                local_feature = self._select_patch_features(patch_features, score)
                patch_logits = 100.0 * local_feature @ self.clip_weights
                compute_cost += float(self.patch_k)
            elif action_id == ACTION_RETRIEVE_TOPR_SUPPORT:
                class_mask = F.one_hot(state.top_candidates, num_classes=self.cache_values.shape[1]).sum(dim=1).clamp(max=1)
                cache_logits = self._cache_logits(image_features, class_mask=class_mask)
                compute_cost += 5.0
            elif action_id == ACTION_RETRIEVE_ALIGNED_SUPPORT:
                local_scores = torch.einsum("bpd,dk->bpk", patch_features, self.cache_keys)
                support_score = local_scores.mean(dim=1)
                keep = support_score.topk(k=min(64, support_score.shape[-1]), dim=-1).indices
                sparse = torch.zeros_like(support_score)
                sparse.scatter_(1, keep, 1.0)
                affinity = image_features @ self.cache_keys
                cache_values = self.cache_values.to(dtype=affinity.dtype, device=affinity.device)
                cache_logits = ((-1) * (self.beta - self.beta * (affinity * sparse))).exp() @ cache_values
                compute_cost += 8.0
            elif action_id == ACTION_REFINE_FUSION:
                final_logits = self.refine(clip_logits, cache_logits, patch_logits, template_id)
                compute_cost += 2.0

            trajectory.append((log_prob, value, action, state))

        return final_logits, trajectory, compute_cost

    def build_imitation_targets(self, image_features: torch.Tensor, patch_features: torch.Tensor, labels: torch.Tensor):
        clip_logits = 100.0 * image_features @ self.clip_weights
        cache_logits = self._cache_logits(image_features)
        patch_logits = self._patch_logits(patch_features)
        state = self._build_state(clip_logits, cache_logits, patch_logits)

        action = self.heuristic_action(state)
        top1 = (clip_logits + self.alpha * cache_logits).argmax(dim=-1)
        verifier_label = (top1 == labels).float()
        return state.state_vector, torch.full_like(labels, action), verifier_label

    def train_imitation(self, train_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], epochs: int = 5, lr: float = 1e-3):
        params = list(self.policy.parameters()) + list(self.verifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)

        for _ in range(epochs):
            for image_features, patch_features, labels in train_batches:
                state, action_label, verifier_label = self.build_imitation_targets(image_features, patch_features, labels)
                action_logits, _, _ = self.policy(state)
                verifier_logits = self.verifier(state)

                action_loss = F.cross_entropy(action_logits, action_label)
                verifier_loss = F.binary_cross_entropy_with_logits(verifier_logits, verifier_label)
                loss = action_loss + verifier_loss
                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

    def train_reinforce(self, train_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], epochs: int = 3, lr: float = 1e-4,
                        reward_alpha: float = 0.1, reward_beta: float = 0.1, reward_gamma: float = 0.002):
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

        for _ in range(epochs):
            for image_features, patch_features, labels in train_batches:
                if image_features.shape[0] != 1:
                    raise ValueError("train_reinforce requires seq_batch_size=1 for per-sample episodes.")
                init_logits = 100.0 * image_features @ self.clip_weights + self.alpha * self._cache_logits(image_features)
                init_top2 = init_logits.topk(k=2, dim=-1).values
                init_margin = init_top2[:, 0] - init_top2[:, 1]

                final_logits, trajectory, compute_cost = self.forward_episode(image_features, patch_features, training=True)
                preds = final_logits.argmax(dim=-1)
                correct = (preds == labels).float()

                final_top2 = final_logits.topk(k=2, dim=-1).values
                margin_gain = (final_top2[:, 0] - final_top2[:, 1]) - init_margin

                trust = torch.sigmoid(self.verifier(trajectory[-1][3].state_vector)).detach()
                trust_target = correct
                q_gain = 1.0 - (trust - trust_target).abs()

                reward = correct + reward_alpha * margin_gain + reward_beta * q_gain - reward_gamma * compute_cost
                reward = reward.detach()

                policy_loss = 0.0
                value_loss = 0.0
                for log_prob, value, _, _ in trajectory:
                    advantage = reward - value.detach()
                    policy_loss = policy_loss + (-log_prob * advantage).mean()
                    value_loss = value_loss + F.mse_loss(value, reward)

                loss = policy_loss + 0.5 * value_loss
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                optimizer.step()

    @torch.no_grad()
    def predict(self, image_features: torch.Tensor, patch_features: torch.Tensor):
        logits = []
        for i in range(image_features.shape[0]):
            final_logits, _, _ = self.forward_episode(
                image_features[i : i + 1], patch_features[i : i + 1], training=False
            )
            logits.append(final_logits)
        return torch.cat(logits, dim=0)
"""Dense selection among four language-depth responses, memory, and no update."""
from __future__ import annotations

import torch
from torch import nn

from model_settings import normalize_router_config


ROUTING_SOURCES = ("text_1", "text_2", "text_3", "text_4", "memory", "none")
ROUTING_INTERVENTIONS = ("uniform", "fixed", "image_mean", "shuffle", "none")


class JointGuidanceRouter(nn.Module):
    """Route already projected/scaled responses [B,N,6,C] into one residual.

    Dynamic scores depend on the visual token and its candidate response, which
    contains text or previous-stage context. All candidates are computed densely.
    """

    def __init__(self, dim: int, settings: dict, assigned_group: int) -> None:
        super().__init__()
        self.settings = normalize_router_config(settings)
        self.mode = self.settings["mode"]
        if self.mode == "off":
            raise ValueError("An off router should not be instantiated.")
        if assigned_group not in range(4):
            raise ValueError("assigned_group must be in [0, 3].")
        self.assigned_group = assigned_group
        self.visual_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.candidate_norm = nn.LayerNorm(dim, elementwise_affine=False)
        if self.mode in {"static", "image", "region"}:
            self.source_bias = nn.Parameter(torch.zeros(6))
        if self.mode in {"image", "region"}:
            hidden = self.settings["hidden_dim"]
            self.visual_proj = nn.Linear(dim, hidden)
            self.candidate_proj = nn.Linear(dim, hidden, bias=False)
            self.score = nn.Linear(hidden, 1, bias=False)
            self.activation = nn.GELU()
            # Near-uniform, input-dependent initialization with gradients to both projections.
            nn.init.normal_(self.score.weight, std=0.02)

    def _available(self, device, has_memory: bool) -> torch.Tensor:
        available = torch.ones(6, device=device, dtype=torch.bool)
        if self.settings["language"] == "assigned":
            available[:4] = False
            available[self.assigned_group] = True
        available[4] = has_memory and self.settings["memory"] == "selective"
        available[5] = self.settings["allow_none"]
        return available

    def _fixed_weights(self, reference, available):
        weights = torch.zeros_like(reference)
        weights[..., self.assigned_group] = 1.0
        weights[..., 4] = available[4].to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True)

    def forward(self, visual, candidates, has_memory: bool, override=None,
                return_diagnostics: bool = False):
        if candidates.shape != (*visual.shape[:2], 6, visual.shape[-1]):
            raise ValueError("Router candidates must have shape [B,N,6,C].")
        if override is not None:
            if self.training:
                raise ValueError("Routing interventions are evaluation-only.")
            if override not in ROUTING_INTERVENTIONS:
                raise ValueError(f"Unknown routing intervention: {override}")
        available = self._available(visual.device, has_memory)
        batch, tokens, _ = visual.shape
        if self.mode in {"image", "region"}:
            v = self.visual_norm(visual)
            c = self.candidate_norm(candidates)
            if self.mode == "image":
                v = v.mean(dim=1, keepdim=True)
                c = c.mean(dim=1, keepdim=True)
            features = self.visual_proj(v).unsqueeze(2) + self.candidate_proj(c)
            logits = self.score(self.activation(features)).squeeze(-1) + self.source_bias
        elif self.mode == "static":
            logits = self.source_bias.view(1, 1, 6)
        else:
            logits = visual.new_zeros(1, 1, 6)
        logits = logits.expand(batch, tokens, 6).float() / self.settings["temperature"]
        weights = logits.masked_fill(~available, float("-inf")).softmax(dim=-1)
        if self.mode == "fixed" or override == "fixed":
            weights = self._fixed_weights(weights, available)
        if override == "uniform":
            weights = available.to(weights.dtype).expand_as(weights) / available.sum()
        elif override == "image_mean":
            weights = weights.mean(dim=1, keepdim=True).expand_as(weights)
        elif override == "shuffle":
            weights = torch.stack([w[torch.randperm(tokens, device=w.device)] for w in weights])
        elif override == "none":
            if not self.settings["allow_none"]:
                raise ValueError("The none intervention requires allow_none=True.")
            weights = torch.zeros_like(weights)
            weights[..., 5] = 1.0

        # Enforce zero responses for unavailable candidates, including the none branch.
        valid_response = available.clone()
        valid_response[5] = False
        responses = candidates * valid_response.view(1, 1, 6, 1)
        contributions = responses * weights.to(candidates.dtype).unsqueeze(-1)
        if has_memory and self.settings["memory"] == "always":
            # This control keeps the full parallel memory residual outside the simplex.
            contributions = contributions.clone()
            contributions[..., 4, :] = candidates[..., 4, :]
        contributions = contributions * self.settings["residual_scale"]
        update = contributions.sum(dim=2)
        if return_diagnostics:
            return update, weights.detach(), contributions.detach().float().norm(dim=-1)
        return update, None, None

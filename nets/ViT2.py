from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from model_settings import normalize_router_config
from .routing import JointGuidanceRouter

try:
    from timm.layers import PatchEmbed, Mlp, DropPath
except ImportError:
    from timm.models.layers import PatchEmbed, Mlp, DropPath


def _safe_heads(dim: int, requested: int) -> int:
    """Return largest divisor of dim that is <= requested."""
    for h in range(requested, 0, -1):
        if dim % h == 0:
            return h
    return 1


class LayerScale(nn.Module):
    """Per-channel learnable scaling of a residual branch (CaiT).

    init_values follows CaiT: ~0.1 for shallow stacks (depth <= 18), smaller
    (1e-5/1e-6) only for very deep stacks. The branch output is multiplied by
    gamma before being added back to the residual stream.
    """

    def __init__(self, dim: int, init_values: float = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class AdaptiveGatedFusion(nn.Module):
    """Adaptive channel-wise fusion for visual and text-enhanced tokens.

    This is the token-domain equivalent of:
        G = sigmoid(Conv1x1(concat(V, V_text)))
        F = G * V_text + (1 - G) * V
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, visual: torch.Tensor, text_enhanced: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([visual, text_enhanced], dim=-1)))
        return gate * text_enhanced + (1.0 - gate) * visual


class FrequencyModule(nn.Module):
    """Learnable frequency-domain residual enhancement (improved LFM).

    Splits the feature map into low/high frequency parts via a Gaussian on the
    fft-shifted spectrum (DC at center), then adds a signed, per-channel mix of the
    high-frequency component back to the input:  out = x + gamma * (x - lowpass(x)).
    gamma>0 sharpens edges, gamma<0 smooths; gamma init 0 -> starts as identity.
    sigma = softplus(sigma_raw) stays positive and is learned per stage.

    Fixes vs the reference LFM: the reference applied its Gaussian on the *unshifted*
    spectrum (center = highest freq, not DC), so it silently low-passed instead of
    enhancing edges. fftshift makes the high/low split well-defined and learnable.
    """

    def __init__(self, channels: int, img_size: int, sigma_init: float = 5.0) -> None:
        super().__init__()
        # softplus(sigma_raw) == sigma_init at init; softplus keeps sigma > 0 always.
        self.sigma_raw = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(float(sigma_init))))
        )
        self.gamma = nn.Parameter(torch.zeros(channels))
        # Squared distance from the (post-fftshift) DC center; cached for the common
        # square case, recomputed on the fly if a stage ever sees a different H,W.
        self.register_buffer(
            "dist2", self._make_dist2(img_size, img_size), persistent=False
        )

    @staticmethod
    def _make_dist2(h: int, w: int, device=None) -> torch.Tensor:
        yy = torch.arange(h, device=device).float() - h // 2
        xx = torch.arange(w, device=device).float() - w // 2
        d2 = yy[:, None] ** 2 + xx[None, :] ** 2
        return d2.view(1, 1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if tuple(self.dist2.shape[-2:]) == (H, W):
            dist2 = self.dist2
        else:
            dist2 = self._make_dist2(H, W, x.device)

        in_dtype = x.dtype
        xf = x.float()                                   # FFT needs float (AMP-safe)
        Y = torch.fft.fft2(xf)
        Yc = torch.fft.fftshift(Y, dim=(-2, -1))         # DC -> center
        sigma = F.softplus(self.sigma_raw)
        g = torch.exp(-dist2 / (2.0 * sigma ** 2))       # low-pass mask, [1,1,H,W]
        low = torch.fft.ifft2(
            torch.fft.ifftshift(Yc * g, dim=(-2, -1))
        ).real
        high = xf - low                                   # exact complement
        out = xf + self.gamma.view(1, C, 1, 1) * high
        return out.to(in_dtype)


class CascadeViTBlock(nn.Module):
    """One transformer block: TextCrossAttn+GatedFusion -> PrevCrossAttn -> SelfAttn -> FFN.

    Stage 1 passes prev_dim=None; prev_cross_attn is skipped when prev_tokens is None.
    Text stays at full 768-dim; kdim/vdim handle the projection internally.

    The text branch builds a text-enhanced candidate and adaptively gates it with
    the original visual tokens. Other residual branches use
    x = x + DropPath(LayerScale(branch(PreNorm(x)))). The attention branches
    additionally apply a NormFormer-style post-norm before LayerScale; the FFN
    branch gets LayerScale only.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        prev_dim: int | None = None,
        text_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.1,
        use_prev_cross_attention: bool = True,
        use_adaptive_gated_fusion: bool = True,
        router_config: dict | None = None,
        text_group_index: int = 0,
    ) -> None:
        super().__init__()
        self.use_prev_cross_attention = use_prev_cross_attention
        self.use_adaptive_gated_fusion = use_adaptive_gated_fusion
        routing = normalize_router_config(router_config)
        if routing["mode"] != "off" and use_adaptive_gated_fusion:
            raise ValueError("Joint routing and legacy adaptive gated fusion are alternative guidance paths.")
        safe = _safe_heads(embed_dim, num_heads)
        if safe != num_heads:
            warnings.warn(
                f"num_heads={num_heads} does not divide embed_dim={embed_dim}; "
                f"reduced to {safe}.",
                UserWarning,
                stacklevel=2,
            )

        # 1. Text cross-attention branch
        self.norm_text = nn.LayerNorm(embed_dim)
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=safe,
            kdim=text_dim, vdim=text_dim,
            dropout=dropout, batch_first=True,
        )
        self.post_norm_text = nn.LayerNorm(embed_dim)
        self.ls_text = LayerScale(embed_dim, init_values)
        self.text_gate = AdaptiveGatedFusion(embed_dim)

        # 2. Prev-stage cross-attention branch (optional)
        self.norm_prev = None
        self.prev_cross_attn = None
        self.post_norm_prev = None
        self.ls_prev = None
        if prev_dim is not None:
            self.norm_prev = nn.LayerNorm(embed_dim)
            self.prev_cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=safe,
                kdim=prev_dim, vdim=prev_dim,
                dropout=dropout, batch_first=True,
            )
            self.post_norm_prev = nn.LayerNorm(embed_dim)
            self.ls_prev = LayerScale(embed_dim, init_values)

        # 3. Self-attention branch
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=safe,
            dropout=dropout, batch_first=True,
        )
        self.post_norm_self = nn.LayerNorm(embed_dim)
        self.ls_self = LayerScale(embed_dim, init_values)

        # 4. FFN branch
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=dropout,
        )
        self.ls_ffn = LayerScale(embed_dim, init_values)

        # Stochastic depth: drop whole residual branches per-sample during training.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.router = (
            JointGuidanceRouter(embed_dim, routing, text_group_index)
            if routing["mode"] != "off" else None
        )

    def _legacy_guidance(self, x, text, prev_tokens, text_mask):
        # 1. Text cross-attention + adaptive gated fusion: Q=image, KV=text [B, 80, 768]
        visual = x
        attn_out, _ = self.text_cross_attn(
            self.norm_text(x), text, text, need_weights=False,
            key_padding_mask=(text_mask == 0) if text_mask is not None else None,
        )
        text_update = self.drop_path(self.ls_text(self.post_norm_text(attn_out)))
        if self.use_adaptive_gated_fusion:
            text_enhanced = visual + text_update
            x = self.text_gate(visual, text_enhanced)
        else:
            x = x + text_update

        # 2. Prev-stage cross-attention: Q=image, KV=prev [B, N_prev, C_prev]
        if (
            self.use_prev_cross_attention
            and self.prev_cross_attn is not None
            and prev_tokens is not None
        ):
            attn_out, _ = self.prev_cross_attn(
                self.norm_prev(x), prev_tokens, prev_tokens, need_weights=False,
            )
            x = x + self.drop_path(self.ls_prev(self.post_norm_prev(attn_out)))
        return x

    def _routed_guidance(self, x, text, prev_tokens, text_mask, override, diagnostics):
        if text.ndim != 4 or text.shape[1] != 4:
            raise ValueError("Joint routing requires four text groups [B,4,L,D].")
        if text_mask is not None and not text_mask.bool().any(dim=-1).all():
            raise ValueError("Each description needs at least one unmasked token.")
        query = self.norm_text(x)
        candidates = []
        # One shared attention module; each group has its own token-wise softmax.
        for group in text.unbind(dim=1):
            response, _ = self.text_cross_attn(
                query, group, group, need_weights=False,
                key_padding_mask=(text_mask == 0) if text_mask is not None else None,
            )
            candidates.append(self.ls_text(self.post_norm_text(response)))
        has_memory = (
            self.use_prev_cross_attention and self.prev_cross_attn is not None
            and prev_tokens is not None and self.router.settings["memory"] != "off"
        )
        memory = torch.zeros_like(x)
        if has_memory:
            response, _ = self.prev_cross_attn(
                self.norm_prev(x), prev_tokens, prev_tokens, need_weights=False,
            )
            memory = self.ls_prev(self.post_norm_prev(response))
        candidates.extend((memory, torch.zeros_like(x)))
        update, weights, norms = self.router(
            x, torch.stack(candidates, dim=2), has_memory,
            override=override, return_diagnostics=diagnostics,
        )
        # One DropPath draw for the combined guidance, shared by all routed controls.
        return x + self.drop_path(update), weights, norms

    def forward(self, x, text, prev_tokens=None, text_mask=None,
                return_routing=False, routing_override=None):
        weights = norms = None
        if self.router is None:
            if return_routing or routing_override is not None:
                raise ValueError("Routing diagnostics/interventions require an enabled router.")
            x = self._legacy_guidance(x, text, prev_tokens, text_mask)
        else:
            x, weights, norms = self._routed_guidance(
                x, text, prev_tokens, text_mask, routing_override, return_routing,
            )

        # 3. Self-attention
        normed = self.norm_self(x)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        x = x + self.drop_path(self.ls_self(self.post_norm_self(attn_out)))

        # 4. FFN
        x = x + self.drop_path(self.ls_ffn(self.ffn(self.norm_ffn(x))))

        return (x, weights, norms) if return_routing else x


class CascadeViTStage(nn.Module):
    """One stage of the cascade: PatchEmbed -> pos_embed -> depth x CascadeViTBlock -> LayerNorm.

    patch_size=1 keeps tokens at ConvNeXt's native spatial resolution.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        prev_dim: int | None = None,
        text_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float | list[float] = 0.0,
        init_values: float = 0.1,
        use_freq: bool = True,
        freq_sigma_init: float = 5.0,
        use_checkpoint: bool = False,
        use_prev_cross_attention: bool = True,
        use_adaptive_gated_fusion: bool = True,
        router_config: dict | None = None,
        text_group_index: int = 0,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Frequency enhancement on the spatial feature map, before tokenization.
        self.freq = (
            FrequencyModule(in_channels, img_size, freq_sigma_init)
            if use_freq else None
        )

        if isinstance(drop_path, (list, tuple)):
            if len(drop_path) != depth:
                raise ValueError(
                    f"drop_path list length {len(drop_path)} != depth {depth}"
                )
            dpr = list(drop_path)
        else:
            dpr = [drop_path] * depth

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=1,
            in_chans=in_channels, embed_dim=embed_dim,
            strict_img_size=False,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.train_grid_size = img_size  # square; remembered for pos_embed interpolation

        self.blocks = nn.ModuleList([
            CascadeViTBlock(
                embed_dim=embed_dim, num_heads=num_heads,
                prev_dim=prev_dim, text_dim=text_dim,
                mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[i],
                init_values=init_values,
                use_prev_cross_attention=use_prev_cross_attention,
                use_adaptive_gated_fusion=use_adaptive_gated_fusion,
                router_config=router_config, text_group_index=text_group_index,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_embed(self, pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate pos_embed [1, N, C] to a new (h, w) grid via bilinear resampling."""
        _, N, C = pos_embed.shape
        g = self.train_grid_size
        if N != g * g:
            raise RuntimeError(
                f"pos_embed token count {N} doesn't match train grid {g}x{g}={g*g}"
            )
        pe = pos_embed.reshape(1, g, g, C).permute(0, 3, 1, 2)   # [1, C, g, g]
        pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, h * w, C)

    def forward(
        self,
        f: torch.Tensor,
        text: torch.Tensor,
        prev_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_routing: bool = False,
        routing_override: str | None = None,
    ) -> torch.Tensor:
        if self.freq is not None:
            f = self.freq(f)
        x = self.patch_embed(f)
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            # Variable input size (e.g., multi-scale TTA): bilinearly interpolate pos_embed.
            h, w = f.shape[-2:]
            x = x + self._interpolate_pos_embed(self.pos_embed, h, w)
        weights, norms = [], []
        for blk in self.blocks:
            if self.use_checkpoint:
                result = grad_checkpoint(
                    blk, x, text, prev_tokens, text_mask, return_routing, routing_override,
                    use_reentrant=False,
                )
            else:
                result = blk(x, text, prev_tokens, text_mask, return_routing, routing_override)
            if return_routing:
                x, w, n = result
                weights.append(w)
                norms.append(n)
            else:
                x = result
        x = self.norm(x)
        if return_routing:
            return x, torch.stack(weights, dim=1), torch.stack(norms, dim=1)
        return x

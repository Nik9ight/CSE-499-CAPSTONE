from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrUpBlock

from .ViT2 import CascadeViTStage
from model_settings import normalize_router_config

try:
    import timm
except ImportError:
    timm = None

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None


def _lib_dir() -> str:
    try:
        import Config as _cfg
        return getattr(_cfg, "lib_dir", "./lib/")
    except Exception:
        return "./lib/"


def _default_text_encoder_name() -> str:
    try:
        import Config as _cfg
        return getattr(_cfg, "text_encoder", "microsoft/BiomedVLP-CXR-BERT-specialized")
    except Exception:
        return "microsoft/BiomedVLP-CXR-BERT-specialized"


def _ensure_hf_text_model_local(model_name: str, lib_dir: str) -> str:
    basename = model_name.split("/")[-1]
    local_dir = os.path.join(lib_dir, basename)
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        from huggingface_hub import snapshot_download
        os.makedirs(local_dir, exist_ok=True)
        print(f"[lib/] downloading {model_name} -> {local_dir}")
        snapshot_download(repo_id=model_name, local_dir=local_dir)
    return local_dir


class TextEncoder(nn.Module):
    """Frozen HuggingFace text encoder. Runs once per batch on the model's device.

    Returns the per-layer hidden states stacked as [B, num_layers, L, H] so the
    downstream HierarchicalTextFusion can group BERT layers across ViT stages.
    """

    def __init__(self, model_name: str, lib_dir: str = "./lib/") -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for the in-model TextEncoder.")
        local_path = _ensure_hf_text_model_local(model_name, lib_dir)
        self.model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bert_core = getattr(self.model, "bert", self.model)
        out = bert_core(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
        # hidden_states: tuple of (num_layers + 1) tensors [B, L, H]; index 0 is the
        # non-contextual embedding layer, so drop it and stack the transformer layers.
        return torch.stack(out.hidden_states[1:], dim=1)  # [B, num_layers, L, H]


class HierarchicalTextFusion(nn.Module):
    """Split BERT transformer layers into ordered groups, one per ViT stage.

    For BERT-base this maps layers 1-3 -> stage1, 4-6 -> stage2,
    7-9 -> stage3, and 10-12 -> stage4. Within each group the layers are
    mixed via a learnable softmax weight.
    """

    def __init__(self, num_layers: int = 12, num_groups: int = 4) -> None:
        super().__init__()
        if num_layers % num_groups != 0:
            raise ValueError(
                f"num_layers ({num_layers}) must be divisible by num_groups ({num_groups})."
            )
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.group_size = num_layers // num_groups
        self.alpha = nn.Parameter(torch.zeros(num_groups, self.group_size))

    def forward(self, hidden_stack: torch.Tensor):
        # hidden_stack: [B, num_layers, L, H], ordered layer 1..num_layers
        B, N, L, H = hidden_stack.shape
        if N != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {N}.")
        groups = hidden_stack.reshape(B, self.num_groups, self.group_size, L, H)
        weights = torch.softmax(self.alpha, dim=1).to(hidden_stack.dtype)
        mixed = torch.einsum("bgklh,gk->bglh", groups, weights)  # [B, num_groups, L, H]
        return mixed.unbind(dim=1)


def get_backbone_features(
    backbone_name: str = "convnext_small",
    pretrained: bool = True,
    in_chans: int = 3,
    out_indices: tuple = (0, 1, 2, 3),
):
    """Build a timm features-only backbone, caching weights under lib_dir on first run."""
    if timm is None:
        raise ImportError("timm is required.")

    if not pretrained:
        model = timm.create_model(
            backbone_name, features_only=True, pretrained=False,
            in_chans=in_chans, out_indices=out_indices
        )
    else:
        local_dir = os.path.join(_lib_dir(), backbone_name)
        local_pth = os.path.join(local_dir, "pytorch_model.bin")
        if os.path.exists(local_pth):
            model = timm.create_model(
                backbone_name, features_only=True, pretrained=False,
                in_chans=in_chans, out_indices=out_indices
            )
            state = torch.load(local_pth, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        else:
            print(f"[lib/] downloading {backbone_name} via timm -> {local_pth}")
            model = timm.create_model(
                backbone_name, features_only=True, pretrained=True,
                in_chans=in_chans, out_indices=out_indices
            )
            os.makedirs(local_dir, exist_ok=True)
            torch.save(model.state_dict(), local_pth)

    channels = [f["num_chs"] for f in model.feature_info.info]
    return model, channels


def tokens_to_spatial(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """tokens [B, N, C] -> spatial map [B, C, H, W]."""
    B, _N, C = tokens.shape
    return tokens.permute(0, 2, 1).reshape(B, C, h, w)


def pool_text_tokens(tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool text tokens into one vector per sample, respecting padding when available."""
    if attention_mask is None:
        return tokens.mean(dim=1)
    mask = attention_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / denom


class ASPP2D(nn.Module):
    """ASPP with dilations tuned for 14x14 feature maps (stage-3 skip).

    Effective receptive fields with dilations (1, 3, 6, 9): 3, 7, 13, 19 — genuinely
    different scales spanning local to near-global. Padding equal to dilation keeps
    output spatial size equal to input.
    """

    def __init__(self, in_ch: int = 384, out_ch: int = 384,
                 dilations=(1, 3, 6, 9), dropout: float = 0.1) -> None:
        super().__init__()
        branches = [nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )]
        for d in dilations:
            branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.branches = nn.ModuleList(branches)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        n_branches = len(self.branches) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        gp = F.interpolate(self.gap(x), size=(H, W),
                           mode="bilinear", align_corners=False)
        outs.append(gp)
        return self.project(torch.cat(outs, dim=1))


class LViTN(nn.Module):
    """Cascade ViT segmentation model.

    Forward returns a dict with raw segmentation logits under "main" [B,1,H,W].
    Text may be tokenizer output, per-layer hidden states, or legacy embeddings.
    Precomputed text can supply text_mask [B,L] to exclude padded positions.
    Sigmoid is applied in the loss or inference, not here.
    config is retained for API compatibility and is not currently used.
    """

    def __init__(
        self,
        config,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        backbone_name: str = "convnext_small",
        backbone_pretrained: bool = True,
        vit_depth: int = 3,                         # default changed from 4
        vit_heads: int = 8,
        text_embed_dim: int = 768,
        text_encoder_name: str | None = None,
        bert_num_layers: int = 12,
        num_text_groups: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,                # default changed from 0.1
        layerscale_init: float = 0.1,
        use_freq: bool = False,
        freq_sigma_init: float = 5.0,
        use_checkpoint: bool = False,
        use_prev_cross_attention: bool = True,
        use_adaptive_gated_fusion: bool = True,
        # New (Dice-first hybrid):
        deep_supervision: bool = True,
        aspp_in_out_ch: int | None = None,
        aspp_dilations=(1, 3, 6, 9),
        aspp_dropout: float = 0.1,
        router_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        self.text_embed_dim = text_embed_dim
        self.router_config = normalize_router_config(router_config)
        if self.router_config["mode"] != "off" and num_text_groups != 4:
            raise ValueError("Joint routing requires exactly four language groups.")

        resolved_text_encoder = text_encoder_name or _default_text_encoder_name()
        self.text_encoder_name = resolved_text_encoder
        self.text_encoder = TextEncoder(model_name=resolved_text_encoder, lib_dir=_lib_dir())
        self.text_fusion = HierarchicalTextFusion(
            num_layers=bert_num_layers, num_groups=num_text_groups,
        )

        self.backbone, channels = get_backbone_features(
            backbone_name, pretrained=backbone_pretrained, in_chans=n_channels,
            out_indices=(0, 1, 2, 3),
        )
        if len(channels) < 4:
            raise RuntimeError("Backbone must expose 4 feature stages.")
        ch1, ch2, ch3, ch4 = channels
        if not (ch2 == ch1 * 2 and ch3 == ch2 * 2 and ch4 == ch3 * 2):
            raise RuntimeError(
                f"LViTN decoder requires backbone channels to double each stage "
                f"(got {channels}). Use a ConvNeXt-style backbone."
            )
        self.peal_visual_projs = nn.ModuleList([
            nn.Linear(ch1, text_embed_dim),
            nn.Linear(ch2, text_embed_dim),
            nn.Linear(ch3, text_embed_dim),
            nn.Linear(ch4, text_embed_dim),
        ])

        # Global stochastic-depth schedule: linearly increasing drop rate across
        # all 4*vit_depth blocks, sliced per stage (standard ViT recipe).
        dpr = torch.linspace(0.0, drop_path_rate, 4 * vit_depth).tolist()

        self.stage1 = CascadeViTStage(
            img_size=img_size // 4, in_channels=ch1, embed_dim=ch1,
            depth=vit_depth, num_heads=vit_heads, prev_dim=None,
            text_dim=text_embed_dim, dropout=drop_rate, init_values=layerscale_init,
            use_freq=use_freq, freq_sigma_init=freq_sigma_init,
            drop_path=dpr[0:vit_depth], use_checkpoint=use_checkpoint,
            use_prev_cross_attention=use_prev_cross_attention,
            use_adaptive_gated_fusion=use_adaptive_gated_fusion,
            router_config=self.router_config, text_group_index=0,
        )
        self.stage2 = CascadeViTStage(
            img_size=img_size // 8, in_channels=ch2, embed_dim=ch2,
            depth=vit_depth, num_heads=vit_heads, prev_dim=ch1,
            text_dim=text_embed_dim, dropout=drop_rate, init_values=layerscale_init,
            use_freq=use_freq, freq_sigma_init=freq_sigma_init,
            drop_path=dpr[vit_depth:2 * vit_depth], use_checkpoint=use_checkpoint,
            use_prev_cross_attention=use_prev_cross_attention,
            use_adaptive_gated_fusion=use_adaptive_gated_fusion,
            router_config=self.router_config, text_group_index=1,
        )
        self.stage3 = CascadeViTStage(
            img_size=img_size // 16, in_channels=ch3, embed_dim=ch3,
            depth=vit_depth, num_heads=vit_heads, prev_dim=ch2,
            text_dim=text_embed_dim, dropout=drop_rate, init_values=layerscale_init,
            use_freq=use_freq, freq_sigma_init=freq_sigma_init,
            drop_path=dpr[2 * vit_depth:3 * vit_depth], use_checkpoint=use_checkpoint,
            use_prev_cross_attention=use_prev_cross_attention,
            use_adaptive_gated_fusion=use_adaptive_gated_fusion,
            router_config=self.router_config, text_group_index=2,
        )
        self.stage4 = CascadeViTStage(
            img_size=img_size // 32, in_channels=ch4, embed_dim=ch4,
            depth=vit_depth, num_heads=vit_heads, prev_dim=ch3,
            text_dim=text_embed_dim, dropout=drop_rate, init_values=layerscale_init,
            use_freq=use_freq, freq_sigma_init=freq_sigma_init,
            drop_path=dpr[3 * vit_depth:4 * vit_depth], use_checkpoint=use_checkpoint,
            use_prev_cross_attention=use_prev_cross_attention,
            use_adaptive_gated_fusion=use_adaptive_gated_fusion,
            router_config=self.router_config, text_group_index=3,
        )

        # ASPP on the stage-3 spatial skip. Channels match ch3 (384 for ConvNeXt-Tiny/Small)
        # so decoder3's skip input shape is unchanged.
        aspp_ch = aspp_in_out_ch if aspp_in_out_ch is not None else ch3
        if aspp_ch != ch3:
            raise ValueError(
                f"ASPP in/out channels ({aspp_ch}) must match stage-3 channels ({ch3})"
            )
        self.aspp = ASPP2D(
            in_ch=ch3, out_ch=ch3,
            dilations=tuple(aspp_dilations), dropout=aspp_dropout,
        )
        # UNETR decoder -- monai UnetrUpBlock with residual conv blocks.
        # Requires skip channels == out_channels (holds for ConvNeXt: ch_i = ch_{i-1}*2).
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2, in_channels=ch4, out_channels=ch4 // 2,
            kernel_size=3, upsample_kernel_size=2, norm_name="batch", res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2, in_channels=ch4 // 2, out_channels=ch3 // 2,
            kernel_size=3, upsample_kernel_size=2, norm_name="batch", res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=2, in_channels=ch3 // 2, out_channels=ch2 // 2,
            kernel_size=3, upsample_kernel_size=2, norm_name="batch", res_block=True,
        )

        # --- High-resolution tail ---
        # The backbone's finest feature is at H/4, so there are no native skips at
        # H/2 or H/1. A shallow stem on the raw input supplies those detail skips,
        # and two learned 2x upsampling blocks (H/4 -> H/2 -> H) replace the old
        # fixed 4x bilinear so the network can reconstruct sharp boundaries.
        h2_ch = ch1 // 2
        h1_ch = ch1 // 4

        self.stem_h1 = nn.Sequential(
            nn.Conv2d(n_channels, h1_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h1_ch),
            nn.ReLU(inplace=True),
        )
        self.stem_h2 = nn.Sequential(
            nn.Conv2d(h1_ch, h2_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h2_ch),
            nn.ReLU(inplace=True),
        )

        self.up2 = UnetrUpBlock(
            spatial_dims=2, in_channels=ch2 // 2, out_channels=h2_ch,
            kernel_size=3, upsample_kernel_size=2, norm_name="batch", res_block=True,
        )
        self.up1 = UnetrUpBlock(
            spatial_dims=2, in_channels=h2_ch, out_channels=h1_ch,
            kernel_size=3, upsample_kernel_size=2, norm_name="batch", res_block=True,
        )
        self.seg_head = nn.Conv2d(h1_ch, n_classes, kernel_size=1)
        # Focal-style bias prior: start predicting low foreground probability so the
        # small-lesion class isn't drowned out at init. pi is the assumed foreground
        # fraction; tune toward the dataset's positive ratio if known.
        prior = 0.01
        nn.init.constant_(self.seg_head.bias, -math.log((1.0 - prior) / prior))
        # Auxiliary heads for deep supervision (training only).
        # d1 has ch2//2 channels at H/4; d2 has ch3//2 channels at H/8.
        self.aux_head_d1 = nn.Conv2d(ch2 // 2, n_classes, kernel_size=1)
        self.aux_head_d2 = nn.Conv2d(ch3 // 2, n_classes, kernel_size=1)
        prior = 0.01
        for h in (self.aux_head_d1, self.aux_head_d2):
            nn.init.constant_(h.bias, -math.log((1.0 - prior) / prior))

    def forward(self, x: torch.Tensor, text, text_mask=None,
                return_routing: bool = False, routing_override: str | None = None) -> dict:
        # --- Per-stage text features ---
        # Accept either tokenizer output (dict) for in-model BERT, a pre-computed
        # 4D layer stack [B, num_layers, L, H], or a 3D legacy embedding [B, L, H]
        # which is broadcast unchanged to every stage.
        if isinstance(text, dict):
            text_mask = text["attention_mask"]
            hidden_stack = self.text_encoder(text["input_ids"], text["attention_mask"])
            tx1, tx2, tx3, tx4 = self.text_fusion(hidden_stack)
        elif torch.is_tensor(text):
            if text.dim() == 4:
                tx1, tx2, tx3, tx4 = self.text_fusion(text)
            elif text.dim() == 3:
                if self.router_config["mode"] != "off":
                    raise ValueError("Language-depth routing needs tokenizer output or a BERT layer stack.")
                tx1 = tx2 = tx3 = tx4 = text
            else:
                raise ValueError(
                    f"Expected text shaped [B, L, H] or [B, num_layers, L, H], "
                    f"got {tuple(text.shape)}."
                )
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

        # --- Backbone ---
        f1, f2, f3, f4 = self.backbone(x)

        # PEAL keeps its original stage/group targets; routing selects guidance separately.
        text_groups = (tx1, tx2, tx3, tx4)
        all_text = torch.stack(text_groups, dim=1) if self.router_config["mode"] != "off" else None
        tokens, routing = [], {}
        previous = None
        for index, (stage, feature, stage_text) in enumerate(zip(
            (self.stage1, self.stage2, self.stage3, self.stage4),
            (f1, f2, f3, f4), text_groups,
        )):
            result = stage(
                feature, all_text if all_text is not None else stage_text,
                prev_tokens=previous, text_mask=text_mask,
                return_routing=return_routing, routing_override=routing_override,
            )
            if return_routing:
                previous, weights, norms = result
                b, depth = weights.shape[:2]
                h, w = feature.shape[-2:]
                routing[f"stage{index + 1}"] = {
                    "weights": weights.reshape(b, depth, h, w, 6),
                    "contribution_norms": norms.reshape(b, depth, h, w, 6),
                }
            else:
                previous = result
            tokens.append(previous)
        t1, t2, t3, t4 = tokens

        peal_visual = tuple(
            proj(tokens.mean(dim=1))
            for proj, tokens in zip(self.peal_visual_projs, (t1, t2, t3, t4))
        )
        peal_text = tuple(
            pool_text_tokens(stage_text, text_mask)
            for stage_text in (tx1, tx2, tx3, tx4)
        )

        # --- Token -> spatial ---
        s4 = tokens_to_spatial(t4, *f4.shape[-2:])
        s3 = tokens_to_spatial(t3, *f3.shape[-2:])
        s2 = tokens_to_spatial(t2, *f2.shape[-2:])
        s1 = tokens_to_spatial(t1, *f1.shape[-2:])

        # --- ASPP on stage-3 skip (multi-scale aggregation at 14x14) ---
        s3 = self.aspp(s3)

        # --- UNETR decoder ---
        d3 = self.decoder3(s4, s3)
        d2 = self.decoder2(d3, s2)
        d1 = self.decoder1(d2, s1)          # [B, ch2//2, H/4, W/4]

        # --- High-res tail: learned upsampling fused with input-stem skips ---
        e1 = self.stem_h1(x)                # [B, h1_ch, H, W]
        e2 = self.stem_h2(e1)               # [B, h2_ch, H/2, W/2]
        u2 = self.up2(d1, e2)               # [B, h2_ch, H/2, W/2]
        u1 = self.up1(u2, e1)               # [B, h1_ch, H, W]

        main = self.seg_head(u1)
        out = {"main": main, "peal_visual": peal_visual, "peal_text": peal_text}
        if return_routing:
            out["routing"] = routing

        if self.training and self.deep_supervision:
            aux1 = self.aux_head_d1(d1)
            aux2 = self.aux_head_d2(d2)
            H, W = x.shape[-2:]
            aux1 = F.interpolate(aux1, size=(H, W), mode="bilinear", align_corners=False)
            aux2 = F.interpolate(aux2, size=(H, W), mode="bilinear", align_corners=False)
            out.update({"aux1": aux1, "aux2": aux2})
        return out

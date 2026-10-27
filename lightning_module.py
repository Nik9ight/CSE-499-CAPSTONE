import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.losses import DiceFocalLoss
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

import Config as config
from nets.LViT_new import LViTN
from model_settings import model_kwargs_from_config, normalize_router_config


class LViTLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config=None,
        backbone_lr: float | None = None,
        head_lr: float | None = None,
        weight_decay: float = 1e-4,
        freeze_backbone_epochs: int | None = None,
        warmup_epochs: int | None = None,
        model_kwargs: dict | None = None,
        backbone_pretrained: bool = True,
        peal_loss_weight: float | None = None,
        deep_supervision_loss_weight: float | None = None,
        aux_loss_weights: tuple | None = None,
        max_epochs: int | None = None,
    ):
        super().__init__()

        self.model_kwargs = dict(model_kwargs) if model_kwargs is not None else model_kwargs_from_config(config)
        self.model_kwargs["router_config"] = normalize_router_config(self.model_kwargs.get("router_config"))
        self.model = LViTN(
            model_config,
            backbone_pretrained=backbone_pretrained,
            **self.model_kwargs,
        )

        self.seg_loss = DiceFocalLoss(
            sigmoid=True,
            lambda_dice=1.0,
            lambda_focal=1.0,
            gamma=2.0,
        )
        self.peal_loss_weight = peal_loss_weight if peal_loss_weight is not None else getattr(config, "peal_loss_weight", 0.1)
        self.deep_supervision_loss_weight = deep_supervision_loss_weight if deep_supervision_loss_weight is not None else getattr(config, "deep_supervision_loss_weight", 0.4)
        self.aux_loss_weights = tuple(aux_loss_weights if aux_loss_weights is not None else config.aux_loss_weights)
        self.max_epochs = max_epochs if max_epochs is not None else config.epochs

        # Head LR defaults to Config.learning_rate; backbone fine-tunes at 0.1x.
        self.head_lr = head_lr if head_lr is not None else config.learning_rate
        self.backbone_lr = backbone_lr if backbone_lr is not None else self.head_lr * 0.1
        self.weight_decay = weight_decay
        self.freeze_backbone_epochs = freeze_backbone_epochs if freeze_backbone_epochs is not None else getattr(config, "freeze_backbone_epochs", 5)
        self.warmup_epochs = warmup_epochs if warmup_epochs is not None else config.warmup_epochs

        # Log the *resolved* values (init args may be None) so the hparams tab/yaml is
        # meaningful. Keys match __init__ kwargs, keeping load_from_checkpoint valid.
        self.save_hyperparameters({
            "model_kwargs": self.model_kwargs,
            "backbone_pretrained": backbone_pretrained,
            "backbone_lr": self.backbone_lr,
            "head_lr": self.head_lr,
            "weight_decay": self.weight_decay,
            "freeze_backbone_epochs": self.freeze_backbone_epochs,
            "warmup_epochs": self.warmup_epochs,
            "peal_loss_weight": self.peal_loss_weight,
            "deep_supervision_loss_weight": self.deep_supervision_loss_weight,
            "aux_loss_weights": self.aux_loss_weights,
            "max_epochs": self.max_epochs,
        })

        self.train_dice = BinaryF1Score()
        self.train_iou = BinaryJaccardIndex()
        self.val_dice = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()

    def _seg_loss(self, logits, masks):
        return self.seg_loss(logits, masks)

    def _peal_loss(self, out, ref_tensor):
        visual_features = out.get("peal_visual")
        text_features = out.get("peal_text")
        if visual_features is None or text_features is None:
            return ref_tensor.new_zeros(())

        losses = [
            1.0 - F.cosine_similarity(visual, text, dim=1).mean()
            for visual, text in zip(visual_features, text_features)
        ]
        return torch.stack(losses).mean()

    def _shared_step(self, batch):
        sampled_batch, _names = batch
        images = sampled_batch["image"]
        masks = sampled_batch["label"]

        text = None
        if "input_ids" in sampled_batch and "attention_mask" in sampled_batch:
            text = {
                "input_ids": sampled_batch["input_ids"],
                "attention_mask": sampled_batch["attention_mask"],
            }
        elif "text" in sampled_batch:
            text = sampled_batch["text"]

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()

        if self.training and config.use_mixup_cutmix:
            raise ValueError(
                "MixUp/CutMix is unsupported for paired image/text training. "
                "Set Config.use_mixup_cutmix=False to preserve text correspondence."
            )

        out = self.model(images, text)
        main_logits = out["main"]
        main_loss = self._seg_loss(main_logits, masks)
        peal_loss = self._peal_loss(out, main_logits)

        if self.training and "aux1" in out:
            aux1_loss = self._seg_loss(out["aux1"], masks)
            aux2_loss = self._seg_loss(out["aux2"], masks)
            # config.aux_loss_weights = (w_aux2, w_aux1)
            w_aux2, w_aux1 = self.aux_loss_weights
            deep_loss = (w_aux1 * aux1_loss + w_aux2 * aux2_loss) / (w_aux1 + w_aux2)
        else:
            aux1_loss = aux2_loss = deep_loss = main_logits.new_zeros(())

        loss = (
            main_loss
            + self.peal_loss_weight * peal_loss
            + self.deep_supervision_loss_weight * deep_loss
        )
        aux_losses = {
            "main_loss": main_loss,
            "peal_loss": peal_loss,
            "deep_supervision_loss": deep_loss,
            "aux1_loss": aux1_loss,
            "aux2_loss": aux2_loss,
        }

        probs = torch.sigmoid(main_logits.detach())
        preds_bin = (probs > 0.5).long().view(-1)
        masks_bin = (masks > 0.5).long().view(-1)

        return loss, preds_bin, masks_bin, images.shape[0], aux_losses

    def training_step(self, batch, batch_idx):
        loss, preds_bin, masks_bin, bs, aux_losses = self._shared_step(batch)
        self.train_dice(preds_bin, masks_bin)
        self.train_iou(preds_bin, masks_bin)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, batch_size=bs)
        for k, v in aux_losses.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds_bin, masks_bin, bs, _aux_losses = self._shared_step(batch)
        self.val_dice(preds_bin, masks_bin)
        self.val_iou(preds_bin, masks_bin)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def on_train_epoch_start(self):
        freeze = self.current_epoch < self.freeze_backbone_epochs
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze

    def configure_optimizers(self):
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            # Frozen BERT lives under text_encoder.* and is never trained.
            if name.startswith("text_encoder."):
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.backbone_lr},
                {"params": head_params, "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )
        # Linear warmup -> cosine decay. Warmup eases the from-scratch head in and
        # keeps the LR low through the backbone unfreeze; cosine then anneals to
        # eta_min over the remaining epochs for a clean low-LR endpoint.
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

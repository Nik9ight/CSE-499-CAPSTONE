import os
import torch
import ml_collections
import time

# ======================================================
# GLOBAL SETTINGS
# ======================================================
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)

deterministic = "warn"
use_cuda = torch.cuda.is_available()

# ======================================================
# TRAIN SETTINGS
# ======================================================
mode            = 'train'      # 'train' or 'test'
task_name       = 'qata'      # dataset folder under ./data/
# model_name      = 'LViT_pretrain'
model_name      = 'LViT'
backbone        = 'convnext_tiny.in12k_ft_in1k'
# convnextv2_base.fcmae_ft_in22k_in1k
# convnext_small.in12k_ft_in1k
# convnext_tiny.in12k_ft_in1k

# Text encoder (HuggingFace repo id). Output hidden size MUST equal text_embed_dim.
text_encoder    = 'microsoft/BiomedVLP-CXR-BERT-specialized'
text_embed_dim  = 768            # 768 for BERT-base, 1024 for BERT-large, etc.
epochs          = 201
img_size        = 224
batch_size      = 32
learning_rate   = 3e-4           # Covid19 recommended 3e-4, MoNuSeg is usually 1e-3
warmup_epochs   = 5              # linear LR warmup epochs before cosine decay
freeze_backbone_epochs = 5       # epochs to freeze backbone before joint fine-tuning

# FLOPs reporting before training. The full path includes the frozen HuggingFace
# text encoder when it traces cleanly; otherwise profiling falls back to a dummy
# precomputed text feature stack and reports that text-encoder FLOPs are excluded.
show_flops      = False
flops_batch_size = batch_size
flops_include_text_encoder = True
flops_text_tokens = 80
flops_text_layers = 12

print_frequency = 100             #originally 200
vis_frequency   = 0
save_after_epoch = 1            # only start saving best after this epoch
early_stopping_patience = 20

num_workers     = 2

pretrain        = False          # use pretrained weights
cosineLR        = True
precision       = "16-mixed"    # AMP: "16-mixed" (fp16) | "bf16-mixed" (bfloat16, Ampere+ only) | "32"

tensorboard     = True
save_model      = True

# Weights & Biases (optional). use_wandb=False keeps runs offline-free; set True to
# log to W&B. wandb_offline=True logs locally without a login/account.
use_wandb       = False
wandb_project   = 'LViT-QaTa-COV19'
wandb_offline   = False

# ======================================================
# VIT CASCADE SETTINGS
# ======================================================
vit_depth              = 2       # transformer blocks per cascade stage  (was 4)
vit_heads              = 8       # attention heads per stage
vit_drop_rate          = 0.0     # dropout in attention/FFN of cascade blocks
vit_drop_path_rate     = 0.2     # max stochastic-depth rate (linear across blocks)  (was 0.1)
gradient_checkpointing = False   # set True when batch >= 16 to save memory
use_prev_cross_attention = True  # ablation: stage 2-4 residual cross-attend to previous-stage tokens
use_adaptive_gated_fusion = False  # ablation: gate text-enhanced tokens vs plain residual text update
use_freq                = True   # ablation: frequency-domain enhancement before tokenization
freq_sigma_init         = 5.0     # initial Gaussian sigma for FrequencyModule when enabled

# Joint-router prototype. Use `python train_model.py --experiment router_region`
# for the proposed model; mode="off" preserves the corrected baseline.
router_config = {
    "mode": "off",              # off, fixed, uniform, static, image, region
    "language": "all",          # all four groups or assigned stage group
    "memory": "selective",      # selective, always (outside routing), off
    "allow_none": True,
    "hidden_dim": 64,
    "temperature": 1.0,
    "residual_scale": 1.0,       # fixed multiplier after candidate LayerScale
}

# ======================================================
# ASPP BOTTLENECK (stage-3 skip)
# ======================================================
aspp_in_out_ch          = 384
aspp_dilations          = (1, 3, 6, 9)
aspp_dropout            = 0.1

# ======================================================
# DEEP SUPERVISION
# ======================================================
deep_supervision        = True
aux_loss_weights        = (0.3, 0.5)   # (aux2 @ H/8, aux1 @ H/4) inside L_deep_supervision
peal_loss_weight        = 0.2
deep_supervision_loss_weight = 0.3

# ======================================================
# MIXUP / CUTMIX AUGMENTATION (training only)
# ======================================================
use_mixup_cutmix        = False  # image-only mixing breaks paired text supervision
mixup_alpha             = 0.2
cutmix_alpha            = 1.0
mixup_cutmix_prob       = 0.5    # probability that one of MixUp/CutMix fires per batch

# ======================================================
# MULTI-SCALE TTA + THRESHOLD SWEEP (test only)
# ======================================================
use_ms_tta              = True
ms_tta_sizes            = (192, 224, 256)   # explicit sizes (must be multiples of 32)
threshold_sweep         = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

# ======================================================
# PATHS
# ======================================================
# Pretrained encoder weights (BERT + ConvNeXt) are stored under ./lib/<model_name>/
# and auto-downloaded on first run. See bertem.py and nets/LViT_new.py.
lib_dir = './lib/'

data_root = './data/' + task_name + '/'
train_dataset = data_root + 'Train/'
val_dataset = data_root + 'Val/'
test_dataset = data_root + 'Test/'
task_dataset = data_root + 'Train/'
train_text_csv = data_root + 'prompt/train.csv'
val_text_csv   = data_root + 'prompt/val.csv'
test_text_csv  = data_root + 'prompt/test.csv'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

test_session   = session_name

n_channels     = 3
n_labels       = 1   # binary mask

use_tta         = True  # enable test-time augmentation in test_model.py
test_checkpoint = None  # set a checkpoint path here or pass --checkpoint

# ======================================================
# VIT CONFIG
# ======================================================
def get_CTranS_config():
    cfg = ml_collections.ConfigDict()
    cfg.transformer = ml_collections.ConfigDict()

    cfg.KV_size = 960 # Dimension of K and V in Multi-Head Attention
    cfg.transformer.num_heads = 4
    cfg.transformer.num_layers = 4
    cfg.transformer.embeddings_dropout_rate = 0.1
    cfg.transformer.attention_dropout_rate = 0.1
    cfg.transformer.dropout_rate = 0

    cfg.base_channel = 64      # base U-Net channel
    cfg.n_classes = 1
    cfg.expand_ratio = 4
    cfg.patch_sizes = [16, 8, 4, 2]

    return cfg

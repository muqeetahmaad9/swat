"""
SegFormer (MiT-B0) for flood susceptibility mapping — complete PyTorch pipeline.

Why from scratch (no pretrained weights)?  Our input is a ~20-band geophysical
stack (SAR + optical indices + terrain), not 3-band RGB — ImageNet weights do
not transfer to band 1 = elevation, so we train the Mix Transformer from
random init and keep the model small (B0) to match a single-basin dataset.

Framing: dense segmentation. The model sees 128x128 tiles of the feature cube
and predicts a per-pixel flood-susceptibility probability. The loss is masked:
computed ONLY on pixels that are valid AND labelled (0 or 1), so unlabelled
pixels never contribute gradients.

Inputs  (produced by preprocess_pipeline.py):
    data/processed/feature_cube.npy   (H, W, C) float32, normalized
    data/processed/valid_mask.npy     (H, W)    bool
    data/labels/flood_labels.tif      1 = flood, 0 = non-flood, nodata = unknown
Outputs:
    runs/segformer/best_model.pt      best checkpoint (weights + config + metrics)
    runs/segformer/history.json       per-epoch training log
    runs/segformer/susceptibility.npy full-basin probability map (H, W)

Usage:
    pip install torch numpy rasterio
    python segformer_flood.py
"""

# ---------------------------------------------------------------------------#
# Imports                                                                     #
# ---------------------------------------------------------------------------#
import json                                    # write the training history log
import math                                    # cosine schedule & init math
import os                                      # paths / folders
import numpy as np                             # array handling
import torch                                   # core PyTorch
import torch.nn as nn                          # network layers
import torch.nn.functional as F                # functional ops (interpolate, pad)
from torch.utils.data import Dataset, DataLoader  # data pipeline classes

# ---------------------------------------------------------------------------#
# CONFIG — every experiment knob in one place                                 #
# ---------------------------------------------------------------------------#
CFG = {
    "cube_path":  "../data/processed/feature_cube.npy",   # (H, W, C) features
    "valid_path": "../data/processed/valid_mask.npy",     # (H, W) validity mask
    "label_path": "../data/labels/flood_labels.tif",      # label GeoTIFF (same grid!)
    "out_dir":    "../runs/segformer",                     # checkpoints + logs
    "tile":        128,        # training tile size in pixels
    "stride":      64,         # tile sampling stride (50% overlap)
    "min_labeled": 200,        # a tile must contain >= this many labelled pixels
    "block_px":    512,        # spatial-CV block edge for the train/val/test split
    "split_fracs": (0.7, 0.15, 0.15),  # fraction of blocks per split
    "batch_size":  16,         # tiles per batch (fits 6 GB GPU at B0/128px)
    "epochs":      100,        # maximum epochs (early stopping will cut this)
    "lr":          6e-4,       # peak AdamW learning rate (SegFormer default-ish)
    "weight_decay":0.01,       # AdamW weight decay
    "warmup_epochs":5,         # linear LR warm-up length
    "patience":    15,         # early-stopping patience (epochs without val gain)
    "num_workers": 2,          # DataLoader worker processes (0 on Windows if issues)
    "seed":        42,         # global RNG seed for reproducibility
    "aug_noise":   0.02,       # std of additive Gaussian feature noise (augment)
    # MiT-B0 architecture hyper-parameters:
    "embed_dims":  [32, 64, 160, 256],   # channels of the 4 transformer stages
    "depths":      [2, 2, 2, 2],         # blocks per stage
    "num_heads":   [1, 2, 5, 8],         # attention heads per stage
    "sr_ratios":   [8, 4, 2, 1],         # spatial-reduction ratio per stage
    "decoder_dim": 256,                  # unified channel width of the MLP decoder
}

# Fix all random seeds so runs are repeatable (paper requirement).
torch.manual_seed(CFG["seed"])                 # PyTorch CPU RNG
torch.cuda.manual_seed_all(CFG["seed"])        # PyTorch GPU RNG(s)
np.random.seed(CFG["seed"])                    # NumPy RNG

# Pick the GPU if one is available, otherwise fall back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------#
# SEGFORMER — Mix Transformer (MiT) encoder building blocks                   #
# ---------------------------------------------------------------------------#

class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding: a strided conv that downsamples while
    keeping local continuity (unlike ViT's non-overlapping patches)."""

    def __init__(self, in_ch, embed_dim, patch, stride):
        super().__init__()                                     # init parent nn.Module
        self.proj = nn.Conv2d(in_ch, embed_dim,                # conv does the embedding
                              kernel_size=patch, stride=stride,
                              padding=patch // 2)              # 'same-ish' padding keeps geometry
        self.norm = nn.LayerNorm(embed_dim)                    # normalize each token vector

    def forward(self, x):
        x = self.proj(x)                                       # (B, C, H, W) -> (B, D, H', W')
        B, D, H, W = x.shape                                   # remember the new spatial size
        x = x.flatten(2).transpose(1, 2)                       # -> (B, N, D) token sequence
        x = self.norm(x)                                       # layer-norm the tokens
        return x, H, W                                         # tokens + spatial dims for reshaping


class EfficientAttention(nn.Module):
    """Self-attention with Spatial Reduction (SR): keys/values are computed on a
    downsampled map, cutting cost from O(N^2) to O(N^2 / sr^2)."""

    def __init__(self, dim, heads, sr_ratio):
        super().__init__()                                     # init parent
        self.heads = heads                                     # number of attention heads
        self.scale = (dim // heads) ** -0.5                    # 1/sqrt(d_head) scaling
        self.q = nn.Linear(dim, dim)                           # query projection
        self.kv = nn.Linear(dim, dim * 2)                      # joint key+value projection
        self.proj = nn.Linear(dim, dim)                        # output projection
        self.sr_ratio = sr_ratio                               # spatial reduction factor
        if sr_ratio > 1:                                       # only build SR if needed
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)  # strided conv downsamples K/V
            self.norm = nn.LayerNorm(dim)                      # norm after reduction

    def forward(self, x, H, W):
        B, N, D = x.shape                                      # batch, tokens, dim
        q = (self.q(x)                                         # project queries
             .reshape(B, N, self.heads, D // self.heads)       # split into heads
             .permute(0, 2, 1, 3))                             # (B, h, N, d)
        if self.sr_ratio > 1:                                  # spatially reduce K/V?
            xr = x.transpose(1, 2).reshape(B, D, H, W)         # tokens back to a map
            xr = self.sr(xr).flatten(2).transpose(1, 2)        # downsample, re-tokenize
            xr = self.norm(xr)                                 # normalize reduced tokens
        else:
            xr = x                                             # no reduction at stage 4
        kv = (self.kv(xr)                                      # project keys+values
              .reshape(B, -1, 2, self.heads, D // self.heads)  # split K/V and heads
              .permute(2, 0, 3, 1, 4))                         # (2, B, h, N', d)
        k, v = kv[0], kv[1]                                    # unpack keys and values
        attn = (q @ k.transpose(-2, -1)) * self.scale          # scaled dot-product scores
        attn = attn.softmax(dim=-1)                            # attention weights
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)      # weighted sum, merge heads
        return self.proj(out)                                  # final linear projection


class MixFFN(nn.Module):
    """SegFormer's FFN: Linear -> 3x3 depth-wise conv (injects positional
    information, replacing positional encodings) -> GELU -> Linear."""

    def __init__(self, dim, expansion=4):
        super().__init__()                                     # init parent
        hidden = dim * expansion                               # expanded width
        self.fc1 = nn.Linear(dim, hidden)                      # expand
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1,           # 3x3 depth-wise conv
                            groups=hidden)                     # groups=channels => depthwise
        self.fc2 = nn.Linear(hidden, dim)                      # project back
        self.act = nn.GELU()                                   # smooth activation

    def forward(self, x, H, W):
        x = self.fc1(x)                                        # (B, N, hidden)
        B, N, Dh = x.shape                                     # shapes for reshaping
        x = x.transpose(1, 2).reshape(B, Dh, H, W)             # tokens -> feature map
        x = self.dw(x)                                         # depth-wise conv mixes locally
        x = x.flatten(2).transpose(1, 2)                       # map -> tokens
        x = self.act(x)                                        # nonlinearity
        return self.fc2(x)                                     # back to model dim


class TransformerBlock(nn.Module):
    """One encoder block: LN -> EfficientAttention -> residual,
    then LN -> MixFFN -> residual."""

    def __init__(self, dim, heads, sr_ratio):
        super().__init__()                                     # init parent
        self.norm1 = nn.LayerNorm(dim)                         # pre-attention norm
        self.attn = EfficientAttention(dim, heads, sr_ratio)   # efficient self-attention
        self.norm2 = nn.LayerNorm(dim)                         # pre-FFN norm
        self.ffn = MixFFN(dim)                                 # Mix-FFN

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)                 # attention + residual
        x = x + self.ffn(self.norm2(x), H, W)                  # FFN + residual
        return x                                               # updated tokens


class MiTEncoder(nn.Module):
    """4-stage hierarchical encoder producing feature maps at
    1/4, 1/8, 1/16 and 1/32 of the input resolution."""

    def __init__(self, in_ch, cfg):
        super().__init__()                                     # init parent
        dims, depths = cfg["embed_dims"], cfg["depths"]        # stage widths & depths
        heads, srs = cfg["num_heads"], cfg["sr_ratios"]        # heads & SR ratios
        self.stages = nn.ModuleList()                          # container for the 4 stages
        prev = in_ch                                           # channels entering stage 1
        for i in range(4):                                     # build each stage
            patch = 7 if i == 0 else 3                         # 7x7 stem, 3x3 afterwards
            stride = 4 if i == 0 else 2                        # /4 stem, /2 afterwards
            embed = OverlapPatchEmbed(prev, dims[i], patch, stride)  # downsampling embed
            blocks = nn.ModuleList(                            # the transformer blocks
                [TransformerBlock(dims[i], heads[i], srs[i])
                 for _ in range(depths[i])])
            norm = nn.LayerNorm(dims[i])                       # stage-final norm
            self.stages.append(nn.ModuleList([embed, blocks, norm]))  # register stage
            prev = dims[i]                                     # next stage input width

    def forward(self, x):
        feats = []                                             # multi-scale outputs
        for embed, blocks, norm in self.stages:                # run the 4 stages
            x, H, W = embed(x)                                 # embed + downsample
            for blk in blocks:                                 # run each block
                x = blk(x, H, W)                               # tokens updated in place
            x = norm(x)                                        # normalize stage output
            B, N, D = x.shape                                  # token shape
            x = x.transpose(1, 2).reshape(B, D, H, W)          # tokens -> feature map
            feats.append(x)                                    # keep for the decoder
        return feats                                           # [1/4, 1/8, 1/16, 1/32]


class SegFormerHead(nn.Module):
    """All-MLP decoder: project each stage to a common width, upsample all to
    1/4 resolution, concatenate, fuse, and predict 1 logit channel."""

    def __init__(self, cfg):
        super().__init__()                                     # init parent
        dims, dd = cfg["embed_dims"], cfg["decoder_dim"]       # stage widths, decoder width
        self.linears = nn.ModuleList(                          # per-stage 1x1 projections
            [nn.Conv2d(d, dd, 1) for d in dims])
        self.fuse = nn.Sequential(                             # fusion of the 4 projected maps
            nn.Conv2d(dd * 4, dd, 1),                          # 1x1 conv merges channels
            nn.BatchNorm2d(dd),                                # stabilize activations
            nn.ReLU(inplace=True))                             # nonlinearity
        self.classify = nn.Conv2d(dd, 1, 1)                    # final 1-channel logit map

    def forward(self, feats):
        target = feats[0].shape[2:]                            # 1/4-resolution spatial size
        ups = []                                               # upsampled projections
        for f, lin in zip(feats, self.linears):                # each stage feature map
            p = lin(f)                                         # unify channel width
            p = F.interpolate(p, size=target,                  # upsample to 1/4 res
                              mode="bilinear", align_corners=False)
            ups.append(p)                                      # collect
        x = self.fuse(torch.cat(ups, dim=1))                   # concat + fuse
        return self.classify(x)                                # (B, 1, H/4, W/4) logits


class SegFormerFlood(nn.Module):
    """Full model: MiT-B0 encoder + MLP decoder + upsample to input size."""

    def __init__(self, in_ch, cfg):
        super().__init__()                                     # init parent
        self.encoder = MiTEncoder(in_ch, cfg)                  # hierarchical encoder
        self.head = SegFormerHead(cfg)                         # all-MLP decoder

    def forward(self, x):
        size = x.shape[2:]                                     # remember input H, W
        logits = self.head(self.encoder(x))                    # 1/4-resolution logits
        return F.interpolate(logits, size=size,                # upsample to full res
                             mode="bilinear", align_corners=False)

# ---------------------------------------------------------------------------#
# DATASET — tiles from the feature cube with masked labels + augmentation     #
# ---------------------------------------------------------------------------#

def load_labels(path, shape):
    """Read the label GeoTIFF into a float array aligned with the cube:
    1 = flood, 0 = dry, NaN = unknown/nodata."""
    import rasterio                                            # local import (only used here)
    with rasterio.open(path) as src:                           # open the GeoTIFF
        lab = src.read(1).astype("float32")                    # first band as float
        if src.nodata is not None:                             # if nodata is declared...
            lab[lab == src.nodata] = np.nan                    # ...convert it to NaN
    assert lab.shape == shape, (                               # grids MUST match exactly
        f"Label grid {lab.shape} != cube grid {shape}. "
        "Re-export labels on the same grid.")
    return lab                                                 # aligned label array


def build_tiles(valid, labels, cfg, rng):
    """Slide a window over the raster, keep tiles with enough labelled pixels,
    and assign each tile to train/val/test by SPATIAL BLOCK (no leakage)."""
    H, W = valid.shape                                         # raster dimensions
    t, s = cfg["tile"], cfg["stride"]                          # tile size and stride
    corners = []                                               # accepted tile corners
    for r in range(0, H - t + 1, s):                           # slide rows
        for c in range(0, W - t + 1, s):                       # slide cols
            lab = labels[r:r + t, c:c + t]                     # label window
            n_lab = np.isfinite(lab).sum()                     # count labelled pixels
            if n_lab >= cfg["min_labeled"]:                    # enough supervision?
                corners.append((r, c))                         # keep this tile
    corners = np.array(corners)                                # to array for indexing
    # --- spatial block split: whole blocks go to one split ---
    n_bcols = math.ceil(W / cfg["block_px"])                   # blocks per row
    centers = corners + cfg["tile"] // 2                       # tile centre coordinates
    block_id = (centers[:, 0] // cfg["block_px"]) * n_bcols \
             + (centers[:, 1] // cfg["block_px"])              # block index per tile
    uniq = rng.permutation(np.unique(block_id))                # shuffle block ids
    n_tr = int(len(uniq) * cfg["split_fracs"][0])              # number of train blocks
    n_va = int(len(uniq) * cfg["split_fracs"][1])              # number of val blocks
    split_of = {}                                              # block -> split name
    for i, b in enumerate(uniq):                               # assign each block
        split_of[b] = ("train" if i < n_tr else
                       "val" if i < n_tr + n_va else "test")
    splits = {"train": [], "val": [], "test": []}              # tile corners per split
    for corner, b in zip(corners, block_id):                   # route tiles to splits
        splits[split_of[b]].append(tuple(corner))
    return splits                                              # dict of corner lists


class FloodTileDataset(Dataset):
    """Serves (features, labels, mask) tiles; applies geometric + noise
    augmentation on the training split only."""

    def __init__(self, cube, labels, valid, corners, tile, augment, noise_std):
        self.cube = cube                                       # (H, W, C) feature cube
        self.labels = labels                                   # (H, W) labels with NaN
        self.valid = valid                                     # (H, W) validity mask
        self.corners = corners                                 # list of (row, col)
        self.tile = tile                                       # tile edge length
        self.augment = augment                                 # bool: train-time aug on?
        self.noise_std = noise_std                             # feature-noise strength

    def __len__(self):
        return len(self.corners)                               # one item per tile

    def __getitem__(self, idx):
        r, c = self.corners[idx]                               # tile corner
        t = self.tile                                          # shorthand
        x = self.cube[r:r + t, c:c + t, :].copy()              # (t, t, C) features
        y = self.labels[r:r + t, c:c + t].copy()               # (t, t) labels (NaN=unknown)
        m = (self.valid[r:r + t, c:c + t] &                    # supervision mask =
             np.isfinite(y))                                   # valid AND labelled
        y = np.nan_to_num(y, nan=0.0)                          # NaN -> 0 (masked anyway)
        if self.augment:                                       # training-only augmentation
            k = np.random.randint(4)                           # 0-3 quarter turns
            x = np.rot90(x, k, axes=(0, 1)).copy()             # rotate features
            y = np.rot90(y, k).copy()                          # rotate labels identically
            m = np.rot90(m, k).copy()                          # rotate mask identically
            if np.random.rand() < 0.5:                         # 50% horizontal flip
                x, y, m = x[:, ::-1].copy(), y[:, ::-1].copy(), m[:, ::-1].copy()
            if np.random.rand() < 0.5:                         # 50% vertical flip
                x, y, m = x[::-1].copy(), y[::-1].copy(), m[::-1].copy()
            if self.noise_std > 0:                             # additive feature noise
                x += np.random.normal(0, self.noise_std,       # tiny Gaussian jitter
                                      x.shape).astype("float32")
        x = torch.from_numpy(np.ascontiguousarray(             # HWC -> CHW tensor
            x.transpose(2, 0, 1)))
        y = torch.from_numpy(y).float()                        # label tensor (t, t)
        m = torch.from_numpy(m.astype("float32"))              # mask tensor (t, t)
        return x, y, m                                         # one training sample

# ---------------------------------------------------------------------------#
# LOSS & METRICS — everything is computed ONLY on supervised pixels           #
# ---------------------------------------------------------------------------#

def masked_bce_dice(logits, target, mask, dice_w=0.5):
    """BCE + Dice, both restricted to mask==1 pixels."""
    logits = logits.squeeze(1)                                 # (B, H, W)
    bce = F.binary_cross_entropy_with_logits(                  # per-pixel BCE
        logits, target, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1)         # mean over supervised px
    prob = torch.sigmoid(logits)                               # probabilities for Dice
    inter = (prob * target * mask).sum()                       # masked intersection
    denom = ((prob + target) * mask).sum().clamp(min=1e-6)     # masked union proxy
    dice = 1 - (2 * inter + 1) / (denom + 1)                   # soft Dice loss
    return (1 - dice_w) * bce + dice_w * dice                  # weighted combination


def rank_auc(scores, labels):
    """Dependency-free ROC-AUC via the rank statistic (Mann-Whitney U)."""
    order = np.argsort(scores)                                 # ascending score order
    ranks = np.empty_like(order, dtype=np.float64)             # allocate rank array
    ranks[order] = np.arange(1, len(scores) + 1)               # 1-based ranks
    n_pos = labels.sum()                                       # number of positives
    n_neg = len(labels) - n_pos                                # number of negatives
    if n_pos == 0 or n_neg == 0:                               # degenerate case guard
        return float("nan")
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2     # U statistic
    return u / (n_pos * n_neg)                                 # AUC in [0, 1]


@torch.no_grad()
def evaluate(model, loader):
    """Run the model on a loader; return masked metrics dict."""
    model.eval()                                               # inference mode
    all_p, all_y = [], []                                      # collected probs/labels
    for x, y, m in loader:                                     # iterate batches
        x = x.to(DEVICE, non_blocking=True)                    # features to GPU
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            prob = torch.sigmoid(model(x)).squeeze(1).cpu()    # predicted probabilities
        sel = m.bool()                                         # supervised pixels only
        all_p.append(prob[sel].numpy())                        # keep their probs
        all_y.append(y[sel].numpy())                           # and their labels
    p = np.concatenate(all_p)                                  # flatten over all tiles
    y = np.concatenate(all_y).astype(np.int8)                  # labels as ints
    pred = (p >= 0.5).astype(np.int8)                          # hard predictions @0.5
    tp = int(((pred == 1) & (y == 1)).sum())                   # true positives
    fp = int(((pred == 1) & (y == 0)).sum())                   # false positives
    fn = int(((pred == 0) & (y == 1)).sum())                   # false negatives
    tn = int(((pred == 0) & (y == 0)).sum())                   # true negatives
    precision = tp / max(tp + fp, 1)                           # positive predictive value
    recall = tp / max(tp + fn, 1)                              # sensitivity
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)  # harmonic mean
    iou = tp / max(tp + fp + fn, 1)                            # Jaccard index
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)                # overall accuracy
    return {"auc": rank_auc(p, y), "f1": f1, "iou": iou,       # metric dictionary
            "precision": precision, "recall": recall, "accuracy": acc}

# ---------------------------------------------------------------------------#
# TRAINING — AMP, warm-up + cosine LR, early stopping, checkpointing          #
# ---------------------------------------------------------------------------#

def lr_lambda_factory(cfg):
    """Return a function mapping epoch -> LR multiplier:
    linear warm-up for `warmup_epochs`, then cosine decay to ~0."""
    def fn(epoch):
        if epoch < cfg["warmup_epochs"]:                       # warm-up phase
            return (epoch + 1) / cfg["warmup_epochs"]          # 1/W, 2/W, ..., 1
        t = (epoch - cfg["warmup_epochs"]) / max(              # progress in decay phase
            cfg["epochs"] - cfg["warmup_epochs"], 1)
        return 0.5 * (1 + math.cos(math.pi * t))               # cosine from 1 -> 0
    return fn


def train(cfg=CFG):
    os.makedirs(cfg["out_dir"], exist_ok=True)                 # ensure output folder
    rng = np.random.default_rng(cfg["seed"])                   # seeded NumPy generator

    # ---- load data ---------------------------------------------------------
    cube = np.load(cfg["cube_path"])                           # (H, W, C) features
    valid = np.load(cfg["valid_path"])                         # (H, W) validity
    labels = load_labels(cfg["label_path"], valid.shape)       # (H, W) labels/NaN
    in_ch = cube.shape[-1]                                     # number of feature bands
    print(f"Cube {cube.shape}, labelled px: {np.isfinite(labels).sum():,}")

    # ---- tiles + leakage-safe split ---------------------------------------
    splits = build_tiles(valid, labels, cfg, rng)              # spatial block split
    print({k: len(v) for k, v in splits.items()})              # tiles per split

    def make_loader(split, augment):                           # loader factory
        ds = FloodTileDataset(cube, labels, valid, splits[split],
                              cfg["tile"], augment, cfg["aug_noise"])
        return DataLoader(ds, batch_size=cfg["batch_size"],    # batched tiles
                          shuffle=augment,                     # shuffle only training
                          num_workers=cfg["num_workers"],      # parallel loading
                          pin_memory=DEVICE.type == "cuda",    # faster host->GPU copy
                          drop_last=augment)                   # stable batch stats

    train_loader = make_loader("train", augment=True)          # augmented training data
    val_loader = make_loader("val", augment=False)             # clean validation data
    test_loader = make_loader("test", augment=False)           # clean held-out test data

    # ---- model / optimizer / schedule / AMP --------------------------------
    model = SegFormerFlood(in_ch, cfg).to(DEVICE)              # build & move model
    n_par = sum(p.numel() for p in model.parameters())         # parameter count
    print(f"SegFormer-B0 params: {n_par/1e6:.2f} M on {DEVICE}")
    opt = torch.optim.AdamW(model.parameters(),                # AdamW optimizer
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(                 # warm-up + cosine LR
        opt, lr_lambda_factory(cfg))
    scaler = torch.amp.GradScaler(                             # AMP loss scaler
        enabled=DEVICE.type == "cuda")

    # ---- training loop with early stopping ---------------------------------
    best_auc, best_epoch, history = -1.0, -1, []               # tracking state
    ckpt_path = os.path.join(cfg["out_dir"], "best_model.pt")  # checkpoint file
    for epoch in range(cfg["epochs"]):                         # epoch loop
        model.train()                                          # training mode
        running = 0.0                                          # loss accumulator
        for x, y, m in train_loader:                           # batch loop
            x = x.to(DEVICE, non_blocking=True)                # features -> device
            y = y.to(DEVICE, non_blocking=True)                # labels -> device
            m = m.to(DEVICE, non_blocking=True)                # mask -> device
            opt.zero_grad(set_to_none=True)                    # clear old gradients
            with torch.autocast(DEVICE.type,                   # mixed-precision region
                                enabled=DEVICE.type == "cuda"):
                loss = masked_bce_dice(model(x), y, m)         # masked BCE+Dice loss
            scaler.scale(loss).backward()                      # scaled backprop (AMP)
            scaler.unscale_(opt)                               # unscale before clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            scaler.step(opt)                                   # optimizer step (AMP-safe)
            scaler.update()                                    # adjust the loss scale
            running += loss.item() * x.size(0)                 # accumulate batch loss
        sched.step()                                           # advance the LR schedule
        train_loss = running / len(train_loader.dataset)       # mean epoch loss

        val = evaluate(model, val_loader)                      # validation metrics
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "lr": sched.get_last_lr()[0], **{f"val_{k}": v
                        for k, v in val.items()}})             # log the epoch
        print(f"E{epoch:03d} loss {train_loss:.4f} "
              f"val AUC {val['auc']:.4f} F1 {val['f1']:.4f} "
              f"IoU {val['iou']:.4f} lr {sched.get_last_lr()[0]:.2e}")

        if val["auc"] > best_auc:                              # improvement?
            best_auc, best_epoch = val["auc"], epoch           # update the record
            torch.save({"model": model.state_dict(),           # save weights
                        "cfg": cfg, "epoch": epoch,            # save config + epoch
                        "val_metrics": val,                    # save val metrics
                        "in_ch": in_ch}, ckpt_path)            # save channel count
            print(f"  -> new best (AUC {best_auc:.4f}), checkpoint saved")
        elif epoch - best_epoch >= cfg["patience"]:            # patience exhausted?
            print(f"Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, AUC {best_auc:.4f})")
            break                                              # stop training

    # ---- final test with the BEST checkpoint (never the last epoch) --------
    ckpt = torch.load(ckpt_path, map_location=DEVICE,          # reload best weights
                      weights_only=False)
    model.load_state_dict(ckpt["model"])                       # restore parameters
    test = evaluate(model, test_loader)                        # held-out evaluation
    print("TEST:", {k: round(v, 4) for k, v in test.items()})  # report test metrics

    # ---- persist the run log ------------------------------------------------
    with open(os.path.join(cfg["out_dir"], "history.json"), "w") as f:
        json.dump({"history": history, "test": test,           # full training record
                   "best_epoch": best_epoch}, f, indent=2)

    return model, cube, valid                                   # for full-map inference

# ---------------------------------------------------------------------------#
# FULL-BASIN INFERENCE — sliding window with overlap averaging                #
# ---------------------------------------------------------------------------#

@torch.no_grad()
def predict_map(model, cube, valid, cfg=CFG):
    model.eval()                                               # inference mode
    H, W, C = cube.shape                                       # raster dimensions
    t, s = cfg["tile"], cfg["tile"] // 2                       # window + 50% overlap
    prob = np.zeros((H, W), "float32")                         # probability accumulator
    count = np.zeros((H, W), "float32")                        # overlap counter
    rows = list(range(0, max(H - t, 0) + 1, s)) or [0]         # window row starts
    cols = list(range(0, max(W - t, 0) + 1, s)) or [0]         # window col starts
    for r in rows:                                             # slide over rows
        for c in cols:                                         # slide over cols
            x = cube[r:r + t, c:c + t, :].transpose(2, 0, 1)   # (C, t, t) window
            x = torch.from_numpy(np.ascontiguousarray(x))      # to tensor
            x = x.unsqueeze(0).to(DEVICE)                      # add batch dim, to GPU
            with torch.autocast(DEVICE.type,                   # AMP inference
                                enabled=DEVICE.type == "cuda"):
                p = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
            prob[r:r + t, c:c + t] += p                        # accumulate probabilities
            count[r:r + t, c:c + t] += 1                       # count contributions
    prob /= np.maximum(count, 1)                               # average the overlaps
    prob[~valid] = np.nan                                      # blank invalid pixels
    out = os.path.join(cfg["out_dir"], "susceptibility.npy")   # output path
    np.save(out, prob)                                         # save the final map
    print(f"Susceptibility map saved: {out}")
    return prob                                                # (H, W) probabilities

# ---------------------------------------------------------------------------#
# ENTRY POINT                                                                 #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    model, cube, valid = train()                               # run the full pipeline
    predict_map(model, cube, valid)                            # export the basin map

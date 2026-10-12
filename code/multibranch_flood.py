"""
Multi-branch SegFormer for sensor-level explainable fusion (novelty N1).

Three separate encoders — SAR, optical, terrain/static — each see only their
own data source; their multi-scale features are concatenated per scale and
decoded jointly. Because the sources never mix before fusion, the model
supports true SENSOR-LEVEL attribution:

  - branch ablation      : zero one branch's input -> AUC drop per source
  - branch Grad-CAM      : decoder gradients w.r.t. each branch's features
  - learned fusion share : per-scale channel-norm share of each branch

Band grouping is derived from metadata.json band names:
  SAR      : *s1_*                                   (2 bands)
  OPTICAL  : *s2_composite*, *ndvi, *ndwi, *mndwi    (9 bands)
  TERRAIN  : everything else (DEM & derivatives, HAND/TWI/SPI/TPI/curvature,
             distance/density, rainfall, aspect, LULC one-hots)  (25 bands)

Trains on the 36-band cube; reuses the dataset/loss/eval machinery from
segformer_flood.py. Designed for one Colab T4 session (~1-2 h).

Usage:  python multibranch_flood.py            (train + evaluate + ablation)
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segformer_flood import (CFG, DEVICE, MiTEncoder, SegFormerHead,
                             FloodTileDataset, build_tiles, load_labels,
                             masked_bce_dice, rank_auc, evaluate,
                             lr_lambda_factory)
from torch.utils.data import DataLoader

OUT = "runs/multibranch"

# thin per-branch encoder (three of these ≈ one MiT-B0 in cost)
BRANCH_CFG = {
    "embed_dims": [16, 32, 80, 128],
    "depths":     [2, 2, 2, 2],
    "num_heads":  [1, 2, 5, 8],
    "sr_ratios":  [8, 4, 2, 1],
    "decoder_dim": 256,
}


def band_groups(meta_path):
    """Split band indices into (sar, optical, terrain) by name."""
    names = json.load(open(meta_path))["band_names"]
    sar = [i for i, n in enumerate(names) if "s1_" in n]
    opt = [i for i, n in enumerate(names)
           if "s2_composite" in n or n.endswith(("ndvi", "ndwi", "mndwi"))]
    ter = [i for i in range(len(names)) if i not in sar + opt]
    return {"SAR": sar, "OPTICAL": opt, "TERRAIN": ter}, names


class MultiBranchSegFormer(nn.Module):
    """Three MiT encoders (one per source) + concat fusion + MLP decoder."""

    def __init__(self, groups, bcfg=BRANCH_CFG):
        super().__init__()
        self.groups = groups                                   # dict name -> band idx
        self.order = list(groups)                              # fixed branch order
        self.encoders = nn.ModuleDict({
            g: MiTEncoder(len(idx), bcfg) for g, idx in groups.items()})
        fused = [d * len(groups) for d in bcfg["embed_dims"]]  # concat widths
        head_cfg = dict(bcfg); head_cfg["embed_dims"] = fused
        self.head = SegFormerHead(head_cfg)

    def forward(self, x, drop_branch=None):
        """drop_branch: optionally zero one source's INPUT (ablation)."""
        feats_per_branch = []
        for g in self.order:
            xg = x[:, self.groups[g], :, :]
            if drop_branch == g:
                xg = torch.zeros_like(xg)                      # 0 = feature mean
            feats_per_branch.append(self.encoders[g](xg))
        fused = [torch.cat([fb[s] for fb in feats_per_branch], dim=1)
                 for s in range(4)]                            # concat per scale
        logits = self.head(fused)
        return F.interpolate(logits, size=x.shape[2:],
                             mode="bilinear", align_corners=False)


@torch.no_grad()
def branch_ablation(model, loader, groups):
    """AUC with each branch's input zeroed -> per-source importance."""
    out = {}
    for mode in [None] + list(groups):
        ps, ys = [], []
        for x, y, m in loader:
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                p = torch.sigmoid(model(x.to(DEVICE), drop_branch=mode))
            sel = m.bool()
            ps.append(p.squeeze(1).cpu()[sel].numpy())
            ys.append(y[sel].numpy())
        auc = rank_auc(np.concatenate(ps), np.concatenate(ys).astype(np.int8))
        out["full" if mode is None else mode] = float(auc)
        print(f"  ablation {'full' if mode is None else mode}: AUC {auc:.4f}",
              flush=True)
    return out


def train():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])

    groups, names = band_groups("data/processed/metadata.json")
    print({g: len(i) for g, i in groups.items()})

    cube = np.load(CFG["cube_path"], mmap_mode="r")
    valid = np.load(CFG["valid_path"])
    labels = load_labels(CFG["label_path"], valid.shape)
    rng = np.random.default_rng(CFG["seed"])
    splits = build_tiles(valid, labels, CFG, rng)
    print({k: len(v) for k, v in splits.items()})

    def loader(split, augment):
        ds = FloodTileDataset(np.asarray(cube) if DEVICE.type == "cuda" else cube,
                              labels, valid, splits[split], CFG["tile"],
                              augment, CFG["aug_noise"])
        return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=augment,
                          num_workers=CFG["num_workers"],
                          pin_memory=DEVICE.type == "cuda", drop_last=augment)

    train_dl, val_dl, test_dl = (loader("train", True), loader("val", False),
                                 loader("test", False))

    model = MultiBranchSegFormer(groups).to(DEVICE)
    print(f"Multi-branch params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(CFG))
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_auc, best_ep, hist = -1, -1, []
    ckpt = f"{OUT}/best_model.pt"
    for ep in range(CFG["epochs"]):
        model.train(); running = 0.0
        for x, y, m in train_dl:
            x, y, m = (t.to(DEVICE, non_blocking=True) for t in (x, y, m))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = masked_bce_dice(model(x), y, m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0)
        sched.step()
        val = evaluate(model, val_dl)
        hist.append({"epoch": ep, "train_loss": running / len(train_dl.dataset),
                     **{f"val_{k}": v for k, v in val.items()}})
        print(f"E{ep:03d} loss {hist[-1]['train_loss']:.4f} "
              f"val AUC {val['auc']:.4f} F1 {val['f1']:.4f}", flush=True)
        if val["auc"] > best_auc:
            best_auc, best_ep = val["auc"], ep
            torch.save({"model": model.state_dict(), "groups": groups,
                        "branch_cfg": BRANCH_CFG, "epoch": ep,
                        "val_metrics": val}, ckpt)
            print(f"  -> new best (AUC {best_auc:.4f})", flush=True)
        elif ep - best_ep >= CFG["patience"]:
            print(f"Early stopping at {ep} (best {best_ep})")
            break

    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    test = evaluate(model, test_dl)
    print("TEST:", {k: round(v, 4) for k, v in test.items()})

    # ---- sensor-level attribution: branch ablation on the test set ---------
    print("Branch ablation (sensor-level attribution):")
    ablation = branch_ablation(model, test_dl, groups)

    with open(f"{OUT}/history.json", "w") as f:
        json.dump({"history": hist, "test": test, "ablation": ablation,
                   "groups": {g: len(i) for g, i in groups.items()},
                   "best_epoch": best_ep}, f, indent=2)
    print("DONE. runs/multibranch/: best_model.pt, history.json")


if __name__ == "__main__":
    train()

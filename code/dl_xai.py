"""
DL-side explainability for the trained SegFormer (CPU-friendly).

  Part 1 (mode "igcam"):
    - Integrated Gradients per band (batched; baseline = 0 = feature mean)
    - Grad-CAM example rows
    - Training-curve figure from history.json
    - SegFormer vs XGBoost susceptibility map comparison (masked to the
      floodable domain -> fixes the extrapolation-artifact display)
  Part 2 (mode "perm"):
    - Permutation importance on the SegFormer (3 repeats)
    - Three-way XAI consistency figure (TreeSHAP vs IG vs permutation)

Usage: python dl_xai.py igcam   |   python dl_xai.py perm
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

from segformer_flood import (CFG, DEVICE, SegFormerFlood, FloodTileDataset,
                             build_tiles, load_labels, rank_auc)
from xai_analysis import (save_fig, band_names, gradcam, XCFG,
                          INK, MUTED, BAR_HUE, SEQ_CMAP, DIV_CMAP)

OUT = "../runs/segformer"
N_PROBE = 32          # probe tiles for IG + permutation (CPU budget)
IG_STEPS = 16         # Riemann steps for Integrated Gradients
N_REPEATS = 3         # permutation repeats


# --------------------------------------------------------------------------- #
# Shared setup                                                                 #
# --------------------------------------------------------------------------- #

def load_all():
    ckpt = torch.load(f"{OUT}/best_model.pt", map_location=DEVICE,
                      weights_only=False)
    model = SegFormerFlood(ckpt["in_ch"], ckpt["cfg"]).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    cube = np.load(CFG["cube_path"], mmap_mode="r")
    valid = np.load(CFG["valid_path"])
    labels = load_labels(CFG["label_path"], valid.shape)
    rng = np.random.default_rng(CFG["seed"])
    splits = build_tiles(valid, labels, CFG, rng)
    ds = FloodTileDataset(np.asarray(cube), labels, valid, splits["test"],
                          CFG["tile"], augment=False, noise_std=0)
    probe = [ds[i * max(len(ds) // N_PROBE, 1)] for i in range(min(N_PROBE, len(ds)))]
    tiles = torch.stack([p[0] for p in probe])
    ys = torch.stack([p[1] for p in probe])
    ms = torch.stack([p[2] for p in probe])
    return model, ds, tiles, ys, ms, labels


# --------------------------------------------------------------------------- #
# Part 1a — Integrated Gradients (batched, per band)                           #
# --------------------------------------------------------------------------- #

def integrated_gradients(model, tiles, names):
    B = 8                                                # tiles per batch
    n, C = tiles.shape[0], tiles.shape[1]
    ig = torch.zeros(n, C)                               # per-tile, per-band net IG
    for s in range(0, n, B):
        x = tiles[s:s + B].to(DEVICE)                    # (b, C, H, W)
        grads_sum = torch.zeros_like(x)
        for k in range(1, IG_STEPS + 1):                 # Riemann midpoint path
            alpha = (k - 0.5) / IG_STEPS
            xa = (x * alpha).requires_grad_(True)
            out = model(xa).mean(dim=(1, 2, 3)).sum()    # scalar over batch
            out.backward()
            grads_sum += xa.grad
        attr = x.cpu() * grads_sum.cpu() / IG_STEPS      # IG = x * mean grad (baseline 0)
        ig[s:s + B] = attr.sum(dim=(2, 3))               # aggregate over space
        print(f"  IG: {min(s + B, n)}/{n} tiles", flush=True)
    ig = ig.numpy()
    xb = tiles.numpy().mean(axis=(2, 3))                 # mean band value per tile

    imp = np.abs(ig).mean(0)
    order = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    vmax = np.percentile(np.abs(xb), 95)
    for row, i in enumerate(order[::-1]):
        yjit = row + (np.random.rand(len(ig)) - 0.5) * 0.55
        sc = ax.scatter(ig[:, i], yjit, c=xb[:, i], cmap=DIV_CMAP,
                        vmin=-vmax, vmax=vmax, s=14, alpha=0.8, edgecolors="none")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order[::-1]])
    ax.set_xlabel("Integrated Gradients attribution (net effect on flood logit)")
    ax.set_title("SegFormer — Integrated Gradients per band", loc="left")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("band value (z-score)", color=MUTED)
    save_fig(fig, "figB1_ig_segformer")

    np.savetxt("../figures/importance_ig_dl.csv",
               np.column_stack([np.array(names)[order], imp[order]]),
               fmt="%s", delimiter=",", header="feature,mean_abs_ig")
    return dict(zip(names, imp))


# --------------------------------------------------------------------------- #
# Part 1b — Grad-CAM rows, training curves, map comparison                     #
# --------------------------------------------------------------------------- #

def gradcam_figure(model, ds, names):
    n = 4
    elev_idx = names.index("swat_dem")
    # pick the test tiles that actually CONTAIN flood labels — evenly spaced
    # sampling lands on barren mountain tiles where prediction ≈ 0 everywhere.
    t = ds.tile
    flood_count = [np.nansum(ds.labels[r:r + t, c:c + t] == 1)
                   for r, c in ds.corners]
    top = np.argsort(flood_count)[::-1][:n]              # most flooded tiles
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.4 * n))
    for row in range(n):
        x, y, m = ds[int(top[row])]
        xb = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(xb))[0, 0].cpu().numpy()
        cam = gradcam(model, xb)
        bg = x[elev_idx].numpy()
        axes[row, 0].imshow(bg, cmap="Greys_r")
        axes[row, 1].imshow(prob, cmap=SEQ_CMAP, vmin=0, vmax=1)
        axes[row, 2].imshow(bg, cmap="Greys_r")
        im = axes[row, 2].imshow(cam, cmap=SEQ_CMAP, alpha=0.55, vmin=0, vmax=1)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_title("Elevation (context)", loc="left")
    axes[0, 1].set_title("Predicted susceptibility", loc="left")
    axes[0, 2].set_title("Grad-CAM attention", loc="left")
    cb = fig.colorbar(im, ax=axes[:, 2], shrink=0.85, pad=0.02)
    cb.set_label("normalized attention / probability", color=MUTED)
    save_fig(fig, "figC1_gradcam_segformer")


def training_curves():
    h = json.load(open(f"{OUT}/history.json"))["history"]
    ep = [r["epoch"] for r in h]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3))
    ax1.plot(ep, [r["train_loss"] for r in h], color=BAR_HUE, lw=1.6)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
    ax1.set_title("SegFormer training loss", loc="left")
    ax2.plot(ep, [r["val_auc"] for r in h], color=BAR_HUE, lw=1.6)
    best = max(range(len(h)), key=lambda i: h[i]["val_auc"])
    ax2.scatter([ep[best]], [h[best]["val_auc"]], color="#D55E00", zorder=5,
                label=f"best (E{ep[best]}, AUC {h[best]['val_auc']:.3f})")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("validation AUC")
    ax2.set_title("Validation AUC", loc="left"); ax2.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig_training_curves_segformer")


def map_comparison(labels):
    import rasterio
    seg = np.load(f"{OUT}/susceptibility.npy")
    with rasterio.open("../data/processed/susceptibility_xgb.tif") as src:
        xgb = src.read(1)
    floodable = np.isfinite(labels)                     # labelled 0/1 = floodable domain
    for m in (seg, xgb):
        m[~floodable] = np.nan                          # mask extrapolation zone
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.6))
    for ax, m, title in [(axes[0], xgb, "XGBoost"), (axes[1], seg, "SegFormer")]:
        im = ax.imshow(m[::4, ::4], cmap=SEQ_CMAP, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{title} — floodable domain", loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("P(flood)", color=MUTED)
    save_fig(fig, "fig_map_comparison_masked")


# --------------------------------------------------------------------------- #
# Part 2 — permutation importance + consistency                                #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def batch_auc(model, tiles, ys, ms):
    probs = []
    for i in range(0, len(tiles), 16):
        probs.append(torch.sigmoid(model(tiles[i:i + 16].to(DEVICE))).squeeze(1).cpu())
    p = torch.cat(probs)
    sel = ms.bool()
    return rank_auc(p[sel].numpy(), ys[sel].numpy().astype(np.int8))


def permutation(model, tiles, ys, ms, names):
    rng = np.random.default_rng(CFG["seed"])
    base = batch_auc(model, tiles, ys, ms)
    print(f"baseline probe AUC: {base:.4f}", flush=True)
    C = tiles.shape[1]
    drops = np.zeros((C, N_REPEATS))
    for c in range(C):
        for r in range(N_REPEATS):
            perm = tiles.clone()
            perm[:, c] = tiles[rng.permutation(len(tiles))][:, c]
            drops[c, r] = base - batch_auc(model, perm, ys, ms)
        print(f"  perm band {c + 1}/{C} ({names[c]}): "
              f"drop {drops[c].mean():.4f}", flush=True)
    mean_d, sd = drops.mean(1), drops.std(1)
    order = np.argsort(mean_d)[::-1][:15]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh([names[i] for i in order][::-1], mean_d[order][::-1],
            xerr=sd[order][::-1], color=BAR_HUE, height=0.62,
            error_kw={"ecolor": MUTED, "elinewidth": 0.9})
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("AUC drop when band is permuted")
    ax.set_title("SegFormer — permutation importance", loc="left")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "figD1_permutation_segformer")
    np.savetxt("../figures/importance_perm_dl.csv",
               np.column_stack([np.array(names)[order], mean_d[order], sd[order]]),
               fmt="%s", delimiter=",", header="feature,mean_auc_drop,std")
    return dict(zip(names, mean_d))


def consistency(names):
    def read_csv(path, col=1):
        rows = [l.strip().split(",") for l in open(path) if not l.startswith("#")]
        return {r[0]: float(r[col]) for r in rows if len(r) > col}
    methods = {}
    for label, path in [("TreeSHAP (XGB)", "../figures/importance_shap_xgb.csv"),
                        ("IntGrad (SegFormer)", "../figures/importance_ig_dl.csv"),
                        ("Permutation (SegFormer)", "../figures/importance_perm_dl.csv")]:
        if os.path.exists(path):
            methods[label] = read_csv(path)
    keys = list(methods)
    def ranks(d):
        vals = np.array([d.get(n, 0.0) for n in names])
        return np.argsort(np.argsort(-vals))
    fig, axes = plt.subplots(1, len(keys) - 1,
                             figsize=(3.6 * (len(keys) - 1), 3.5), squeeze=False)
    r0, rhos = ranks(methods[keys[0]]), []
    for j, k in enumerate(keys[1:]):
        rj, ax = ranks(methods[k]), axes[0, j]
        rho = 1 - 6 * ((r0 - rj) ** 2).sum() / (len(r0) * (len(r0) ** 2 - 1))
        rhos.append(rho)
        ax.scatter(r0, rj, s=18, color=BAR_HUE, alpha=0.85, edgecolors="none")
        lim = len(names) - 1
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel(f"rank — {keys[0]}")
        ax.set_ylabel(f"rank — {k}")
        ax.set_title(f"Spearman ρ = {rho:.2f}", loc="left")
    fig.suptitle("Explanation consistency across XAI methods", x=0.02, ha="left")
    fig.tight_layout()
    save_fig(fig, "figE1_xai_consistency")
    print("Spearman correlations vs TreeSHAP:", [f"{r:.3f}" for r in rhos])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "igcam"
    names = band_names()
    model, ds, tiles, ys, ms, labels = load_all()
    if mode == "igcam":
        training_curves()
        map_comparison(labels)
        gradcam_figure(model, ds, names)
        integrated_gradients(model, tiles, names)
    elif mode == "cam":
        gradcam_figure(model, ds, names)
    else:
        permutation(model, tiles, ys, ms, names)
        consistency(names)
    print("done:", mode)

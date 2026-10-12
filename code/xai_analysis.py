"""
Explainable AI for the flood susceptibility models — SHAP, Grad-CAM,
and permutation feature importance, with publication-quality figures.

Three complementary explanation levels (matches the paper's XAI framework):

  A. TreeSHAP on the XGBoost tabular baseline   -> WHICH factors drive
     susceptibility globally, and in which direction (beeswarm, bar,
     dependence plots). Exact Shapley values — the reference explanation.

  B. GradientSHAP on the SegFormer/DL model     -> does the deep model use
     the same factors? Per-band attributions aggregated from patches;
     enables the cross-model "explanation consistency" analysis (novelty N2).

  C. Grad-CAM on the SegFormer decoder          -> WHERE in the landscape the
     model looks when predicting flood (spatial saliency overlays).

  D. Permutation importance on the DL model     -> model-agnostic sanity
     check: how much does test AUC drop when one band is scrambled?

Inputs  : outputs of preprocess_pipeline.py + segformer_flood.py checkpoint
Outputs : ../figures/*.png (300 dpi) and *.pdf (vector) + importance CSVs

Usage:
    pip install torch numpy matplotlib shap xgboost rasterio
    python xai_analysis.py          # parts run independently; missing
                                    # optional deps skip their part gracefully
"""

import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# import the model + data utilities from the training script (same folder)
from segformer_flood import (CFG, DEVICE, SegFormerFlood, FloodTileDataset,
                             build_tiles, load_labels, rank_auc)

# ---------------------------------------------------------------------------#
# CONFIG                                                                      #
# ---------------------------------------------------------------------------#
XCFG = {
    "ckpt_path":   "../runs/segformer/best_model.pt",
    "meta_path":   "../data/processed/metadata.json",     # band names live here
    "pixels_dir":  "../data/processed",                    # X_*_pixels.npy for XGBoost
    "fig_dir":     "../figures",
    "n_background": 64,      # background tiles for GradientSHAP baselines
    "n_explain":    128,     # tiles to explain with GradientSHAP
    "n_perm_repeats": 5,     # permutation-importance repeats (error bars)
    "n_cam_examples": 4,     # Grad-CAM example tiles in the overlay figure
    "top_k": 15,             # features shown in ranked charts
}

# ---------------------------------------------------------------------------#
# PUBLICATION FIGURE STYLE                                                    #
# One style block so every figure in the paper looks like one system.        #
# Color rules: sequential magnitude -> viridis (perceptually uniform, CVD-   #
# safe); signed attributions -> blue/white/red diverging with a NEUTRAL      #
# midpoint; single-series bars -> ONE muted hue (no legend needed); text     #
# and axes in neutral ink, recessive grid.                                   #
# ---------------------------------------------------------------------------#
INK      = "#111111"      # primary text/ink (near-black, print-legible)
MUTED    = "#333333"      # secondary ink (ticks, captions) — darkened for print
GRID     = "#dcdcdc"      # recessive gridlines
BAR_HUE  = "#0072B2"      # Okabe-Ito blue — single-series bars (CVD-safe)
ACCENT   = "#D55E00"      # Okabe-Ito vermillion — one highlight, used sparingly
SEQ_CMAP = "viridis"      # sequential: magnitude maps (Grad-CAM, susceptibility)
DIV_CMAP = "RdBu_r"       # diverging: signed SHAP (blue=lowers, red=raises risk)

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 400,               # higher export resolution
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,                               # grid behind the data
    "xtick.color": "#111111", "ytick.color": "#111111",
    "text.color": INK, "axes.labelcolor": INK,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save_fig(fig, name):
    """Save every figure as 300-dpi PNG (review) + vector PDF (submission)."""
    os.makedirs(XCFG["fig_dir"], exist_ok=True)
    fig.savefig(os.path.join(XCFG["fig_dir"], f"{name}.png"))
    fig.savefig(os.path.join(XCFG["fig_dir"], f"{name}.pdf"))
    plt.close(fig)
    print(f"  figure saved: {name}.png / .pdf")


def band_names():
    """Band names in cube order, from the preprocessing metadata."""
    with open(XCFG["meta_path"]) as f:
        return json.load(f)["band_names"]


# ===========================================================================#
# PART A — TreeSHAP on the XGBoost tabular baseline                          #
# ===========================================================================#

def part_a_treeshap(names):
    """Train (or reuse) an XGBoost baseline on the pixel vectors and produce
    the three canonical SHAP figures: beeswarm, ranked bar, dependence."""
    try:
        import shap, xgboost as xgb
    except ImportError:
        print("[A] shap/xgboost not installed -> skipping TreeSHAP part")
        return None

    # ---- load the tabular training data produced by the pipeline ----------
    p = XCFG["pixels_dir"]
    Xtr = np.load(f"{p}/X_train_pixels.npy"); ytr = np.load(f"{p}/y_train_pixels.npy")
    Xte = np.load(f"{p}/X_test_pixels.npy");  yte = np.load(f"{p}/y_test_pixels.npy")

    # ---- fit the baseline ---------------------------------------------------
    model = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="auc", random_state=CFG["seed"])
    model.fit(Xtr, ytr)
    auc = rank_auc(model.predict_proba(Xte)[:, 1], yte)
    print(f"[A] XGBoost test AUC = {auc:.4f}")

    # ---- exact Shapley values on a test subsample ---------------------------
    sub = np.random.default_rng(0).choice(len(Xte), min(3000, len(Xte)), False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte[sub])                   # (n, C) signed values

    # ---- Figure A1: beeswarm (global importance + direction) ----------------
    fig = plt.figure(figsize=(6.5, 5))
    shap.summary_plot(sv, Xte[sub], feature_names=names, show=False,
                      max_display=XCFG["top_k"], cmap=plt.get_cmap(DIV_CMAP),
                      plot_size=None)
    plt.gca().set_xlabel("SHAP value (impact on flood probability)")
    plt.title("XGBoost — SHAP beeswarm (test samples)", loc="left")
    save_fig(fig, "figA1_shap_beeswarm_xgb")

    # ---- Figure A2: ranked mean |SHAP| bar chart -----------------------------
    imp = np.abs(sv).mean(0)                               # global importance
    order = np.argsort(imp)[::-1][:XCFG["top_k"]]          # top-k features
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1],
            color=BAR_HUE, height=0.62)                    # one hue, thin bars
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("XGBoost — global feature importance (SHAP)", loc="left")
    ax.grid(axis="y", visible=False)                       # grid only on value axis
    save_fig(fig, "figA2_shap_importance_bar_xgb")

    # ---- Figure A3: dependence plots for the top-4 factors -------------------
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.6))
    for ax, i in zip(axes.ravel(), order[:4]):
        ax.scatter(Xte[sub][:, i], sv[:, i], s=4, alpha=0.35,
                   c=BAR_HUE, edgecolors="none")           # one hue; density via alpha
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")        # neutral zero reference
        ax.set_xlabel(f"{names[i]} (z-score)")
        ax.set_ylabel("SHAP value")
    fig.suptitle("SHAP dependence — top four factors", x=0.02, ha="left")
    fig.tight_layout()
    save_fig(fig, "figA3_shap_dependence_xgb")

    np.savetxt(os.path.join(XCFG["fig_dir"], "importance_shap_xgb.csv"),
               np.column_stack([np.array(names)[order], imp[order]]),
               fmt="%s", delimiter=",", header="feature,mean_abs_shap")
    return dict(zip(names, imp))                           # for consistency analysis


# ===========================================================================#
# PART B — GradientSHAP on the deep model (per-band attributions)            #
# ===========================================================================#

class ScalarWrapper(torch.nn.Module):
    """Wrap the segmentation model so it outputs ONE scalar per tile
    (the mean flood logit) — what gradient-based SHAP needs to explain."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)


def part_b_gradientshap(model, tiles, names):
    """Per-band SHAP for the deep model, aggregated over space, as a beeswarm-
    style strip plot comparable with Part A's ranking."""
    try:
        import shap
    except ImportError:
        print("[B] shap not installed -> skipping GradientSHAP part")
        return None

    wrapper = ScalarWrapper(model).to(DEVICE).eval()
    bg = tiles[:XCFG["n_background"]].to(DEVICE)           # baseline distribution
    ex = tiles[XCFG["n_background"]:                       # tiles to explain
               XCFG["n_background"] + XCFG["n_explain"]].to(DEVICE)
    explainer = shap.GradientExplainer(wrapper, bg)
    sv = explainer.shap_values(ex)                         # (n, C, H, W) attributions
    sv = sv[0] if isinstance(sv, list) else sv
    sv_band = sv.reshape(sv.shape[0], sv.shape[1], -1).sum(-1)   # (n, C): per-band net
    x_band = ex.cpu().numpy().mean(axis=(2, 3))            # (n, C): mean band value

    imp = np.abs(sv_band).mean(0)                          # global per-band importance
    order = np.argsort(imp)[::-1][:XCFG["top_k"]]

    # ---- Figure B1: strip plot (per-tile signed attribution per band) -------
    fig, ax = plt.subplots(figsize=(6.5, 5))
    vmax = np.percentile(np.abs(x_band), 95)               # robust color scaling
    for row, i in enumerate(order[::-1]):                  # weakest at bottom
        yjit = row + (np.random.rand(len(sv_band)) - 0.5) * 0.55
        sc = ax.scatter(sv_band[:, i], yjit, c=x_band[:, i], cmap=DIV_CMAP,
                        vmin=-vmax, vmax=vmax, s=8, alpha=0.75, edgecolors="none")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order[::-1]])
    ax.set_xlabel("SHAP value (net effect on tile flood logit)")
    ax.set_title("SegFormer — GradientSHAP per band", loc="left")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("band value (z-score)", color=MUTED)
    save_fig(fig, "figB1_gradientshap_segformer")

    np.savetxt(os.path.join(XCFG["fig_dir"], "importance_shap_dl.csv"),
               np.column_stack([np.array(names)[order], imp[order]]),
               fmt="%s", delimiter=",", header="feature,mean_abs_shap")
    return dict(zip(names, imp))


# ===========================================================================#
# PART C — Grad-CAM on the SegFormer decoder                                 #
# ===========================================================================#

def gradcam(model, x):
    """Grad-CAM at the decoder fusion layer: gradient of the mean flood logit
    w.r.t. the fused feature map -> channel weights -> weighted activation."""
    acts, grads = {}, {}
    h1 = model.head.fuse.register_forward_hook(            # capture activations
        lambda m, i, o: acts.update(a=o))
    h2 = model.head.fuse.register_full_backward_hook(      # capture gradients
        lambda m, gi, go: grads.update(g=go[0]))
    model.zero_grad()
    logit = model(x).mean()                                # scalar target
    logit.backward()                                       # populate gradients
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)          # GAP of gradients
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))     # weighted sum, ReLU
    cam = F.interpolate(cam, size=x.shape[2:],             # upsample to tile size
                        mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)  # normalize 0-1


def part_c_gradcam(model, ds, names):
    """Figure C1: rows of [terrain context | model prediction | Grad-CAM]."""
    n = XCFG["n_cam_examples"]
    try:
        elev_idx = names.index("swat_dem")                 # background = elevation
    except ValueError:
        elev_idx = 0
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.4 * n))
    for row in range(n):
        x, y, m = ds[row * max(len(ds) // n, 1)]           # spread examples out
        xb = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(xb))[0, 0].cpu().numpy()
        cam = gradcam(model, xb)
        bg = x[elev_idx].numpy()
        axes[row, 0].imshow(bg, cmap="Greys_r")            # context: terrain
        axes[row, 1].imshow(prob, cmap=SEQ_CMAP, vmin=0, vmax=1)  # prediction
        axes[row, 2].imshow(bg, cmap="Greys_r")            # CAM over context
        im = axes[row, 2].imshow(cam, cmap=SEQ_CMAP, alpha=0.55, vmin=0, vmax=1)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_title("Elevation (context)", loc="left")
    axes[0, 1].set_title("Predicted susceptibility", loc="left")
    axes[0, 2].set_title("Grad-CAM attention", loc="left")
    cb = fig.colorbar(im, ax=axes[:, 2], shrink=0.85, pad=0.02)
    cb.set_label("normalized attention / probability", color=MUTED)
    save_fig(fig, "figC1_gradcam_examples")


# ===========================================================================#
# PART D — Permutation importance on the deep model                          #
# ===========================================================================#

@torch.no_grad()
def batch_auc(model, tiles, ys, ms):
    """Masked AUC of the model over a fixed tile batch."""
    probs = []
    for i in range(0, len(tiles), 16):                     # mini-batches of 16
        xb = tiles[i:i + 16].to(DEVICE)
        probs.append(torch.sigmoid(model(xb)).squeeze(1).cpu())
    p = torch.cat(probs)
    sel = ms.bool()
    return rank_auc(p[sel].numpy(), ys[sel].numpy().astype(np.int8))


def part_d_permutation(model, tiles, ys, ms, names):
    """Scramble one band at a time across tiles; AUC drop = importance."""
    rng = np.random.default_rng(CFG["seed"])
    base = batch_auc(model, tiles, ys, ms)
    print(f"[D] baseline AUC on probe tiles: {base:.4f}")
    C = tiles.shape[1]
    drops = np.zeros((C, XCFG["n_perm_repeats"]))
    for c in range(C):                                     # each band...
        for r in range(XCFG["n_perm_repeats"]):            # ...several repeats
            perm = tiles.clone()
            idx = rng.permutation(len(tiles))              # shuffle band across tiles
            perm[:, c] = tiles[idx][:, c]                  # break band<->label link
            drops[c, r] = base - batch_auc(model, perm, ys, ms)
    mean_d, sd = drops.mean(1), drops.std(1)
    order = np.argsort(mean_d)[::-1][:XCFG["top_k"]]

    # ---- Figure D1: ranked AUC-drop bar chart with repeat error bars ---------
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh([names[i] for i in order][::-1], mean_d[order][::-1],
            xerr=sd[order][::-1], color=BAR_HUE, height=0.62,
            error_kw={"ecolor": MUTED, "elinewidth": 0.9})
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("AUC drop when band is permuted")
    ax.set_title("SegFormer — permutation feature importance", loc="left")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "figD1_permutation_importance")

    np.savetxt(os.path.join(XCFG["fig_dir"], "importance_permutation_dl.csv"),
               np.column_stack([np.array(names)[order], mean_d[order], sd[order]]),
               fmt="%s", delimiter=",", header="feature,mean_auc_drop,std")
    return dict(zip(names, mean_d))


# ===========================================================================#
# PART E — explanation consistency (bonus: novelty N2 in one figure)         #
# ===========================================================================#

def part_e_consistency(imp_a, imp_b, imp_d, names):
    """Rank-agreement scatter between the three importance rankings."""
    methods = {"TreeSHAP (XGB)": imp_a, "GradSHAP (DL)": imp_b,
               "Permutation (DL)": imp_d}
    methods = {k: v for k, v in methods.items() if v}      # keep available ones
    if len(methods) < 2:
        return
    keys = list(methods)
    def ranks(d):                                          # importance -> rank
        vals = np.array([d.get(n, 0) for n in names])
        return np.argsort(np.argsort(-vals))
    fig, axes = plt.subplots(1, len(keys) - 1,
                             figsize=(3.4 * (len(keys) - 1), 3.4), squeeze=False)
    r0 = ranks(methods[keys[0]])
    for j, k in enumerate(keys[1:]):
        rj, ax = ranks(methods[k]), axes[0, j]
        rho = 1 - 6 * ((r0 - rj) ** 2).sum() / (len(r0) * (len(r0) ** 2 - 1))
        ax.scatter(r0, rj, s=14, color=BAR_HUE, alpha=0.8, edgecolors="none")
        lim = len(names) - 1
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel(f"rank — {keys[0]}")
        ax.set_ylabel(f"rank — {k}")
        ax.set_title(f"Spearman ρ = {rho:.2f}", loc="left")
    fig.suptitle("Explanation consistency across XAI methods", x=0.02, ha="left")
    fig.tight_layout()
    save_fig(fig, "figE1_xai_consistency")


# ===========================================================================#
# MAIN                                                                        #
# ===========================================================================#

def main():
    names = band_names()
    os.makedirs(XCFG["fig_dir"], exist_ok=True)

    # ---- Part A runs on tabular data alone (no DL checkpoint needed) --------
    imp_a = part_a_treeshap(names)

    # ---- load model + probe tiles for parts B-D -----------------------------
    if not os.path.exists(XCFG["ckpt_path"]):
        print("No SegFormer checkpoint found -> DL parts skipped. "
              "Train segformer_flood.py first.")
        return
    ckpt = torch.load(XCFG["ckpt_path"], map_location=DEVICE, weights_only=False)
    model = SegFormerFlood(ckpt["in_ch"], ckpt["cfg"]).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()

    cube = np.load(CFG["cube_path"])
    valid = np.load(CFG["valid_path"])
    labels = load_labels(CFG["label_path"], valid.shape)
    rng = np.random.default_rng(CFG["seed"])
    splits = build_tiles(valid, labels, CFG, rng)
    ds = FloodTileDataset(cube, labels, valid, splits["test"],
                          CFG["tile"], augment=False, noise_std=0)
    n_probe = min(len(ds), XCFG["n_background"] + XCFG["n_explain"])
    probe = [ds[i] for i in range(n_probe)]                # fixed probe batch
    tiles = torch.stack([p[0] for p in probe])
    ys = torch.stack([p[1] for p in probe])
    ms = torch.stack([p[2] for p in probe])

    imp_b = part_b_gradientshap(model, tiles, names)       # Part B
    part_c_gradcam(model, ds, names)                       # Part C
    imp_d = part_d_permutation(model, tiles, ys, ms, names)  # Part D
    part_e_consistency(imp_a, imp_b, imp_d, names)         # Part E (bonus)
    print("All XAI figures written to", os.path.abspath(XCFG["fig_dir"]))


if __name__ == "__main__":
    main()

"""
Manuscript figures that need no further model runs:

  fig1_study_area        — hillshade + drainage + flood inventory + scale bar,
                           north arrow, lon/lat ticks  (paper Figure 1)
  fig2_inventory_events  — 2022-only / 2025-only / both-events flood extents
  fig_cv_splits          — AUC distribution across the 5 spatial CV splits
  fig_transfer           — temporal-transfer vs in-event AUC, both directions

All styled consistently with the existing figure set (Okabe-Ito, viridis,
recessive grid, 300-dpi PNG + vector PDF).
"""

import json

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch

from xai_analysis import save_fig, INK, BAR_HUE, ACCENT
import matplotlib as mpl

mpl.rcParams.update({
    "savefig.dpi": 400,
    "xtick.color": "#111111", "ytick.color": "#111111",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "text.color": "#111111",
})
MUTED = "#333333"                        # darker labels than the screen style

DS = 3                                   # finer display sampling (was 4)
PX = 30 * DS                             # metres per displayed pixel

# Okabe-Ito assignments (fixed): 2022-only orange, 2025-only sky blue,
# both-events vermillion (persistent susceptibility = strongest signal).
C2022, C2025, CBOTH = "#E69F00", "#56B4E9", "#D55E00"
MODEL_COLORS = {"RF": "#0072B2", "XGBoost": "#D55E00", "LightGBM": "#009E73"}


def read(path, band=1):
    with rasterio.open(path) as src:
        a = src.read(band).astype("float32")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        return a[::DS, ::DS], src.transform, str(src.crs)


def lonlat_ticks(ax, transform, crs, shape):
    """Label axes in geographic degrees despite the UTM grid."""
    from pyproj import Transformer
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    H, W = shape
    xs = [transform * (c * DS, H * DS // 2) for c in np.linspace(0, W, 5)]
    ys = [transform * (W * DS // 2, r * DS) for r in np.linspace(0, H, 5)]
    lon = [t.transform(x, y)[0] for x, y in xs]
    lat = [t.transform(x, y)[1] for x, y in ys]
    ax.set_xticks(np.linspace(0, W, 5))
    ax.set_xticklabels([f"{v:.1f}°E" for v in lon], fontsize=7)
    ax.set_yticks(np.linspace(0, H, 5))
    ax.set_yticklabels([f"{v:.1f}°N" for v in lat], fontsize=7)
    ax.grid(False)


def scalebar_north(ax, shape):
    H, W = shape
    bar_px = 20000 / PX                                  # 20 km bar
    x0, y0 = W * 0.05, H * 0.95
    ax.plot([x0, x0 + bar_px], [y0, y0], color=INK, lw=2.5)
    ax.text(x0 + bar_px / 2, y0 - H * 0.015, "20 km", ha="center",
            fontsize=7, color=INK)
    ax.annotate("N", xy=(W * 0.95, H * 0.06), xytext=(W * 0.95, H * 0.13),
                ha="center", fontsize=9, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))


def hillshade():
    dem, transform, crs = read("../data/raw/swat_dem.tif")
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem)),
                        vert_exag=0.05), dem, transform, crs


def fig1_study_area():
    hs, dem, transform, crs = hillshade()
    facc, _, _ = read("../data/raw/swat_logfacc.tif")
    with rasterio.open("../data/labels/flood_labels.tif") as src:
        lab = src.read(1)[::DS, ::DS]
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.imshow(hs, cmap="Greys_r", alpha=0.9)
    # tint elevation lightly over the hillshade for orientation
    ax.imshow(dem, cmap="terrain", alpha=0.25)
    # drainage: larger channels only, thin blue
    streams = np.where(facc > 4, 1.0, np.nan)            # ln(upa+1) > 4 ≈ 55 km²
    ax.imshow(streams, cmap=ListedColormap(["#2166ac"]), alpha=0.8)
    # flood inventory (combined events)
    flood = np.where(lab == 1, 1.0, np.nan)
    ax.imshow(flood, cmap=ListedColormap([CBOTH]), alpha=0.95)
    lonlat_ticks(ax, transform, crs, hs.shape)
    scalebar_north(ax, hs.shape)
    ax.set_title("Study area — Swat basin region, Khyber Pakhtunkhwa, Pakistan",
                 loc="left")
    ax.legend(handles=[
        Patch(color="#2166ac", label="Drainage network (>55 km² catchment)"),
        Patch(color=CBOTH, label="Flood inventory 2022+2025 (Sentinel-1)")],
        loc="lower right", fontsize=7)
    save_fig(fig, "fig1_study_area")


def fig2_inventory_events():
    from scipy.ndimage import binary_dilation
    hs, _, transform, crs = hillshade()
    with rasterio.open("../data/labels/flood_labels_2022.tif") as s:
        f22 = s.read(1)[::DS, ::DS] == 1
    with rasterio.open("../data/labels/flood_labels_2025.tif") as s:
        f25 = s.read(1)[::DS, ::DS] == 1
    # dilate thin/sparse flood pixels so they are visible over the hillshade
    st = np.ones((3, 3), bool)
    f22, f25 = binary_dilation(f22, st, 2), binary_dilation(f25, st, 2)
    cat = np.full(f22.shape, np.nan, "float32")
    cat[f22 & ~f25] = 0                                   # 2022 only
    cat[f25 & ~f22] = 1                                   # 2025 only
    cat[f22 & f25] = 2                                    # both events
    fig, ax = plt.subplots(figsize=(8.6, 10.2))          # large scale
    ax.imshow(hs, cmap="Greys_r", alpha=0.70)
    ax.imshow(cat, cmap=ListedColormap([C2022, C2025, CBOTH]),
              vmin=0, vmax=2, interpolation="nearest")
    lonlat_ticks(ax, transform, crs, hs.shape)
    scalebar_north(ax, hs.shape)
    n22, n25, nb = int((f22 & ~f25).sum()), int((f25 & ~f22).sum()), int((f22 & f25).sum())
    tot = n22 + n25 + nb
    ax.legend(handles=[
        Patch(color=C2022, label=f"2022 only ({100*n22/tot:.0f}%)"),
        Patch(color=C2025, label=f"2025 only ({100*n25/tot:.0f}%)"),
        Patch(color=CBOTH, label=f"Both events ({100*nb/tot:.0f}%)")],
        loc="lower right", fontsize=7, title="Flood extent", title_fontsize=7)
    ax.set_title("Sentinel-1 flood inventory by event", loc="left")
    save_fig(fig, "fig2_inventory_events")


def fig_cv_splits():
    """Per-split AUC, the mean, and the ORIGINAL single split for comparison.

    The single-split value is the point of the figure: the manuscript argues
    that the headline numbers came from a fortunate draw of test blocks, and
    that claim is only checkable if the original value is drawn alongside the
    resampled distribution. Jitter is seeded so the figure is reproducible.
    """
    from matplotlib.lines import Line2D
    cv = json.load(open("../runs/experiments/cv_results.json"))
    # single-split ROC-AUC from the baseline table (paper Table 2)
    single = {"RF": 0.9316, "XGBoost": 0.9510, "LightGBM": 0.9527}

    rng = np.random.default_rng(0)                 # reproducible jitter
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    for i, (m, d) in enumerate(cv.items()):
        vals = d["per_split"]
        x = np.full(len(vals), i) + (rng.random(len(vals)) - 0.5) * 0.17
        ax.scatter(x, vals, s=30, color=MODEL_COLORS[m], alpha=0.85,
                   edgecolors="none", zorder=3)
        ax.plot([i - 0.22, i + 0.22], [d["mean"]] * 2, color=INK, lw=1.6,
                zorder=4)
        # the original single split, above every resampled value
        ax.plot(i, single[m], marker="D", ms=6.5, mfc="none", mec=INK,
                mew=1.4, zorder=5)
        ax.annotate("", xy=(i + 0.30, single[m]), xytext=(i + 0.30, d["mean"]),
                    arrowprops=dict(arrowstyle="<->", color="#777777", lw=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=7),
                    zorder=4)
        ax.text(i + 0.34, (single[m] + d["mean"]) / 2,
                f"+{single[m] - d['mean']:.3f}", ha="left", va="center",
                fontsize=7, color="#555555")

    ax.set_xticks(range(len(cv)))
    ax.set_xticklabels(list(cv))
    ax.set_xlim(-0.45, len(cv) - 0.30)
    ax.set_ylim(0.882, 0.962)
    ax.set_ylabel("test AUC")
    ax.set_title("AUC across five spatial block splits", loc="left")
    ax.grid(axis="x", visible=False)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=6, color="#777777",
               label="one spatial split (5 per model)"),
        Line2D([], [], ls="-", lw=1.6, color=INK, label="mean of the 5 splits"),
        Line2D([], [], marker="D", ls="none", ms=6.5, mfc="none", mec=INK,
               mew=1.4, label="original single split (Table 2)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, fontsize=7.4, frameon=False, handlelength=1.4,
              columnspacing=1.3, handletextpad=0.5)
    save_fig(fig, "fig_cv_splits")


def fig_transfer():
    """Paired bars per direction. The quantity of interest is the GAP between
    the pair, so both bars are labelled and the gap is drawn explicitly.
    The legend sits below the axes: placed inside, it landed on top of the
    tall reference bar and became unreadable."""
    GREY = "#6E6E6E"                    # genuinely grey, as the caption says
    tr = json.load(open("../runs/experiments/transfer_results.json"))
    dirs = ["train2022_test2025", "train2025_test2022"]
    labels = ["train 2022 →\ntest 2025", "train 2025 →\ntest 2022"]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.bar(x - 0.18, [tr[d]["auc_in_event"] for d in dirs], width=0.32,
           color=GREY, label="in-event reference")
    ax.bar(x + 0.18, [tr[d]["auc_transfer"] for d in dirs], width=0.32,
           color=BAR_HUE, label="cross-event transfer")

    for xi, d in zip(x, dirs):
        hi, lo = tr[d]["auc_in_event"], tr[d]["auc_transfer"]
        # values sit INSIDE the bars in white, leaving the space above the
        # bars free for the gap bracket
        ax.text(xi - 0.18, hi - 0.016, f"{hi:.3f}", ha="center", va="top",
                fontsize=7.5, color="white", fontweight="bold", zorder=5)
        ax.text(xi + 0.18, lo - 0.016, f"{lo:.3f}", ha="center", va="top",
                fontsize=7.5, color="white", fontweight="bold", zorder=5)
        # gap bracket: works at any gap size, unlike a double-headed arrow,
        # which collapses to a few pixels when the loss is only 0.009
        top = hi + 0.032
        ax.plot([xi - 0.18, xi - 0.18, xi + 0.18, xi + 0.18],
                [hi + 0.006, top, top, lo + 0.006],
                color=INK, lw=0.9, solid_capstyle="round", zorder=4)
        ax.text(xi, top + 0.008, f"$-${hi - lo:.3f} AUC", ha="center",
                va="bottom", fontsize=8, color=INK, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("test AUC (spatially disjoint blocks)")
    ax.set_title("Temporal transferability (LightGBM)", loc="left")
    # below the axes, clear of every bar
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              fontsize=8, frameon=False, handlelength=1.4,
              columnspacing=1.6, handletextpad=0.6)
    ax.grid(axis="x", visible=False)
    save_fig(fig, "fig_transfer")


if __name__ == "__main__":
    fig_cv_splits()
    fig_transfer()
    fig2_inventory_events()
    fig1_study_area()
    print("all figures done")

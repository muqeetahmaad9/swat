"""
Remaining manuscript figures (no model runs needed):

  fig3_factor_panel   — 3x3 grid of key conditioning factors
  fig4_flowchart      — methodology flowchart (data -> labels -> models -> XAI)
  fig1_study_area     — regenerated WITH a Pakistan inset locator
"""

import json

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch, Rectangle

from xai_analysis import save_fig, INK, BAR_HUE, ACCENT
from make_figures import (read, lonlat_ticks, scalebar_north, hillshade,
                          DS, CBOTH)
import matplotlib as mpl

mpl.rcParams.update({
    "savefig.dpi": 400,
    "xtick.color": "#111111", "ytick.color": "#111111",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "text.color": "#111111",
})
MUTED = "#333333"

# --------------------------------------------------------------------------- #
# fig3 — factor panel                                                          #
# --------------------------------------------------------------------------- #

FACTORS = [  # (file, label, cmap, robust percentile clip)
    ("swat_dem.tif",       "Elevation (m)",             "terrain", (2, 98)),
    ("swat_slope.tif",     "Slope (°)",                 "viridis", (2, 98)),
    ("swat_hand.tif",      "HAND (m)",                  "viridis", (2, 90)),
    ("swat_twi.tif",       "TWI (–)",                   "viridis", (2, 98)),
    ("swat_distriver.tif", "Distance to drainage (m)",  "viridis", (2, 98)),
    ("swat_draindens.tif", "Drainage density (–)",      "viridis", (2, 98)),
    ("swat_rainfall.tif",  "Mean annual rainfall (mm)", "viridis", (2, 98)),
    ("swat_s1_vh.tif",     "Sentinel-1 VH (dB)",        "viridis", (2, 98)),
    ("swat_ndvi.tif",      "NDVI (–)",                  "YlGn",    (2, 98)),
]


def fig3_factor_panel():
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 9.2))
    for ax, (fname, label, cmap, clip) in zip(axes.ravel(), FACTORS):
        a, _, _ = read(f"../data/raw/{fname}")
        vmin, vmax = np.nanpercentile(a, clip)
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(label, fontsize=9, loc="left", color="#111111",
                     fontweight="bold")
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.ax.tick_params(labelsize=8, colors="#111111")
    fig.suptitle("Selected flood-conditioning factors", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", color="#111111")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_fig(fig, "fig3_factor_panel")


# --------------------------------------------------------------------------- #
# fig4 — methodology flowchart                                                 #
# --------------------------------------------------------------------------- #

def box(ax, xy, w, h, text, fc="#eef2f7", ec=MUTED, fs=7.6, bold=False):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                       fc=fc, ec=ec, lw=1.0)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, fontweight="bold" if bold else "normal")
    return b


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 color=MUTED, lw=1.1))


def fig4_flowchart():
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    # column 1 — data sources
    box(ax, (0.2, 5.6), 2.2, 0.9, "Sentinel-1 SAR\n(VV, VH, multi-temporal)",
        fc="#dce9f5")
    box(ax, (0.2, 4.4), 2.2, 0.9, "Sentinel-2 optical\n(bands + indices)",
        fc="#dce9f5")
    box(ax, (0.2, 3.2), 2.2, 0.9, "Terrain & static\n(DEM, HAND, TWI, TPI,\nLULC, rainfall …)",
        fc="#dce9f5")
    box(ax, (0.2, 1.2), 2.2, 1.2, "Flood labels\nper-orbit S1 change\ndetection, 2022 + 2025",
        fc="#fdeadd")

    # column 2 — preprocessing
    box(ax, (3.1, 3.6), 2.1, 1.6,
        "Preprocessing\n30 m grid · z-score\nsin/cos · one-hot\n36-band cube", bold=True)
    box(ax, (3.1, 1.2), 2.1, 1.2, "Sampling\nfloodable domain\n10k + 10k points\nspatial block split")

    # column 3 — models
    box(ax, (5.9, 4.9), 1.9, 1.1, "Ensembles\nRF · XGBoost\nLightGBM")
    box(ax, (5.9, 3.5), 1.9, 1.1, "Patch CNN\n32×32 context")
    box(ax, (5.9, 2.1), 1.9, 1.1, "SegFormer\n+ multi-branch\nfusion")
    box(ax, (5.9, 0.5), 1.9, 1.1, "Evaluation\nrepeated spatial CV\nDeLong · transfer",
        fc="#fdeadd")

    # column 4 — XAI & outputs
    box(ax, (8.35, 4.9), 1.5, 1.1, "TreeSHAP")
    box(ax, (8.35, 3.5), 1.5, 1.1, "IG · Grad-CAM\npermutation")
    box(ax, (8.35, 2.1), 1.5, 1.1, "Consistency\nanalysis (ρ)", fc="#fdeadd")
    box(ax, (8.35, 0.5), 1.5, 1.1, "Susceptibility\nmaps + attribution",
        fc="#e3f1e6", bold=True)

    # arrows
    for y in (6.05, 4.85, 3.65):
        arrow(ax, (2.4, y), (3.1, 4.4))
    arrow(ax, (2.4, 1.8), (3.1, 1.8))
    arrow(ax, (5.2, 4.4), (5.9, 5.45))
    arrow(ax, (5.2, 4.2), (5.9, 4.05))
    arrow(ax, (5.2, 4.0), (5.9, 2.65))
    arrow(ax, (4.15, 3.6), (4.15, 2.4))
    arrow(ax, (5.2, 1.8), (5.9, 1.05))
    arrow(ax, (7.8, 5.45), (8.35, 5.45))
    arrow(ax, (7.8, 4.05), (8.35, 4.05))
    arrow(ax, (9.1, 4.9), (9.1, 3.2 + 1.4))   # SHAP -> consistency
    arrow(ax, (9.1, 3.5), (9.1, 3.2))
    arrow(ax, (9.1, 2.1), (9.1, 1.6))
    arrow(ax, (7.8, 1.05), (8.35, 1.05))
    ax.set_title("Methodological framework", loc="left")
    save_fig(fig, "fig4_flowchart")


# --------------------------------------------------------------------------- #
# fig1 — study area, regenerated with Pakistan inset                           #
# --------------------------------------------------------------------------- #

def pakistan_rings():
    gj = json.load(open("../data/ne_countries.geojson", encoding="utf-8"))
    for feat in gj["features"]:
        if feat["properties"].get("ADMIN") == "Pakistan":
            geom = feat["geometry"]
            if geom["type"] == "Polygon":
                return [geom["coordinates"][0]]
            return [poly[0] for poly in geom["coordinates"]]
    raise KeyError("Pakistan not found in Natural Earth file")


def kashmir_disputed_rings():
    """Disputed Kashmir-region polygons (Natural Earth 10m disputed areas):
    Jammu & Kashmir, Aksai Chin, Shaksam Valley, Siachen — drawn hatched with
    dashed edges (neither asserted to any country)."""
    keep = {"Jammu and Kashmir"}          # IoK only — single dotted outline
    gj = json.load(open("../data/ne_disputed10.geojson", encoding="utf-8"))
    rings = []
    for feat in gj["features"]:
        if feat["properties"].get("BRK_NAME") in keep:
            geom = feat["geometry"]
            if geom["type"] == "Polygon":
                rings.append(geom["coordinates"][0])
            else:
                rings += [poly[0] for poly in geom["coordinates"]]
    return rings


def fig1_with_inset():
    hs, dem, transform, crs = hillshade()
    facc, _, _ = read("../data/raw/swat_logfacc.tif")
    with rasterio.open("../data/labels/flood_labels.tif") as src:
        lab = src.read(1)[::DS, ::DS]
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.imshow(hs, cmap="Greys_r", alpha=0.9)
    ax.imshow(dem, cmap="terrain", alpha=0.25)
    ax.imshow(np.where(facc > 4, 1.0, np.nan),
              cmap=ListedColormap(["#2166ac"]), alpha=0.8)
    ax.imshow(np.where(lab == 1, 1.0, np.nan),
              cmap=ListedColormap([CBOTH]), alpha=0.95)
    lonlat_ticks(ax, transform, crs, hs.shape)
    scalebar_north(ax, hs.shape)
    ax.set_title("Study area — Swat basin region, Khyber Pakhtunkhwa, Pakistan",
                 loc="left")
    ax.legend(handles=[
        Patch(color="#2166ac", label="Drainage network (>55 km² catchment)"),
        Patch(color=CBOTH, label="Flood inventory 2022+2025 (Sentinel-1)")],
        loc="lower right", fontsize=7)

    # ---- inset locator: Pakistan outline + AOI rectangle -------------------
    axi = ax.inset_axes([0.015, 0.015, 0.30, 0.26])
    for ring in pakistan_rings():
        xs, ys = zip(*ring)
        axi.fill(xs, ys, fc="#e8e8e8", ec=MUTED, lw=0.7)
    # IoK (Jammu & Kashmir): same fill as Pakistan, dotted boundary only
    for ring in kashmir_disputed_rings():
        xs, ys = zip(*ring)
        axi.fill(xs, ys, fc="#e8e8e8", ec=MUTED, lw=0.9, ls=":")
    axi.add_patch(Rectangle((71.0, 34.0), 1.9, 1.95, fc="none",
                            ec=ACCENT, lw=1.4))
    axi.text(70.0, 26.0, "Pakistan", fontsize=6.5, color=MUTED, ha="center")
    axi.set_xlim(60, 81.5); axi.set_ylim(22, 38)
    axi.set_xticks([]); axi.set_yticks([]); axi.grid(False)
    for s in axi.spines.values():
        s.set_color(MUTED)
    save_fig(fig, "fig1_study_area")


if __name__ == "__main__":
    fig3_factor_panel()
    fig4_flowchart()
    fig1_with_inset()
    print("all done")

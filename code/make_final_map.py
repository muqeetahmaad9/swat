"""
Final classified susceptibility map (paper's closing figure):
5 classes (quantile breaks within the floodable domain), masked outside it,
with a class-area inset table printed for the manuscript.

Default source: the XGBoost 36-band map (susceptibility_xgb.tif). Switch
SOURCE to the SegFormer or multi-branch .npy if that becomes the final model.

    python make_final_map.py
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from xai_analysis import save_fig, MUTED
from make_figures import lonlat_ticks, scalebar_north, DS
from segformer_flood import CFG, load_labels

SOURCE = "../data/processed/susceptibility_xgb.tif"     # or runs/*/susceptibility.npy

# 5-class sequential scheme (single-hue ramp, light -> dark; CVD-safe)
CLASS_COLORS = ["#d3e3f0", "#bdc9e1", "#74a9cf", "#2b8cbe", "#045a8d"]
CLASS_NAMES = ["Very low", "Low", "Moderate", "High", "Very high"]


def main():
    if SOURCE.endswith(".tif"):
        with rasterio.open(SOURCE) as src:
            prob = src.read(1)
            transform, crs = src.transform, str(src.crs)
    else:
        prob = np.load(SOURCE)
        with rasterio.open("../data/raw/swat_dem.tif") as src:
            transform, crs = src.transform, str(src.crs)

    valid = np.load(CFG["valid_path"].replace("..", ".."))
    labels = load_labels("../data/labels/flood_labels.tif", prob.shape)
    floodable = np.isfinite(labels)
    prob = np.where(floodable, prob, np.nan)

    # quantile class breaks WITHIN the floodable domain
    vals = prob[np.isfinite(prob)]
    breaks = np.nanquantile(vals, [0.2, 0.4, 0.6, 0.8])
    bounds = [0] + list(breaks) + [1.0001]
    classes = np.digitize(prob, breaks)                      # 0..4
    classes = np.where(np.isfinite(prob), classes, np.nan)

    # class areas (km2) for the manuscript table
    print("class breaks (P):", [f"{b:.3f}" for b in breaks])
    for c, name in enumerate(CLASS_NAMES):
        km2 = np.nansum(classes == c) * 900 / 1e6
        print(f"  {name:10s}: {km2:8.1f} km2")

    d = classes[::DS, ::DS]
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    cmap = ListedColormap(CLASS_COLORS)
    ax.imshow(np.where(np.isnan(d), 0.0, np.nan),                 # grey where NOT assessed
              cmap=ListedColormap(["#e6e6e6"]))
    ax.imshow(d, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    lonlat_ticks(ax, transform, crs, d.shape)
    scalebar_north(ax, d.shape)
    handles = [Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    handles.append(Patch(color="#e6e6e6", label="Not assessed"))
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              title="Susceptibility", title_fontsize=7)
    ax.set_title("Flood susceptibility â€” classified (floodable domain)",
                 loc="left")
    save_fig(fig, "fig_final_classified_map")


if __name__ == "__main__":
    main()

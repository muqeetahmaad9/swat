"""
Multi-panel study-area figure (Figure 1), in the style of top FSM papers:
  (a) Pakistan with Khyber Pakhtunkhwa highlighted + study box
  (b) Khyber Pakhtunkhwa province with study box
  (c) Elevation (DEM) with flood / non-flood sample points
  (d) Land use (ESA WorldCover)
Overwrites fig1_study_area.
"""

import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch
from pyproj import Transformer

from xai_analysis import save_fig, INK, MUTED
import matplotlib as mpl

# darker, higher-resolution output (override the muted screen style)
mpl.rcParams.update({
    "savefig.dpi": 400,
    "xtick.color": "#111111", "ytick.color": "#111111",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "text.color": "#111111",
})
MUTED = "#333333"                        # darker "muted" for labels in this fig

DS = 3                                   # finer display sampling (was 6) -> sharper
AOI = (71.0, 34.0, 72.9, 35.95)          # lon/lat bounds of the study rectangle

# ESA WorldCover class -> (label, colour)
WC = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"),
      30: ("Grassland", "#ffff4c"), 40: ("Cropland", "#f096ff"),
      50: ("Built-up", "#fa0000"), 60: ("Bare / sparse", "#b4b4b4"),
      70: ("Snow / ice", "#f0f0f0"), 80: ("Water", "#0064c8"),
      90: ("Wetland", "#0096a0"), 95: ("Mangrove", "#00cf75"),
      100: ("Moss / lichen", "#fae6a0")}


def read(path, band=1):
    with rasterio.open(path) as src:
        a = src.read(band, out_shape=(src.height // DS, src.width // DS)).astype("float32")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        return a, src.transform, str(src.crs), src


def provinces(admin, names=None):
    gj = json.load(open("../data/ne_provinces.geojson", encoding="utf-8"))
    out = []
    for f in gj["features"]:
        if f["properties"].get("admin") == admin:
            if names is None or f["properties"].get("name") in names:
                g = f["geometry"]
                polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
                out.append((f["properties"].get("name"), polys))
    return out


def draw_polys(ax, polys, fc, ec, lw=0.6, alpha=1.0, ls="solid"):
    for _, multi in polys:
        for poly in multi:
            ring = poly[0] if isinstance(poly[0][0], (list, tuple)) else poly
            xs, ys = zip(*[(p[0], p[1]) for p in ring])
            ax.fill(xs, ys, facecolor=fc, edgecolor=ec, lw=lw, alpha=alpha,
                    linestyle=ls)


def kashmir_rings():
    """Disputed Jammu & Kashmir polygon (Natural Earth disputed areas)."""
    gj = json.load(open("../data/ne_disputed10.geojson", encoding="utf-8"))
    out = []
    for f in gj["features"]:
        if f["properties"].get("BRK_NAME") == "Jammu and Kashmir":
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            out.append((None, polys))
    return out


def main():
    dem, transform, crs, src = read("../data/raw/swat_dem.tif")
    H, W = dem.shape
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    fig = plt.figure(figsize=(8.6, 10.2))          # taller -> fills the page
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1],
                           wspace=0.06, hspace=0.10,
                           left=0.02, right=0.98, top=0.98, bottom=0.02)

    # ---- (a) Pakistan, KP highlighted ------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    pak = provinces("Pakistan")
    draw_polys(axa, pak, "#ececec", MUTED, 0.4)
    # disputed Jammu & Kashmir: same fill as Pakistan, dotted boundary only
    draw_polys(axa, kashmir_rings(), "#ececec", MUTED, 0.7, ls=(0, (1, 1)))
    draw_polys(axa, provinces("Pakistan", {"K.P."}), "#f4a582", "#b2182b", 0.6)
    axa.add_patch(Rectangle((AOI[0], AOI[1]), AOI[2]-AOI[0], AOI[3]-AOI[1],
                            fill=False, ec="#000000", lw=1.2))
    axa.set_xlim(60, 80); axa.set_ylim(23, 38)
    axa.set_title("(a) Pakistan", loc="left", fontsize=9, fontweight="bold")
    axa.text(69, 30.5, "Khyber\nPakhtunkhwa", fontsize=6, color="#b2182b", ha="center")
    axa.set_xticks([]); axa.set_yticks([]); axa.set_aspect("equal"); axa.grid(False)

    # ---- (b) KP province with study box ----------------------------------
    axb = fig.add_subplot(gs[1, 0])
    kp = provinces("Pakistan", {"K.P."})
    draw_polys(axb, provinces("Pakistan"), "#f2f2f2", MUTED, 0.3)
    draw_polys(axb, kp, "#d1e5f0", "#2166ac", 0.7)
    axb.add_patch(Rectangle((AOI[0], AOI[1]), AOI[2]-AOI[0], AOI[3]-AOI[1],
                            fill=False, ec="#d6273e", lw=1.6))
    # zoom to KP bounds
    allx = [p[0] for _, m in kp for poly in m for ring in [poly[0] if isinstance(poly[0][0],(list,tuple)) else poly] for p in ring]
    ally = [p[1] for _, m in kp for poly in m for ring in [poly[0] if isinstance(poly[0][0],(list,tuple)) else poly] for p in ring]
    axb.set_xlim(min(allx)-0.3, max(allx)+0.3); axb.set_ylim(min(ally)-0.3, max(ally)+0.3)
    axb.set_title("(b) Khyber Pakhtunkhwa", loc="left", fontsize=9, fontweight="bold")
    axb.text((AOI[0]+AOI[2])/2, AOI[3]+0.25, "study area", fontsize=6,
             color="#d6273e", ha="center")
    axb.set_xticks([]); axb.set_yticks([]); axb.set_aspect("equal"); axb.grid(False)

    # ---- (c) elevation + flood / non-flood points ------------------------
    axc = fig.add_subplot(gs[0, 1])
    im = axc.imshow(dem, cmap="terrain")
    # sample points (subset for legibility)
    pts = np.load("../data/processed/points_test.npy")
    y = np.load("../data/processed/y_test.npy")
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pts), min(500, len(pts)), replace=False)
    pr, yr = pts[sel], y[sel]
    fl = pr[yr == 1]; nf = pr[yr == 0]
    axc.scatter(nf[:, 1]/DS, nf[:, 0]/DS, s=5, c="#3b3b3b", marker="o",
                linewidths=0, label="Non-flood point", alpha=0.7)
    axc.scatter(fl[:, 1]/DS, fl[:, 0]/DS, s=6, c="#d6273e", marker="o",
                linewidths=0, label="Flood point", alpha=0.85)
    axc.set_title("(c) Elevation and flood inventory", loc="left",
                  fontsize=9, fontweight="bold")
    axc.legend(loc="upper left", fontsize=6, framealpha=0.92, markerscale=1.6,
               borderpad=0.4)
    cb = fig.colorbar(im, ax=axc, shrink=0.7, pad=0.02)
    cb.set_label("Elevation (m)", fontsize=7); cb.ax.tick_params(labelsize=6)
    # north arrow
    axc.annotate("N", xy=(W*0.94, H*0.10), xytext=(W*0.94, H*0.24),
                 ha="center", fontsize=8, fontweight="bold", color=INK,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    # scale bar (20 km)
    px20 = 20000 / (30*DS)
    axc.plot([W*0.06, W*0.06+px20], [H*0.93, H*0.93], color=INK, lw=2.2)
    axc.text(W*0.06+px20/2, H*0.90, "20 km", ha="center", fontsize=6)
    axc.set_xticks([]); axc.set_yticks([]); axc.grid(False)

    # ---- (d) land use ----------------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    lulc, ltr, lcrs, _ = read("../data/raw/swat_lulc.tif")
    present = [c for c in WC if np.any(lulc == c)]
    cmap = ListedColormap([WC[c][1] for c in present])
    norm = BoundaryNorm([present[0]-1] + [c+0.5 for c in present], cmap.N)
    remap = np.full_like(lulc, np.nan)
    for i, c in enumerate(present):
        remap[lulc == c] = c
    axd.imshow(remap, cmap=cmap, norm=norm, interpolation="nearest")
    axd.set_title("(d) Land use / land cover", loc="left",
                  fontsize=9, fontweight="bold")
    axd.legend(handles=[Patch(color=WC[c][1], label=WC[c][0]) for c in present],
               loc="lower left", fontsize=5.5, ncol=2, framealpha=0.9)
    axd.annotate("N", xy=(lulc.shape[1]*0.94, lulc.shape[0]*0.10),
                 xytext=(lulc.shape[1]*0.94, lulc.shape[0]*0.24), ha="center",
                 fontsize=8, fontweight="bold", color=INK,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    axd.set_xticks([]); axd.set_yticks([]); axd.grid(False)

    save_fig(fig, "fig1_study_area")


if __name__ == "__main__":
    main()

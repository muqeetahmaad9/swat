"""Methodology flowchart v3 — rebuilt.

Design changes over v2:
  * orthogonal (right-angle) connectors only; no crossing arcs
  * merge/split "bus" rails so many-to-one links read as one flow
  * corrected logic: models feed Evaluation (v2 routed it from the dataset)
  * balanced grid — the three predictor sources align row-wise with the three
    model families, and the label path runs along a dedicated bottom lane
  * the dead space at lower-left now carries the leakage note (the two
    Sentinel-1 boxes use disjoint time windows), which is a real claim in
    Section 3 rather than decoration
  * stage headers are filled bands, not hairlines

Overwrites fig4_flowchart.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from xai_analysis import save_fig, INK, MUTED

# ---------------------------------------------------------------- palette --
C_DATA  = dict(fc="#E7F0F9", ec="#7FA6CC")   # satellite / terrain inputs
C_PROC  = dict(fc="#EEF1F5", ec="#A3AEBD")   # dataset construction
C_LABEL = dict(fc="#FDF0E2", ec="#D6A472")   # inventory + evaluation
C_MODEL = dict(fc="#E8F3F1", ec="#7EB0A7")   # model families
C_XAI   = dict(fc="#F1EBF8", ec="#AA93CC")   # explainability
C_OUT   = dict(fc="#E5F2E8", ec="#82B891")   # products

ARROW = "#57616F"
BAND  = "#F4F6F8"

FS = 8.0                                     # base font size

# ------------------------------------------------------------------ grid --
CW, CH = 12.60, 7.80                         # canvas
MARGIN, GAP = 0.40, 0.55
W = (CW - 2 * MARGIN - 3 * GAP) / 4          # column / box width
X = [MARGIN + i * (W + GAP) for i in range(4)]

ROW = [6.05, 4.90, 3.75]                     # three aligned content rows
LANE = 1.55                                  # bottom lane (labels/eval/out)
H, HT = 0.92, 1.06                           # standard / tall box height


# -------------------------------------------------------------- drawing ---
def box(ax, col, yc, lines, style, h=H):
    """Rounded box centred at (column, yc). lines[0] is the bold title."""
    x, y = X[col], yc - h / 2
    ax.add_patch(FancyBboxPatch((x, y), W, h, boxstyle="round,pad=0.015",
                                lw=1.1, zorder=3, **style))
    title, *rest = lines
    if rest:
        ax.text(x + W / 2, y + h - 0.28, title, ha="center", va="center",
                fontsize=FS, color=INK, fontweight="bold", zorder=4)
        ax.text(x + W / 2, y + (h - 0.46) / 2, "\n".join(rest), ha="center",
                va="center", fontsize=FS - 1.2, color="#3C4453", zorder=4,
                linespacing=1.45)
    else:
        ax.text(x + W / 2, y + h / 2, title, ha="center", va="center",
                fontsize=FS, color=INK, fontweight="bold", zorder=4)
    return dict(x=x, y=y, w=W, h=h, cx=x + W / 2, cy=yc,
                l=x, r=x + W, t=y + h, b=y)


def path(ax, pts, lw=1.15, color=ARROW, dashed=False, head=True):
    """Orthogonal polyline; arrowhead on the final segment."""
    ls = (0, (3.5, 2.5)) if dashed else "-"
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, ls=ls, solid_capstyle="round",
                zorder=2, clip_on=False)
    style = "-|>" if head else "-"
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=11))


def merge(ax, sources, target, rail_x):
    """Many right-edges -> vertical rail -> single arrow into target's left."""
    ys = [s["cy"] for s in sources]
    for s in sources:                                   # stubs out to the rail
        ax.plot([s["r"], rail_x], [s["cy"], s["cy"]],
                color=ARROW, lw=1.15, zorder=2, solid_capstyle="round")
    ax.plot([rail_x, rail_x], [min(ys), max(ys)],       # the rail itself
            color=ARROW, lw=1.15, zorder=2, solid_capstyle="round")
    path(ax, [(rail_x, target["cy"]), (target["l"], target["cy"])])


def split(ax, source, targets, rail_x):
    """Single right-edge -> vertical rail -> arrows into each target."""
    ys = [t["cy"] for t in targets]
    ax.plot([source["r"], rail_x], [source["cy"], source["cy"]],
            color=ARROW, lw=1.15, zorder=2, solid_capstyle="round")
    ax.plot([rail_x, rail_x], [min(ys + [source["cy"]]),
                               max(ys + [source["cy"]])],
            color=ARROW, lw=1.15, zorder=2, solid_capstyle="round")
    for t in targets:
        path(ax, [(rail_x, t["cy"]), (t["l"], t["cy"])])


def vlink(ax, a, b):
    """Straight vertical arrow between two boxes in the same column."""
    if a["cy"] > b["cy"]:
        path(ax, [(a["cx"], a["b"]), (b["cx"], b["t"])])
    else:
        path(ax, [(a["cx"], a["t"]), (b["cx"], b["b"])])


def hlink(ax, a, b):
    """Straight horizontal arrow, same row."""
    path(ax, [(a["r"], a["cy"]), (b["l"], b["cy"])])


def stage(ax, col, text):
    ax.add_patch(Rectangle((X[col], 6.98), W, 0.40, facecolor="#E4E9EF",
                           edgecolor="none", zorder=1))
    ax.text(X[col] + W / 2, 7.18, text, ha="center", va="center",
            fontsize=8.3, color="#2B3440", fontweight="bold", zorder=2)


# ------------------------------------------------------------------ main --
def main():
    fig, ax = plt.subplots(figsize=(CW, CH))
    ax.set_xlim(0, CW); ax.set_ylim(0, CH); ax.axis("off")

    # faint column bands so the four stages read as groups
    for i in range(4):
        ax.add_patch(Rectangle((X[i] - 0.13, 0.72), W + 0.26, 6.20,
                               facecolor=BAND, edgecolor="none", zorder=0))

    stage(ax, 0, "1  DATA SOURCES")
    stage(ax, 1, "2  DATASET CONSTRUCTION")
    stage(ax, 2, "3  MODELLING & EVALUATION")
    stage(ax, 3, "4  EXPLAINABILITY & PRODUCTS")

    # -- 1. inputs ----------------------------------------------------------
    s1 = box(ax, 0, ROW[0], ["Sentinel-1 SAR  (2023)",
                             "VV · VH annual median",
                             "predictor composite"], C_DATA)
    s2 = box(ax, 0, ROW[1], ["Sentinel-2 optical",
                             "6 bands · NDVI",
                             "NDWI · MNDWI"], C_DATA)
    tr = box(ax, 0, ROW[2], ["Terrain & static",
                             "DEM · HAND · TWI · SPI · TPI",
                             "curvature · land cover · rainfall"], C_DATA)
    ev = box(ax, 0, LANE,  ["Sentinel-1 event scenes",
                            "pre-monsoon vs flood window",
                            "2022 & 2025"], C_DATA)

    # -- 2. dataset ---------------------------------------------------------
    cube = box(ax, 1, ROW[0], ["36-band feature cube",
                               "30 m grid · EPSG:32643",
                               "z-score · sin/cos · one-hot"], C_PROC, h=HT)
    dset = box(ax, 1, ROW[2], ["Training dataset",
                               "floodable domain · 10k + 10k",
                               "6 km spatial block split"], C_PROC, h=HT)
    inv  = box(ax, 1, LANE,   ["Flood inventory",
                               "per-orbit change detection",
                               "slope · HAND · water filters"], C_LABEL, h=HT)

    # -- 3. models + evaluation --------------------------------------------
    ens = box(ax, 2, ROW[0], ["Ensembles",
                              "RF · XGBoost · LightGBM"], C_MODEL)
    cnn = box(ax, 2, ROW[1], ["Patch CNN",
                              "32×32 spatial context"], C_MODEL)
    seg = box(ax, 2, ROW[2], ["Transformers",
                              "SegFormer · multi-branch",
                              "fusion (SAR / optical / terrain)"], C_MODEL)
    evl = box(ax, 2, LANE,   ["Evaluation",
                              "repeated spatial CV · DeLong",
                              "temporal transfer, both ways"], C_LABEL, h=HT)

    # -- 4. explainability + products --------------------------------------
    ts  = box(ax, 3, ROW[0], ["TreeSHAP",
                              "exact factor attribution"], C_XAI)
    dl  = box(ax, 3, ROW[1], ["Deep-model attribution",
                              "IG · Grad-CAM · permutation",
                              "branch ablation"], C_XAI)
    con = box(ax, 3, ROW[2], ["Consistency analysis",
                              "cross-method rank ρ",
                              "hydrological audit"], C_XAI)
    out = box(ax, 3, LANE,   ["Products",
                              "masked susceptibility maps",
                              "factor & sensor attribution"], C_OUT, h=HT)

    # -- connectors ---------------------------------------------------------
    r1 = X[1] - GAP / 2                       # rail between columns 1 and 2
    r2 = X[2] - GAP / 2                       # between 2 and 3
    r3 = X[3] - GAP / 2                       # between 3 and 4

    merge(ax, [s1, s2, tr], cube, r1)         # three sources -> feature cube
    hlink(ax, ev, inv)                        # event scenes -> inventory
    vlink(ax, cube, dset)                     # cube -> training dataset
    vlink(ax, inv, dset)                      # inventory -> training dataset

    split(ax, dset, [ens, cnn, seg], r2)      # dataset -> three model families

    # every model family is evaluated: the three right-edge stubs share a rail
    # at r3, which branches right into stage 4 and down into Evaluation.
    y_join = 2.58
    for m in (ens, cnn, seg):
        ax.plot([m["r"], r3], [m["cy"], m["cy"]],
                color=ARROW, lw=1.15, zorder=2, solid_capstyle="round")
    ax.plot([r3, r3], [y_join, ens["cy"]],
            color=ARROW, lw=1.15, zorder=2, solid_capstyle="round")
    path(ax, [(r3, y_join), (evl["cx"], y_join), (evl["cx"], evl["t"])])

    hlink(ax, ens, ts)                        # ensembles -> TreeSHAP
    hlink(ax, cnn, dl)                        # CNN -> deep attribution
    path(ax, [(seg["r"], seg["cy"]), (r3, seg["cy"]),
              (r3, dl["cy"] - 0.18), (dl["l"], dl["cy"] - 0.18)])

    # TreeSHAP joins the consistency analysis via a right-hand rail
    far = X[3] + W + 0.26
    path(ax, [(ts["r"], ts["cy"] - 0.22), (far, ts["cy"] - 0.22),
              (far, con["cy"]), (con["r"], con["cy"])])
    vlink(ax, dl, con)                        # deep attribution -> consistency
    vlink(ax, con, out)                       # consistency -> products
    hlink(ax, evl, out)                       # evaluation -> products

    # -- leakage note, set in the gap between the two Sentinel-1 boxes ------
    ny = (tr["b"] + ev["t"]) / 2
    ax.add_patch(FancyBboxPatch((X[0] + 0.10, ny - 0.42), W - 0.20, 0.84,
                                boxstyle="round,pad=0.015", fc="#FFFFFF",
                                ec="#B9C2CD", lw=0.9, ls=(0, (3, 2)),
                                zorder=3))
    ax.text(X[0] + W / 2, ny,
            "Sentinel-1 supplies both predictors\n"
            "and labels, but from disjoint time\n"
            "windows — the inventory cannot\nleak into the feature cube",
            ha="center", va="center", fontsize=FS - 1.5, color="#4A5462",
            style="italic", linespacing=1.5, zorder=4)

    save_fig(fig, "fig4_flowchart")


if __name__ == "__main__":
    main()

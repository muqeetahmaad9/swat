"""
Figures for the revision (JSTARS), drawn from the revision experiments:

  fig_cv_splits        - per-split AUC for all four models (36 bands), the
                         5-split mean, and the original single split
  fig_source_ablation  - AUC lost when each data source is removed and the
                         model retrained (runs/cv_all_models_no*)
  fig_xai_agreement    - rank agreement between explanation methods within
                         and across models, plus source shares per method

Style follows xai_analysis.py (Okabe-Ito colours, viridis, recessive grid,
PNG + vector PDF).
"""

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
FIG = os.path.join(HERE, "..", "figures")

INK, GRID = "#111111", "#dcdcdc"
MODEL_COLORS = {"RF": "#0072B2", "XGBoost": "#D55E00", "LightGBM": "#009E73",
                "CNN": "#CC79A7"}
SOURCE_COLORS = {"SAR": "#0072B2", "OPT": "#E69F00", "TER": "#009E73"}
SOURCE_LABEL = {"SAR": "SAR (2 bands)", "OPT": "Optical (9 bands)",
                "TER": "Terrain and static (25 bands)"}
MODELS = ["RF", "XGBoost", "LightGBM", "CNN"]
# single-split ROC-AUC on the original split (paper Table 2)
SINGLE = {"RF": 0.9316, "XGBoost": 0.9510, "LightGBM": 0.9527, "CNN": 0.9567}

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load(name):
    with open(os.path.join(RUNS, name, "results.json")) as f:
        return json.load(f)


def save_fig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.png"))
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)
    print(f"  figure saved: {name}.png / .pdf")


def fig_cv_splits():
    cv = load("cv_all_models")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for i, m in enumerate(MODELS):
        vals = cv[m]["per_split"]
        x = np.full(len(vals), i) + (rng.random(len(vals)) - 0.5) * 0.17
        ax.scatter(x, vals, s=30, color=MODEL_COLORS[m], alpha=0.85,
                   edgecolors="none", zorder=3)
        ax.plot([i - 0.22, i + 0.22], [cv[m]["mean"]] * 2, color=INK, lw=1.6, zorder=4)
        ax.plot(i, SINGLE[m], marker="D", ms=6.5, mfc="none", mec=INK, mew=1.4, zorder=5)
        ax.annotate("", xy=(i + 0.30, SINGLE[m]), xytext=(i + 0.30, cv[m]["mean"]),
                    arrowprops=dict(arrowstyle="<->", color="#777777", lw=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=7), zorder=4)
        ax.text(i + 0.34, (SINGLE[m] + cv[m]["mean"]) / 2,
                f"+{SINGLE[m] - cv[m]['mean']:.3f}", ha="left", va="center",
                fontsize=7, color="#555555")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_xlim(-0.45, len(MODELS) - 0.30)
    ax.set_ylim(0.885, 0.965)
    ax.set_ylabel("Test AUC")
    ax.grid(axis="x", visible=False)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=6, color="#777777",
               label="one spatial split"),
        Line2D([], [], ls="-", lw=1.6, color=INK, label="mean of 5 splits"),
        Line2D([], [], marker="D", ls="none", ms=6.5, mfc="none", mec=INK,
               mew=1.4, label="original single split"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, fontsize=7.6, handlelength=1.4, columnspacing=1.3,
              handletextpad=0.5)
    save_fig(fig, "fig_cv_splits")


def fig_source_ablation():
    full = load("cv_all_models")
    drop = {s: load(f"cv_all_models_no{s}") for s in ["SAR", "OPT", "TER"]}
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    w = 0.25
    for j, s in enumerate(["SAR", "OPT", "TER"]):
        means, sds = [], []
        for m in MODELS:
            d = np.array(full[m]["per_split"]) - np.array(drop[s][m]["per_split"])
            means.append(d.mean()); sds.append(d.std())
        x = np.arange(len(MODELS)) + (j - 1) * w
        ax.bar(x, means, w * 0.92, yerr=sds, color=SOURCE_COLORS[s],
               error_kw=dict(lw=0.8, capsize=2, ecolor="#444444"),
               label=SOURCE_LABEL[s], zorder=3)
        for xi, v, sd in zip(x, means, sds):
            ax.text(xi, v + sd + 0.002, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.4, color="#333333")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("AUC lost when source is removed\n(model retrained)")
    ax.set_ylim(0, 0.108)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
              fontsize=7.6, handlelength=1.2, columnspacing=1.2)
    save_fig(fig, "fig_source_ablation")


def fig_xai_agreement():
    r = load("xai_agreement")
    methods = ["XGB_TreeSHAP", "XGB_Perm", "CNN_IG", "CNN_GradSHAP", "CNN_Perm"]
    labels = ["XGBoost\nTreeSHAP", "XGBoost\npermutation", "CNN\nIG",
              "CNN\ngradient SHAP", "CNN\npermutation"]
    n = len(methods)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = r["pairs"][f"{methods[i]}|{methods[j]}"]["mean"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.9),
                                 gridspec_kw=dict(width_ratios=[1.05, 1]))
    im = ax.imshow(M, cmap="viridis", vmin=0.3, vmax=1.0)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.7 else INK)
    for (a, b) in [(0, 2), (2, 5)]:           # outline the within-model blocks
        ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False,
                               ec="white", lw=2.2))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7.0, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.grid(False)
    ax.set_title("(a) Rank agreement (mean Spearman $\\rho$, 5 splits)", loc="left",
                 fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=7.5)

    share = r["source_share_mean"]
    y = np.arange(n)
    left = np.zeros(n)
    for s in ["TER", "OPT", "SAR"]:
        v = np.array([share[m][s] for m in methods])
        bx.barh(y, v, left=left, color=SOURCE_COLORS[s], height=0.62,
                label=SOURCE_LABEL[s], zorder=3)
        for yi, li, vi in zip(y, left, v):
            if vi > 0.07:
                bx.text(li + vi / 2, yi, f"{100 * vi:.0f}%", ha="center", va="center",
                        fontsize=7.2, color="white" if s != "OPT" else INK)
        left += v
    bx.set_yticks(y); bx.set_yticklabels(labels, fontsize=7.4)
    bx.invert_yaxis()
    bx.set_xlim(0, 1); bx.set_xticks([0, 0.25, 0.5, 0.75, 1])
    bx.set_xticklabels(["0", "25", "50", "75", "100"])
    bx.set_xlabel("Share of total attribution (%)", fontsize=8.5)
    bx.grid(axis="y", visible=False)
    bx.set_title("(b) Share of attribution by data source", loc="left", fontsize=9.5)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=7.2,
              handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    save_fig(fig, "fig_xai_agreement")


if __name__ == "__main__":
    fig_cv_splits()
    fig_source_ablation()
    fig_xai_agreement()

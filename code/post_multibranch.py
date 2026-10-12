"""
Post-processing for the multi-branch Colab run.

Reads runs/multibranch/history.json and produces:
  - console summary: test metrics + sensor-level attribution numbers
    (the exact content for manuscript Section 4.8)
  - fig_branch_attribution : AUC drop when each source is ablated
  - fig_training_curves_multibranch

Run AFTER placing best_model.pt + history.json in runs/multibranch/:
    python post_multibranch.py
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from xai_analysis import save_fig, INK, MUTED, BAR_HUE

OUT = "../runs/multibranch"

BRANCH_COLORS = {"SAR": "#0072B2", "OPTICAL": "#E69F00", "TERRAIN": "#009E73"}


def main():
    path = f"{OUT}/history.json"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — download the Colab results first.")
    h = json.load(open(path))
    test, abl, groups = h["test"], h["ablation"], h["groups"]

    # ---- console summary (Section 4.8 raw numbers) --------------------------
    print("=" * 60)
    print("MULTI-BRANCH FUSION — test metrics (dense, spatial blocks)")
    for k, v in test.items():
        print(f"  {k:10s} {v:.4f}")
    full = abl["full"]
    print(f"\nSENSOR-LEVEL ATTRIBUTION (branch ablation, full AUC {full:.4f})")
    drops = {g: full - abl[g] for g in groups}
    for g, d in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"  {g:8s} ({groups[g]:2d} bands): AUC without it {abl[g]:.4f} "
              f"-> drop {d:+.4f}")
    share = {g: d / max(sum(max(v, 0) for v in drops.values()), 1e-9)
             for g, d in drops.items()}
    print("relative shares (of total positive drop):",
          {g: f"{100*s:.0f}%" for g, s in share.items()})
    print("=" * 60)

    # ---- figure: branch ablation bars ---------------------------------------
    # Each bar carries BOTH the absolute AUC loss and that source's share of
    # the total deficit: the share is the quantity discussed in the text and
    # was previously computed for the console only, so the figure could not be
    # read without the table beside it. No "+" sign on the value, since the
    # axis already states these are losses.
    order = sorted(drops, key=lambda g: -drops[g])
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.bar(range(len(order)), [drops[g] for g in order],
           color=[BRANCH_COLORS[g] for g in order], width=0.55,
           edgecolor="none", zorder=3)
    # SAR is an acronym, so .capitalize() would render it "Sar"
    NICE = {"SAR": "SAR", "OPTICAL": "Optical", "TERRAIN": "Terrain"}
    top = max(drops.values())
    for i, g in enumerate(order):
        ax.text(i, drops[g] + top * 0.075, f"{drops[g]:.3f}", ha="center",
                va="bottom", fontsize=9, color=INK, fontweight="bold",
                zorder=4)
        ax.text(i, drops[g] + top * 0.018, f"{100 * share[g]:.0f}% of deficit",
                ha="center", va="bottom", fontsize=7.4, color="#555555",
                zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{NICE.get(g, g)}\n({groups[g]} bands)" for g in order],
                       fontsize=8.5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel("test-AUC lost when the source is ablated")
    ax.set_title(f"Sensor-level attribution by branch ablation\n"
                 f"(full-model test AUC {full:.4f})", loc="left", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    save_fig(fig, "fig_branch_attribution")

    # ---- figure: training curves (skipped if the epoch log is empty, e.g.
    # when evaluating a rescued checkpoint from an interrupted run) -----------
    hist = h["history"]
    if not hist:
        print("no epoch history in file -> training-curve figure skipped")
        return
    ep = [r["epoch"] for r in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3))
    a1.plot(ep, [r["train_loss"] for r in hist], color=BAR_HUE, lw=1.6)
    a1.set_xlabel("epoch"); a1.set_ylabel("training loss")
    a1.set_title("Multi-branch training loss", loc="left")
    a2.plot(ep, [r["val_auc"] for r in hist], color=BAR_HUE, lw=1.6)
    b = h["best_epoch"]
    a2.scatter([b], [hist[b]["val_auc"]], color="#D55E00", zorder=5,
               label=f"best (E{b}, AUC {hist[b]['val_auc']:.3f})")
    a2.set_xlabel("epoch"); a2.set_ylabel("validation AUC")
    a2.set_title("Validation AUC", loc="left"); a2.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig_training_curves_multibranch")


if __name__ == "__main__":
    main()

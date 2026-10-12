"""
Tabular baseline comparison for the paper's Results section.

- Models : Random Forest, XGBoost, LightGBM
- Rigor  : 5 seeds per model (mean ± SD of every metric), DeLong test on AUC
           differences between all model pairs (single representative seed)
- Metrics: Accuracy, Precision, Recall, F1, IoU, ROC-AUC, PR-AUC
- Extras : ROC + PR curve figure; basin-wide susceptibility map from the best
           model (GeoTIFF + publication PNG)

Inputs : data/processed/X_*_pixels.npy, y_*_pixels.npy, feature_cube.npy,
         valid_mask.npy, metadata.json
Outputs: runs/baselines/results.json, results_table.csv
         figures/fig_roc_pr_curves.(png|pdf)
         figures/fig_susceptibility_map.(png|pdf)
         data/processed/susceptibility_xgb.tif
"""

import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths & shared style (consistent with xai_analysis.py)                       #
# --------------------------------------------------------------------------- #
P = "../data/processed"
OUT = "../runs/baselines"
FIG = "../figures"
SEEDS = [0, 1, 2, 3, 4]

INK, MUTED, GRID = "#1a1a2e", "#6b7280", "#e5e7eb"
# Okabe-Ito, fixed order: RF -> blue, XGBoost -> vermillion, LightGBM -> green
MODEL_COLORS = {"RF": "#0072B2", "XGBoost": "#D55E00", "LightGBM": "#009E73"}
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save_fig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.png"))
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)
    print(f"  figure saved: {name}")


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def all_metrics(y, p, thr=0.5):
    """Full metric set at a fixed threshold + threshold-free AUCs."""
    from sklearn.metrics import (roc_auc_score, average_precision_score)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {
        "accuracy":  (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": prec,
        "recall":    rec,
        "f1":        2 * prec * rec / max(prec + rec, 1e-9),
        "iou":       tp / max(tp + fp + fn, 1),
        "roc_auc":   roc_auc_score(y, p),
        "pr_auc":    average_precision_score(y, p),
    }


def delong_test(y, p1, p2):
    """Fast DeLong test for the difference between two correlated AUCs.
    Returns (auc1, auc2, z, two-sided p). Implementation: Sun & Xu (2014)."""
    y = np.asarray(y).astype(bool)
    pos1, neg1 = p1[y], p1[~y]
    pos2, neg2 = p2[y], p2[~y]
    m, n = len(pos1), len(neg1)

    def structural(pos, neg):
        # midrank-based V10 / V01 components
        all_ = np.concatenate([pos, neg])
        order = np.argsort(all_)
        ranks = np.empty(len(all_)); ranks[order] = np.arange(1, len(all_) + 1)
        # handle ties with midranks
        sorted_ = all_[order]
        i = 0
        while i < len(sorted_):
            j = i
            while j < len(sorted_) - 1 and sorted_[j + 1] == sorted_[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        r_pos, r_neg = ranks[:m], ranks[m:]
        auc = (r_pos.sum() - m * (m + 1) / 2) / (m * n)
        v10 = (r_pos - np.arange(1, m + 1)) / n            # per-positive components
        v01 = 1 - (r_neg - np.arange(1, n + 1)) / m        # per-negative components
        return auc, v10, v01

    auc1, v10_1, v01_1 = structural(pos1, neg1)
    auc2, v10_2, v01_2 = structural(pos2, neg2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))                # positive-side covariance
    s01 = np.cov(np.vstack([v01_1, v01_2]))                # negative-side covariance
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m \
        + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (auc1 - auc2) / max(np.sqrt(var), 1e-12)
    from math import erf
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
    return auc1, auc2, z, pval


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #

def make_model(name, seed):
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                      n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="auc", random_state=seed, n_jobs=-1)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                  subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, min_child_samples=20,
                                  random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(OUT, exist_ok=True)
    Xtr = np.load(f"{P}/X_train_pixels.npy"); ytr = np.load(f"{P}/y_train_pixels.npy")
    Xva = np.load(f"{P}/X_val_pixels.npy");   yva = np.load(f"{P}/y_val_pixels.npy")
    Xte = np.load(f"{P}/X_test_pixels.npy");  yte = np.load(f"{P}/y_test_pixels.npy")
    # train on train+val for the tabular models (no epoch-level early stopping
    # to tune here); test block stays untouched.
    Xfit = np.vstack([Xtr, Xva]); yfit = np.concatenate([ytr, yva])
    print(f"fit: {Xfit.shape}  test: {Xte.shape}  "
          f"(test prevalence {yte.mean():.2f})")

    results, test_probs = {}, {}
    for name in ["RF", "XGBoost", "LightGBM"]:
        per_seed = []
        for seed in SEEDS:
            model = make_model(name, seed)
            model.fit(Xfit, yfit)
            p = model.predict_proba(Xte)[:, 1]
            per_seed.append(all_metrics(yte, p))
            if seed == SEEDS[0]:                       # representative run for
                test_probs[name] = p                   # curves + DeLong + map
                if name == "XGBoost":
                    import pickle
                    with open(f"{OUT}/xgb_seed0.pkl", "wb") as f:
                        pickle.dump(model, f)
        agg = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
                   "std":  float(np.std([m[k] for m in per_seed]))}
               for k in per_seed[0]}
        results[name] = agg
        print(f"{name}: AUC {agg['roc_auc']['mean']:.4f}±{agg['roc_auc']['std']:.4f} "
              f"F1 {agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f}")

    # ---- DeLong pairwise significance (seed-0 probabilities) ----------------
    pairs, names = {}, list(test_probs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            auc1, auc2, z, pv = delong_test(yte, test_probs[a], test_probs[b])
            pairs[f"{a}_vs_{b}"] = {"auc1": auc1, "auc2": auc2,
                                    "z": z, "p_value": pv}
            print(f"DeLong {a} ({auc1:.4f}) vs {b} ({auc2:.4f}): "
                  f"z={z:.2f}, p={pv:.4f}")

    # ---- ROC + PR curves figure ---------------------------------------------
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for name, p in test_probs.items():
        c = MODEL_COLORS[name]
        fpr, tpr, _ = roc_curve(yte, p)
        pr, rc, _ = precision_recall_curve(yte, p)
        auc = results[name]["roc_auc"]["mean"]
        ap = results[name]["pr_auc"]["mean"]
        ax1.plot(fpr, tpr, color=c, lw=1.6, label=f"{name} (AUC {auc:.3f})")
        ax2.plot(rc, pr, color=c, lw=1.6, label=f"{name} (AP {ap:.3f})")
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC — spatial-block test set", loc="left"); ax1.legend(loc="lower right")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_ylim(0.4, 1.02)
    ax2.set_title("Precision–Recall", loc="left"); ax2.legend(loc="lower left")
    fig.tight_layout()
    save_fig(fig, "fig_roc_pr_curves")

    # ---- basin-wide susceptibility map (XGBoost seed 0) ---------------------
    import pickle
    with open(f"{OUT}/xgb_seed0.pkl", "rb") as f:
        best = pickle.load(f)
    cube = np.load(f"{P}/feature_cube.npy", mmap_mode="r")   # avoid RAM copy
    valid = np.load(f"{P}/valid_mask.npy")
    H, W, C = cube.shape
    prob = np.full((H, W), np.nan, "float32")
    rows_idx, cols_idx = np.where(valid)                     # valid pixels only
    CH = 2_000_000                                           # prediction chunks
    for s in range(0, len(rows_idx), CH):
        r, c = rows_idx[s:s + CH], cols_idx[s:s + CH]
        prob[r, c] = best.predict_proba(np.asarray(cube[r, c, :]))[:, 1]
        print(f"  map: {min(s + CH, len(rows_idx)):,}/{len(rows_idx):,} px")

    # GeoTIFF with georeferencing from the pipeline metadata
    import rasterio
    from rasterio.transform import Affine
    with open(f"{P}/metadata.json") as f:
        meta = json.load(f)
    t = meta["transform"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    with rasterio.open(
            f"{P}/susceptibility_xgb.tif", "w", driver="GTiff",
            height=H, width=W, count=1, dtype="float32",
            crs=meta["crs"], transform=transform, nodata=np.nan,
            compress="deflate") as dst:
        dst.write(prob, 1)
    print("GeoTIFF saved: susceptibility_xgb.tif")

    # publication PNG (downsampled 4x for rendering speed)
    fig, ax = plt.subplots(figsize=(6.2, 7))
    im = ax.imshow(prob[::4, ::4], cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("Flood susceptibility — XGBoost (probability)", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("P(flood)", color=MUTED)
    save_fig(fig, "fig_susceptibility_map")

    # ---- persist everything --------------------------------------------------
    with open(f"{OUT}/results.json", "w") as f:
        json.dump({"per_model": results, "delong": pairs,
                   "seeds": SEEDS, "n_test": int(len(yte))}, f, indent=2)
    with open(f"{OUT}/results_table.csv", "w") as f:
        cols = ["accuracy", "precision", "recall", "f1", "iou", "roc_auc", "pr_auc"]
        f.write("model," + ",".join(cols) + "\n")
        for name, agg in results.items():
            f.write(name + "," + ",".join(
                f"{agg[k]['mean']:.4f}±{agg[k]['std']:.4f}" for k in cols) + "\n")
    print("Results written to runs/baselines/")


if __name__ == "__main__":
    main()

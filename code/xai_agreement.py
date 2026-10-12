"""
Revision experiment R3: explanation agreement WITHIN a model, ACROSS models,
and ACROSS spatial splits.

The submitted paper compared TreeSHAP on XGBoost with Integrated Gradients /
permutation importance on SegFormer, so its low rank agreement (rho 0.33-0.42)
confounded "different explanation method" with "different model". Here every
method is applied to the same trained model, on the same test points, over
the five repeated spatial splits of cv_all_models.py:

  XGBoost : TreeSHAP (exact), permutation importance
  CNN     : Integrated Gradients, Gradient SHAP (expected gradients),
            permutation importance

Band importance for the CNN gradient methods = mean over test patches of
|sum over the patch of the attribution for that band| (completeness-preserving
band total). Permutation importance = mean AUC drop over R repeats when a band
is shuffled across test samples (for the CNN the whole band patch is swapped).

Usage:  python xai_agreement.py
Output: ../runs/xai_agreement/results.json, importances_split{k}.npz
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.argv = sys.argv[:1]                      # cv_all_models parses argv on import
import cv_all_models as cv                   # same data, sampling, splits, CNN

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "xai_agreement")
os.makedirs(OUT, exist_ok=True)
R_PERM = 3
IG_STEPS = 32
GS_SAMPLES = 32
METHODS = ["XGB_TreeSHAP", "XGB_Perm", "CNN_IG", "CNN_GradSHAP", "CNN_Perm"]
MODEL_OF = {m: m.split("_")[0] for m in METHODS}


def xgb_model(Xf, yf, seed=0):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="auc", random_state=seed, n_jobs=-1)
    return m.fit(Xf, yf)


def perm_importance(score_fn, X, y, rng):
    """AUC drop when band j is shuffled across samples (R_PERM repeats)."""
    base = roc_auc_score(y, score_fn(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(R_PERM):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]   # (N,C) or (N,C,H,W): whole band
            drops.append(base - roc_auc_score(y, score_fn(Xp)))
        imp[j] = np.mean(drops)
    return imp, base


def logits_grad(model, x):
    x = x.detach().requires_grad_(True)
    out = model(x)
    g, = torch.autograd.grad(out.sum(), x)
    return g


def integrated_gradients(model, X, batch=256):
    """Baseline = 0 (feature mean in standardized units); Riemann midpoint."""
    model.eval()
    tot = np.zeros(X.shape[1])
    for s in range(0, len(X), batch):
        x = torch.from_numpy(X[s:s + batch].astype("float32"))
        g = torch.zeros_like(x)
        for k in range(IG_STEPS):
            g += logits_grad(model, x * ((k + 0.5) / IG_STEPS))
        attr = (x * g / IG_STEPS).sum(dim=(2, 3))      # band totals per sample
        tot += attr.abs().sum(0).numpy()
    return tot / len(X)


def gradient_shap(model, X, Xbg, rng, batch=256):
    """Expected gradients: baselines drawn from training patches."""
    model.eval()
    tot = np.zeros(X.shape[1])
    for s in range(0, len(X), batch):
        x = torch.from_numpy(X[s:s + batch].astype("float32"))
        acc = torch.zeros_like(x)
        for _ in range(GS_SAMPLES):
            b = torch.from_numpy(Xbg[rng.integers(0, len(Xbg), len(x))].astype("float32"))
            a = torch.from_numpy(rng.random(len(x)).astype("float32")).view(-1, 1, 1, 1)
            acc += logits_grad(model, b + a * (x - b)) * (x - b)
        attr = (acc / GS_SAMPLES).sum(dim=(2, 3))
        tot += attr.abs().sum(0).numpy()
    return tot / len(X)


def rho(a, b):
    return float(spearmanr(a, b).correlation)


def topk_overlap(a, b, k=5):
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def main():
    labels = cv.load_labels(os.path.join(cv.DATA, "labels", "flood_labels.tif"))
    names = [cv.BAND_NAMES[i] for i in cv.KEEP]
    groups = np.array([cv.group(n) for n in names])
    per_split = []
    for k in range(cv.N_SPLITS):
        t0 = time.time()
        rng = np.random.default_rng(k)
        pts, y = cv.sample(labels, rng)
        fit_b, test_b = cv.split_blocks(cv.block_of(pts), rng)
        in_test = np.isin(cv.block_of(pts), list(test_b))
        keep = cv.interior(pts)
        te = in_test & keep
        yt = y[te]
        prng = np.random.default_rng(1000 + k)
        imp = {}

        # ---- XGBoost: TreeSHAP + permutation, same model, same test points
        Xf, yf = cv.features(pts[~in_test]), y[~in_test]
        Xt = cv.features(pts[te])
        xgbm = xgb_model(Xf, yf)
        import shap
        sv = shap.TreeExplainer(xgbm).shap_values(Xt)
        imp["XGB_TreeSHAP"] = np.abs(sv).mean(0)
        imp["XGB_Perm"], auc_x = perm_importance(
            lambda X: xgbm.predict_proba(X)[:, 1], Xt, yt, prng)

        # ---- CNN: IG + GradientSHAP + permutation, same model, same test points
        fit_list = np.array(sorted(fit_b))
        vrng = np.random.default_rng(100 + k)
        n_val = int(round(len(fit_list) * cv.FRACS[1] / (cv.FRACS[0] + cv.FRACS[1])))
        val_b = set(vrng.permutation(fit_list)[:n_val])
        in_val = np.isin(cv.block_of(pts), list(val_b))
        tr = ~in_test & ~in_val & keep
        va = in_val & keep
        Xtr, Xva, Xte = cv.patches(pts[tr]), cv.patches(pts[va]), cv.patches(pts[te])
        model, _, _ = cv.cnn_train(0, Xtr, y[tr], Xva, y[va])
        imp["CNN_IG"] = integrated_gradients(model, Xte)
        bg = Xtr[prng.choice(len(Xtr), 512, replace=False)]
        imp["CNN_GradSHAP"] = gradient_shap(model, Xte, bg, prng)
        Xte32 = Xte.astype("float32")
        imp["CNN_Perm"], auc_c = perm_importance(
            lambda X: cv.cnn_predict(model, X), Xte32, yt, prng)
        del Xtr, Xva, Xte, Xte32

        np.savez(os.path.join(OUT, f"importances_split{k}.npz"), names=names, **imp)
        pair = {f"{a}|{b}": rho(imp[a], imp[b]) for a, b in itertools.combinations(METHODS, 2)}
        share = {m: {g: float(imp[m][groups == g].clip(min=0).sum() /
                              imp[m].clip(min=0).sum()) for g in ["SAR", "OPT", "TER"]}
                 for m in METHODS}
        per_split.append({"auc_xgb": float(auc_x), "auc_cnn": float(auc_c),
                          "rho": pair, "source_share": share})
        print(f"split {k}: AUC xgb {auc_x:.4f} cnn {auc_c:.4f} | " +
              "  ".join(f"{p.replace('|', '~')}={v:.2f}" for p, v in pair.items()) +
              f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate: within-model, cross-model, cross-split stability
    vecs = {m: [np.load(os.path.join(OUT, f"importances_split{k}.npz"))[m]
                for k in range(cv.N_SPLITS)] for m in METHODS}
    pairs = {}
    for a, b in itertools.combinations(METHODS, 2):
        r = [s["rho"][f"{a}|{b}"] for s in per_split]
        pairs[f"{a}|{b}"] = {"mean": float(np.mean(r)), "std": float(np.std(r)),
                             "per_split": r,
                             "type": "within-model" if MODEL_OF[a] == MODEL_OF[b] else "cross-model",
                             "top5_overlap_mean": float(np.mean(
                                 [topk_overlap(vecs[a][k], vecs[b][k]) for k in range(cv.N_SPLITS)]))}
    stability = {}
    for m in METHODS:
        r = [rho(vecs[m][i], vecs[m][j]) for i, j in itertools.combinations(range(cv.N_SPLITS), 2)]
        stability[m] = {"mean": float(np.mean(r)), "min": float(np.min(r)), "max": float(np.max(r))}
    mean_rank = {m: [names[i] for i in np.argsort(-np.mean(vecs[m], axis=0))[:10]] for m in METHODS}
    share_mean = {m: {g: float(np.mean([s["source_share"][m][g] for s in per_split]))
                      for g in ["SAR", "OPT", "TER"]} for m in METHODS}
    within = [v["mean"] for v in pairs.values() if v["type"] == "within-model"]
    cross = [v["mean"] for v in pairs.values() if v["type"] == "cross-model"]
    out = {"pairs": pairs, "cross_split_stability": stability, "top10_mean": mean_rank,
           "source_share_mean": share_mean,
           "summary": {"within_model_rho_mean": float(np.mean(within)),
                       "cross_model_rho_mean": float(np.mean(cross))},
           "per_split": per_split, "band_names": names}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("within-model rho", np.round(np.mean(within), 3),
          " cross-model rho", np.round(np.mean(cross), 3))
    for m in METHODS:
        print(f"{m:13s} split-stability rho {stability[m]['mean']:.2f}  top: {mean_rank[m][:5]}")


if __name__ == "__main__":
    main()

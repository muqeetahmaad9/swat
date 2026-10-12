"""
Two experiments that need no GPU:

A) REPEATED SPATIAL CV — 5 different block-permutation splits x 3 tree models.
   Decides whether model ranking is stable across spatial splits (reviewer R1.3).

B) TEMPORAL TRANSFER — train on the 2022 event, test on the 2025 event
   (and the reverse), with spatially disjoint test blocks in both directions.
   Tests whether the model learns transferable susceptibility rather than
   event-specific signatures (reviewer R2.3).

Usage: python experiments_cv_transfer.py [cv|transfer]
Outputs: runs/experiments/cv_results.json / transfer_results.json
"""

import json
import os
import sys

import numpy as np
import rasterio

P, OUT = "../data/processed", "../runs/experiments"
BLOCK, N_POS, RATIO = 200, 10000, 1.0          # match preprocess_pipeline.py
FRACS = (0.7, 0.15, 0.15)

os.makedirs(OUT, exist_ok=True)
cube = np.load(f"{P}/feature_cube.npy", mmap_mode="r")
valid = np.load(f"{P}/valid_mask.npy")
H, W, C = cube.shape
N_BCOLS = int(np.ceil(W / BLOCK))


def load_labels(path):
    with rasterio.open(path) as src:
        lab = src.read(1).astype("float32")
        if src.nodata is not None:
            lab[lab == src.nodata] = np.nan
        lab[lab == 255] = np.nan
    return lab


def sample(labels, rng, max_pos=N_POS):
    pos = np.argwhere((labels == 1) & valid)
    if len(pos) > max_pos:
        pos = pos[rng.choice(len(pos), max_pos, replace=False)]
    neg_pool = np.argwhere((labels == 0) & valid)
    neg = neg_pool[rng.choice(len(neg_pool), min(int(len(pos) * RATIO),
                                                 len(neg_pool)), replace=False)]
    pts = np.vstack([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype("int8")
    return pts, y


def block_of(pts):
    return (pts[:, 0] // BLOCK) * N_BCOLS + (pts[:, 1] // BLOCK)


def split_blocks(block_ids, rng):
    uniq = rng.permutation(np.unique(block_ids))
    n_tr = int(len(uniq) * FRACS[0]); n_va = int(len(uniq) * FRACS[1])
    return (set(uniq[:n_tr + n_va]), set(uniq[n_tr + n_va:]))   # fit(train+val), test


def features(pts):
    return np.asarray(cube[pts[:, 0], pts[:, 1], :])


def fit_predict(name, Xf, yf, Xt, seed=0):
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(200, min_samples_leaf=2, n_jobs=-1,
                                   random_state=seed)
    elif name == "XGBoost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="auc", random_state=seed, n_jobs=-1)
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
                               subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=seed,
                               n_jobs=-1, verbose=-1)
    m.fit(Xf, yf)
    return m.predict_proba(Xt)[:, 1]


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def run_cv():
    labels = load_labels("../data/labels/flood_labels.tif")
    results = {m: [] for m in ["RF", "XGBoost", "LightGBM"]}
    for split_seed in range(5):
        rng = np.random.default_rng(split_seed)
        pts, y = sample(labels, rng)
        fit_b, test_b = split_blocks(block_of(pts), rng)
        in_test = np.isin(block_of(pts), list(test_b))
        Xf, yf = features(pts[~in_test]), y[~in_test]
        Xt, yt = features(pts[in_test]), y[in_test]
        for name in results:
            a = auc(yt, fit_predict(name, Xf, yf, Xt))
            results[name].append(a)
            print(f"split {split_seed} {name}: AUC {a:.4f} "
                  f"(n_test={len(yt)}, prev={yt.mean():.2f})", flush=True)
    summary = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "per_split": v} for m, v in results.items()}
    with open(f"{OUT}/cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print({m: f"{s['mean']:.4f}±{s['std']:.4f}" for m, s in summary.items()})


def run_transfer():
    lab22 = load_labels("../data/labels/flood_labels_2022.tif")
    lab25 = load_labels("../data/labels/flood_labels_2025.tif")
    rng = np.random.default_rng(42)
    out = {}
    for tr_name, tr_lab, te_name, te_lab in [
            ("2022", lab22, "2025", lab25), ("2025", lab25, "2022", lab22)]:
        rng_d = np.random.default_rng(42)                     # same blocks both ways
        pts_tr, y_tr = sample(tr_lab, np.random.default_rng(0))
        pts_te, y_te = sample(te_lab, np.random.default_rng(1))
        fit_b, test_b = split_blocks(
            block_of(np.vstack([pts_tr, pts_te])), rng_d)
        keep_tr = np.isin(block_of(pts_tr), list(fit_b))      # train: fit blocks only
        keep_te = np.isin(block_of(pts_te), list(test_b))     # test: held-out blocks
        Xf, yf = features(pts_tr[keep_tr]), y_tr[keep_tr]
        Xt, yt = features(pts_te[keep_te]), y_te[keep_te]
        p = fit_predict("LightGBM", Xf, yf, Xt)
        # in-event reference: same model evaluated on ITS OWN event's test blocks
        keep_ref = np.isin(block_of(pts_tr), list(test_b))
        p_ref = fit_predict("LightGBM", Xf, yf, features(pts_tr[keep_ref]))
        out[f"train{tr_name}_test{te_name}"] = {
            "auc_transfer": float(auc(yt, p)),
            "auc_in_event": float(auc(y_tr[keep_ref], p_ref)),
            "n_test": int(len(yt))}
        print(f"train {tr_name} -> test {te_name}: "
              f"transfer AUC {out[f'train{tr_name}_test{te_name}']['auc_transfer']:.4f} "
              f"(in-event ref {out[f'train{tr_name}_test{te_name}']['auc_in_event']:.4f})",
              flush=True)
    with open(f"{OUT}/transfer_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    run_cv() if mode == "cv" else run_transfer()

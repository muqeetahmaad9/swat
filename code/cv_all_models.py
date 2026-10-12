"""
Revision experiment R1: repeated spatial CV for ALL models, including the
patch CNN, on identical splits.

The published repeated-CV table (experiments_cv_transfer.py) covered only the
three tree ensembles; the CNN's headline AUC (0.957) came from the single
original split. This script re-runs the SAME five block-permutation splits
(same sampling RNG, same blocks) and adds the CNN, so all four model families
are compared on identical test terrain.

Trees: fitted on all fit blocks, exactly as in experiments_cv_transfer.py
       (so their per-split AUCs should reproduce cv_results.json).
CNN:   fit blocks are further divided into train/val blocks (val = 15/85 of
       fit blocks, separate RNG) for early stopping, as in train_cnn.py.

Usage:  python cv_all_models.py [--drop SAR|OPT|TER]
        --drop removes one data source's bands (same splits otherwise); used
        for the no-Sentinel-1 check and leave-one-source-out ablation.
Output: ../runs/cv_all_models[_noSAR|_noOPT|_noTER]/results.json,
        per-split test probabilities
"""

import argparse

import json
import os
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# data/ at the repository root by default; override with SWAT_DATA
DATA = os.environ.get("SWAT_DATA",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
P = os.path.join(DATA, "processed")

# Source groups as in the multi-branch model (2 / 9 / 25 bands)
SAR_BANDS = {"swat_s1_vh", "swat_s1_vv"}
OPT_BANDS = {"swat_ndvi", "swat_ndwi", "swat_mndwi"} | {f"swat_s2_composite_b{i}" for i in range(6)}

ap = argparse.ArgumentParser()
ap.add_argument("--drop", choices=["SAR", "OPT", "TER"], default=None)
ARGS = ap.parse_args()
OUT = os.path.join(os.path.dirname(__file__), "..", "runs",
                   "cv_all_models" + (f"_no{ARGS.drop}" if ARGS.drop else ""))
BLOCK, N_POS, RATIO = 200, 10000, 1.0                     # match experiments_cv_transfer.py
FRACS = (0.7, 0.15, 0.15)
N_SPLITS = 5
CNN_SEEDS = [0, 1, 2]
PATCH = 32
BATCH, MAX_EPOCHS, PATIENCE, LR = 128, 30, 5, 1e-3        # match train_cnn.py

torch.set_num_threads(max(1, os.cpu_count() - 2))
os.makedirs(OUT, exist_ok=True)
cube = np.load(os.path.join(P, "feature_cube.npy"), mmap_mode="r")
valid = np.load(os.path.join(P, "valid_mask.npy"))
H, W, C_ALL = cube.shape
BAND_NAMES = json.load(open(os.path.join(P, "metadata.json")))["band_names"]
assert len(BAND_NAMES) == C_ALL


def group(name):
    return "SAR" if name in SAR_BANDS else "OPT" if name in OPT_BANDS else "TER"


assert [group(b) for b in BAND_NAMES].count("SAR") == 2
assert [group(b) for b in BAND_NAMES].count("OPT") == 9
KEEP = np.array([i for i, b in enumerate(BAND_NAMES) if group(b) != ARGS.drop])
C = len(KEEP)
print(f"drop={ARGS.drop}  using {C} of {C_ALL} bands", flush=True)
N_BCOLS = int(np.ceil(W / BLOCK))


# ---- sampling and splitting: copied unchanged from experiments_cv_transfer.py

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
    return np.asarray(cube[pts[:, 0], pts[:, 1], :])[:, KEEP]


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


# ---- CNN: architecture and training loop as in train_cnn.py

class PatchCNN(nn.Module):
    """3 conv blocks -> global average pool -> linear head."""

    def __init__(self, in_ch):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def interior(pts):
    half = PATCH // 2
    return ((pts[:, 0] >= half) & (pts[:, 0] < H - half) &
            (pts[:, 1] >= half) & (pts[:, 1] < W - half))


def patches(pts):
    """NCHW float16 patches centred on each point (stored half-precision to fit RAM)."""
    half = PATCH // 2
    X = np.empty((len(pts), C, PATCH, PATCH), dtype="float16")
    for i, (r, c) in enumerate(pts):
        X[i] = np.asarray(cube[r - half:r + half, c - half:c + half, :])[:, :, KEEP].transpose(2, 0, 1)
    return X


def batches(X, y, shuffle, rng=None):
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for s in range(0, len(X), BATCH):
        b = idx[s:s + BATCH]
        yield torch.from_numpy(X[b].astype("float32")), torch.from_numpy(y[b].astype("float32"))


@torch.no_grad()
def cnn_predict(model, X):
    model.eval()
    return np.concatenate([torch.sigmoid(model(x)).numpy()
                           for x, _ in batches(X, np.zeros(len(X)), False)])


def cnn_train(seed, Xtr, ytr, Xva, yva):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = PatchCNN(C)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    lossfn = nn.BCEWithLogitsLoss()
    best_auc, best_state, best_ep = -1, None, -1
    for ep in range(MAX_EPOCHS):
        model.train()
        for x, y in batches(Xtr, ytr, True, rng):
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[3])
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[2])
            opt.zero_grad(set_to_none=True)
            lossfn(model(x), y).backward(); opt.step()
        sched.step()
        auc = roc_auc_score(yva, cnn_predict(model, Xva))
        if auc > best_auc:
            best_auc, best_ep = auc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_auc, best_ep


def main():
    labels = load_labels(os.path.join(DATA, "labels", "flood_labels.tif"))
    res = {m: [] for m in ["RF", "XGBoost", "LightGBM", "CNN"]}
    cnn_seed_aucs = []
    for k in range(N_SPLITS):
        t0 = time.time()
        rng = np.random.default_rng(k)
        pts, y = sample(labels, rng)
        fit_b, test_b = split_blocks(block_of(pts), rng)
        in_test = np.isin(block_of(pts), list(test_b))

        # common test set: interior points only, so every model sees the same points
        keep = interior(pts)
        te = in_test & keep
        Xf, yf = features(pts[~in_test]), y[~in_test]
        Xt, yt = features(pts[te]), y[te]
        probs = {}
        for name in ["RF", "XGBoost", "LightGBM"]:
            probs[name] = fit_predict(name, Xf, yf, Xt)
            res[name].append(float(roc_auc_score(yt, probs[name])))

        # CNN: split fit blocks into train / val blocks
        fit_list = np.array(sorted(fit_b))
        vrng = np.random.default_rng(100 + k)
        n_val = int(round(len(fit_list) * FRACS[1] / (FRACS[0] + FRACS[1])))
        val_b = set(vrng.permutation(fit_list)[:n_val])
        in_val = np.isin(block_of(pts), list(val_b))
        tr = ~in_test & ~in_val & keep
        va = in_val & keep
        Xtr, Xva, Xte = patches(pts[tr]), patches(pts[va]), patches(pts[te])
        seed_aucs, seed_probs = [], []
        for s in CNN_SEEDS:
            model, vauc, ep = cnn_train(s, Xtr, y[tr], Xva, y[va])
            p = cnn_predict(model, Xte)
            seed_aucs.append(float(roc_auc_score(yt, p))); seed_probs.append(p)
            print(f"  split {k} CNN seed {s}: val {vauc:.4f} (ep {ep}) "
                  f"test {seed_aucs[-1]:.4f}", flush=True)
        probs["CNN"] = np.mean(seed_probs, axis=0)
        res["CNN"].append(float(np.mean(seed_aucs)))
        cnn_seed_aucs.append(seed_aucs)
        del Xtr, Xva, Xte

        np.savez(os.path.join(OUT, f"split{k}_test.npz"), y=yt, pts=pts[te], **probs)
        print(f"split {k}: " + "  ".join(f"{m} {res[m][-1]:.4f}" for m in res) +
              f"  (n_test={len(yt)}, prev={yt.mean():.2f}, {time.time()-t0:.0f}s)",
              flush=True)

    summary = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "min": float(np.min(v)), "max": float(np.max(v)),
                   "per_split": v} for m, v in res.items()}
    summary["CNN"]["per_split_per_seed"] = cnn_seed_aucs
    summary["CNN"]["mean_seed_sd_within_split"] = float(np.mean([np.std(s) for s in cnn_seed_aucs]))
    for m in ["RF", "XGBoost", "LightGBM"]:
        d = np.array(res["CNN"]) - np.array(res[m])
        summary[f"CNN_minus_{m}"] = {"per_split": d.tolist(), "mean": float(d.mean()),
                                     "n_splits_CNN_better": int((d > 0).sum())}
    summary["dropped_source"] = ARGS.drop
    summary["bands_used"] = [BAND_NAMES[i] for i in KEEP]
    summary["notes"] = ("Same 5 split seeds as experiments_cv_transfer.py; test set "
                        "restricted to points >=16 px from raster edge so all models "
                        "share identical test points. CNN AUC per split = mean over seeds.")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print({m: f"{summary[m]['mean']:.4f}±{summary[m]['std']:.4f}" for m in res})


if __name__ == "__main__":
    main()

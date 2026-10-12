"""Ablation: 16-band original feature set vs 36-band enriched set,
same samples, same model, DeLong significance on the paired test predictions."""
import json
import numpy as np
import xgboost as xgb
from train_baselines import delong_test

P = "../data/processed"
names = json.load(open(f"{P}/metadata.json"))["band_names"]

# the original 16 bands, by name, within the 36-band arrays
orig = ["swat_aspect_sin", "swat_aspect_cos", "swat_dem", "swat_mndwi",
        "swat_ndvi", "swat_ndwi", "swat_rainfall", "swat_s1_vh", "swat_s1_vv",
        "swat_s2_composite_b0", "swat_s2_composite_b1", "swat_s2_composite_b2",
        "swat_s2_composite_b3", "swat_s2_composite_b4", "swat_s2_composite_b5",
        "swat_slope"]
idx16 = [names.index(n) for n in orig]

Xtr = np.load(f"{P}/X_train_pixels.npy"); ytr = np.load(f"{P}/y_train_pixels.npy")
Xva = np.load(f"{P}/X_val_pixels.npy");   yva = np.load(f"{P}/y_val_pixels.npy")
Xte = np.load(f"{P}/X_test_pixels.npy");  yte = np.load(f"{P}/y_test_pixels.npy")
Xf, yf = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])

def fit(cols):
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="auc", random_state=42, n_jobs=-1)
    m.fit(Xf[:, cols], yf)
    return m.predict_proba(Xte[:, cols])[:, 1]

p16 = fit(idx16)
p36 = fit(list(range(len(names))))
a16, a36, z, pv = delong_test(yte, p36, p16)
print(f"XGBoost 36-band AUC {a16:.4f} vs 16-band AUC {a36:.4f}  "
      f"(DeLong z={z:.2f}, p={pv:.4f})")

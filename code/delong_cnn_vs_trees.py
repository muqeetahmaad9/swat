"""DeLong significance test: patch-CNN vs the boosted-tree baselines (seed 0)."""
import numpy as np
from train_baselines import delong_test, make_model

P = "../data/processed"
Xtr = np.load(f"{P}/X_train_pixels.npy"); ytr = np.load(f"{P}/y_train_pixels.npy")
Xva = np.load(f"{P}/X_val_pixels.npy");   yva = np.load(f"{P}/y_val_pixels.npy")
Xte = np.load(f"{P}/X_test_pixels.npy");  yte = np.load(f"{P}/y_test_pixels.npy")
Xfit = np.vstack([Xtr, Xva]); yfit = np.concatenate([ytr, yva])

# the patch test set equals the pixel test set here (3448 == 3448, no edge drops)
p_cnn = np.load("../runs/cnn/test_probs_seed0.npy")
assert len(p_cnn) == len(yte), "patch/pixel test size mismatch"

for name in ["LightGBM", "XGBoost"]:
    m = make_model(name, 0)
    m.fit(Xfit, yfit)
    p = m.predict_proba(Xte)[:, 1]
    a1, a2, z, pv = delong_test(yte, p_cnn, p)
    print(f"DeLong CNN ({a1:.4f}) vs {name} ({a2:.4f}): z={z:.2f}, p={pv:.5f}")

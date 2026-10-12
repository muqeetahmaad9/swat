"""Full 36-feature SHAP ranking (the top-15 figure cut off the new terrain bands)."""
import json
import numpy as np
import shap
import xgboost as xgb

P = "../data/processed"
names = json.load(open(f"{P}/metadata.json"))["band_names"]
Xtr = np.load(f"{P}/X_train_pixels.npy"); ytr = np.load(f"{P}/y_train_pixels.npy")
Xva = np.load(f"{P}/X_val_pixels.npy");   yva = np.load(f"{P}/y_val_pixels.npy")
Xte = np.load(f"{P}/X_test_pixels.npy")

model = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="auc", random_state=42, n_jobs=-1)
model.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
sub = np.random.default_rng(0).choice(len(Xte), min(3000, len(Xte)), False)
sv = shap.TreeExplainer(model).shap_values(Xte[sub])
imp = np.abs(sv).mean(0)
order = np.argsort(imp)[::-1]
for rank, i in enumerate(order, 1):
    print(f"{rank:2d}. {names[i]:26s} {imp[i]:.4f}")
np.savetxt("../figures/importance_shap_xgb_full36.csv",
           np.column_stack([np.array(names)[order], imp[order]]),
           fmt="%s", delimiter=",", header="feature,mean_abs_shap")

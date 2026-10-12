"""
Patch-based CNN for flood susceptibility — CPU-feasible deep baseline.

Architecture: compact 3-block convolutional classifier over 32x32x16 patches
(the Maddah-style patch-CNN class from the model comparison), predicting the
label of the patch's centre pixel. Trained with the same rigour protocol as
the tree baselines: 3 seeds, full metric set, results JSON.

Runs in ~5-8 minutes on CPU. Outputs:
    runs/cnn/results.json, cnn_seed0.pt
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

P, OUT = "../data/processed", "../runs/cnn"
SEEDS = [0, 1, 2]
BATCH, MAX_EPOCHS, PATIENCE, LR = 128, 30, 5, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def all_metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "iou": tp / max(tp + fp + fn, 1),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p)}


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


def loader(split, shuffle):
    X = np.load(f"{P}/X_{split}_patches.npy")          # (N, 32, 32, C)
    y = np.load(f"{P}/y_{split}.npy")                  # aligned with patches
    X = torch.from_numpy(X.transpose(0, 3, 1, 2).copy())   # NHWC -> NCHW
    y = torch.from_numpy(y.astype("float32"))
    return DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    ps, ys = [], []
    for x, y in dl:
        ps.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def train_one(seed, train_dl, val_dl):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PatchCNN(next(iter(train_dl))[0].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    lossfn = nn.BCEWithLogitsLoss()
    best_auc, best_state, best_ep = -1, None, -1
    for ep in range(MAX_EPOCHS):
        model.train()
        for x, y in train_dl:
            # light augmentation: random flips of the whole batch
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[3])
            if torch.rand(1) < 0.5: x = torch.flip(x, dims=[2])
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward(); opt.step()
        sched.step()
        p, yv = predict(model, val_dl)
        auc = all_metrics(yv, p)["roc_auc"]
        print(f"  seed {seed} epoch {ep:02d} val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_ep = auc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_auc


def main():
    os.makedirs(OUT, exist_ok=True)
    train_dl = loader("train", True)
    val_dl = loader("val", False)
    test_dl = loader("test", False)
    print(f"device={DEVICE}  train batches={len(train_dl)}")

    per_seed = []
    for seed in SEEDS:
        model, vauc = train_one(seed, train_dl, val_dl)
        p, yt = predict(model, test_dl)
        m = all_metrics(yt, p)
        per_seed.append(m)
        print(f"seed {seed}: test AUC {m['roc_auc']:.4f} F1 {m['f1']:.4f}")
        if seed == SEEDS[0]:
            torch.save({"model": model.state_dict(), "in_ch": 16}, f"{OUT}/cnn_seed0.pt")
            np.save(f"{OUT}/test_probs_seed0.npy", p)

    agg = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
               "std": float(np.std([m[k] for m in per_seed]))} for k in per_seed[0]}
    with open(f"{OUT}/results.json", "w") as f:
        json.dump({"model": "PatchCNN(3xConv-BN-ReLU-Pool, GAP, FC)",
                   "seeds": SEEDS, "metrics": agg}, f, indent=2)
    print("CNN:", {k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items()})


if __name__ == "__main__":
    main()

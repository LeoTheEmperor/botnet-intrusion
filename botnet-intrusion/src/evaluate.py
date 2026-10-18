"""
Final model training + evaluation on the held-out TEST set. All metrics from scratch
(no sklearn.metrics): confusion matrix, precision/recall/F1, one-vs-rest ROC + AUC.
Final model: Random Forest, max_depth=15, n_estimators=30, max_features="sqrt".
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from random_forest import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
RESULTS.mkdir(exist_ok=True)
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x compatibility


def load(split):
    return (np.load(DATA / f"X_{split}.npy"),
            pd.read_csv(DATA / f"y_{split}.csv")["label"].to_numpy())


def confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm                                   # rows = true, cols = predicted


def precision_recall_f1(cm, classes):
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"class": c, "precision": p, "recall": r, "f1": f1, "support": int(cm[i].sum())})
    return pd.DataFrame(rows)


def roc_curve_ovr(y_bin, scores, n_thresholds=100):
    thr = np.linspace(1.0, 0.0, n_thresholds)
    P = y_bin.sum(); N = len(y_bin) - P
    tpr = np.array([np.sum((scores >= t) & (y_bin == 1)) / P if P else 0.0 for t in thr])
    fpr = np.array([np.sum((scores >= t) & (y_bin == 0)) / N if N else 0.0 for t in thr])
    return fpr, tpr


def auc(fpr, tpr):
    o = np.argsort(fpr)
    return float(_trapz(tpr[o], fpr[o]))


def main():
    X_tr, y_tr = load("train"); X_te, y_te = load("test")
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=30, max_depth=15, min_samples_split=10,
                                   min_samples_leaf=5, max_features="sqrt", random_state=42)
    model.fit(X_tr, y_tr)
    print(f"Training time: {time.time() - t0:.1f}s")

    classes = model.classes_
    y_pred, proba = model.predict(X_te), model.predict_proba(X_te)
    print(f"\nTEST ACCURACY: {(y_pred == y_te).mean():.4f}")

    cm = confusion_matrix(y_te, y_pred, classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(RESULTS / "confusion_matrix.csv")
    print("\n=== Confusion matrix ===\n", pd.DataFrame(cm, index=classes, columns=classes))

    prf = precision_recall_f1(cm, classes)
    prf.to_csv(RESULTS / "per_class_metrics.csv", index=False)
    print("\n=== Per-class metrics ===\n", prf.to_string(index=False))
    print(f"\nMacro P/R/F1: {prf.precision.mean():.4f} / {prf.recall.mean():.4f} / {prf.f1.mean():.4f}")

    rows, pts = [], []
    for i, c in enumerate(classes):
        fpr, tpr = roc_curve_ovr((y_te == c).astype(int), proba[:, i])
        rows.append({"class": c, "auc": auc(fpr, tpr)})
        pts += [{"class": c, "fpr": f, "tpr": t} for f, t in zip(fpr, tpr)]
    pd.DataFrame(rows).to_csv(RESULTS / "roc_auc_per_class.csv", index=False)
    pd.DataFrame(pts).to_csv(RESULTS / "roc_curve_points.csv", index=False)
    print("\n=== ROC AUC ===\n", pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame({"y_true": y_te, "y_pred": y_pred}).to_csv(RESULTS / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()

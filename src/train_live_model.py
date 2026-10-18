"""
Train the from-scratch Random Forest on feature CSVs recorded with `live_detector.py collect`,
evaluate with a TIME-ORDERED hold-out, then export a Pi-ready model.

Every contiguous run of one label (a whole single-label file, or one segment of a scheduled
session) is split in time: first 65% -> train, 5% gap, last 30% -> test. This avoids the
random-split leakage seen in the N-BaIoT experiments.

    python3 train_live_model.py --data data/live --out models/live_model.npz
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from random_forest import RandomForestClassifier
from model_io import export_forest


def runs_of(labels):
    """Yield (start, end) index pairs of contiguous equal labels."""
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            yield start, i
            start = i


def fit_rf(X, y, trees, depth):
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    rf = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_split=10,
                                min_samples_leaf=5, max_features="sqrt", random_state=42)
    rf.fit((X - mean) / std, y, verbose=False)
    return rf, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/live")
    ap.add_argument("--out", default="models/live_model.npz")
    ap.add_argument("--trees", type=int, default=30)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--max-rows-per-class", type=int, default=6000)
    ap.add_argument("--final", action="store_true", help="after evaluating, refit on ALL data before export")
    a = ap.parse_args()

    rng = np.random.default_rng(42)
    tr_by_class, te_X, te_y = {}, [], []
    for f in sorted(Path(a.data).glob("*.csv")):
        df = pd.read_csv(f)
        y, X = df["label"].to_numpy(), df.drop(columns="label").to_numpy(dtype=np.float64)
        n_runs = 0
        for s, e in runs_of(y):
            n_runs += 1
            n = e - s
            tr_by_class.setdefault(y[s], []).append(X[s: s + int(n * 0.65)])
            te_X.append(X[s + int(n * 0.70): e]); te_y += [y[s]] * (e - s - int(n * 0.70))
        print(f"{f.name:24s} rows={len(df):>7,}  label-runs={n_runs}")
    Xtr, ytr = [], []
    for lab, parts in sorted(tr_by_class.items()):
        arr = np.vstack(parts)
        k = min(len(arr), a.max_rows_per_class)
        Xtr.append(arr[rng.choice(len(arr), k, replace=False)]); ytr += [lab] * k
        print(f"  class {lab:<14s} train rows {k:>6,}")
    Xtr, ytr, Xte, yte = np.vstack(Xtr), np.array(ytr), np.vstack(te_X), np.array(te_y)

    rf, mean, std = fit_rf(Xtr, ytr, a.trees, a.depth)
    pred = rf.predict((Xte - mean) / std)
    print(f"\ntime-ordered hold-out accuracy: {(pred == yte).mean():.4f}  ({len(yte):,} rows)")
    for c in rf.classes_:
        m = yte == c
        prec = (yte[pred == c] == c).mean() if (pred == c).any() else 0.0
        print(f"  {c:<14s} recall {(pred[m] == c).mean() if m.any() else float('nan'):.3f}   precision {prec:.3f}   test rows {m.sum()}")

    if a.final:
        rf, mean, std = fit_rf(np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]), a.trees, a.depth)
        print("refit on all data for export")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    export_forest(rf, a.out, mean, std)
    print(f"model exported -> {a.out}")


if __name__ == "__main__":
    main()

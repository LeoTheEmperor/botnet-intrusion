"""
Overfitting / underfitting analysis. Logs TRAIN vs VALIDATION accuracy for:
  1. single tree, varying max_depth   2. forest (20 trees), varying max_depth
  3. forest (depth 10), varying n_estimators
Results -> data/sweep_*.csv (the train/val gap is what the report discusses).
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
RESULTS.mkdir(exist_ok=True)
X_tr, y_tr = np.load(DATA / "X_train.npy"), pd.read_csv(DATA / "y_train.csv")["label"].to_numpy()
X_va, y_va = np.load(DATA / "X_val.npy"), pd.read_csv(DATA / "y_val.csv")["label"].to_numpy()


def run(label, key, values, make_model, fit_kwargs=None):
    print(f"\n=== {label} ===")
    rows = []
    for v in values:
        t0 = time.time()
        m = make_model(v).fit(X_tr, y_tr, **(fit_kwargs or {}))
        tr, va = (m.predict(X_tr) == y_tr).mean(), (m.predict(X_va) == y_va).mean()
        print(f"{key}={str(v):>5}  train={tr:.4f}  val={va:.4f}  gap={tr-va:+.4f}  time={time.time()-t0:.1f}s")
        rows.append({key: v, "train_acc": tr, "val_acc": va, "gap": tr - va, "time_s": time.time() - t0})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    depths = [1, 2, 3, 5, 8, 10, 15, 20, None]
    run("Single tree vs max_depth", "depth", depths,
        lambda d: DecisionTreeClassifier(max_depth=d, random_state=42)
        ).to_csv(RESULTS / "sweep_single_tree_depth.csv", index=False)
    run("Forest (20 trees) vs max_depth", "depth", depths,
        lambda d: RandomForestClassifier(n_estimators=20, max_depth=d, random_state=42),
        {"verbose": False}).to_csv(RESULTS / "sweep_forest_depth.csv", index=False)
    run("Forest (depth 10) vs n_estimators", "n_estimators", [1, 5, 10, 20, 30, 50],
        lambda n: RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42),
        {"verbose": False}).to_csv(RESULTS / "sweep_forest_n_estimators.csv", index=False)

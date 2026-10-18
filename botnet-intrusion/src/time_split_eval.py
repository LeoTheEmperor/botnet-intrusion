"""
Stricter evaluation: TIME-ORDERED split instead of a random split.

N-BaIoT features are exponentially-decayed statistics, so consecutive rows are near-duplicates.
A random split can put a row's neighbours in train while the row itself is in test (leakage).
Here, per file: train rows come from the first 65% of the capture, test rows from the last 30%
(5% gap in between). Scaling is fitted on train only. Same model/hyper-parameters as evaluate.py.
"""
import time
import numpy as np
import pandas as pd
from load_data import FILE_LABELS, RAW_DIR
from pathlib import Path
from random_forest import RandomForestClassifier
from evaluate import confusion_matrix, precision_recall_f1

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
Xtr, ytr, Xte, yte = [], [], [], []
for fname, (botnet, subtype) in FILE_LABELS.items():
    df = pd.read_csv(RAW_DIR / fname).to_numpy(dtype=np.float64)
    n = len(df)
    train_pool, test_pool = df[: int(n * 0.65)], df[int(n * 0.70):]
    label = subtype if botnet == "benign" else f"{botnet}_{subtype}"
    Xtr.append(train_pool[rng.choice(len(train_pool), 4000, replace=False)]); ytr += [label] * 4000
    Xte.append(test_pool[rng.choice(len(test_pool), 1500, replace=False)]);  yte += [label] * 1500
    print(f"{fname:20s} rows={n:>8,}")
Xtr, Xte, ytr, yte = np.vstack(Xtr), np.vstack(Xte), np.array(ytr), np.array(yte)

mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1.0
Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std

t0 = time.time()
rf = RandomForestClassifier(n_estimators=30, max_depth=15, min_samples_split=10,
                            min_samples_leaf=5, max_features="sqrt", random_state=42)
rf.fit(Xtr, ytr, verbose=False)
pred = rf.predict(Xte)
print(f"\ntrain {time.time()-t0:.0f}s | TIME-SPLIT TEST ACCURACY: {(pred == yte).mean():.4f}")
cm = confusion_matrix(yte, pred, rf.classes_)
prf = precision_recall_f1(cm, rf.classes_)
print(prf.to_string(index=False))
print(f"macro-F1: {prf.f1.mean():.4f}")
prf.to_csv(RESULTS / "time_split_metrics.csv", index=False)
pd.DataFrame(cm, index=rf.classes_, columns=rf.classes_).to_csv(RESULTS / "time_split_confusion.csv")

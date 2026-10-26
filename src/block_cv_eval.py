"""
Diagnosis of the time-split drop (Section 6.3 of the report): INTERLEAVED block evaluation.

Each capture is cut into 10 contiguous blocks; blocks 0,2,4,6,8 are used for training and blocks
1,3,5,7,9 for testing, so training and test data come from the SAME stretches of the capture.
5% of every block is dropped at both edges so the decayed (up to ~1 minute) statistics of neighbouring
blocks cannot leak across a boundary.

 * high accuracy here + low accuracy in the time-ordered split  -> the classes ARE separable inside a
   capture, but the late part of each capture differs from the early part (drift / phase change)
 * low accuracy here too                                         -> the classes are inherently similar
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from load_data import FILE_LABELS, RAW_DIR
from random_forest import RandomForestClassifier
from evaluate import confusion_matrix, precision_recall_f1

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
N_BLOCKS, EDGE, N_TR, N_TE = 10, 0.05, 4000, 1500
rng = np.random.default_rng(42)

Xtr, ytr, Xte, yte = [], [], [], []
for fname, (botnet, subtype) in FILE_LABELS.items():
    arr = pd.read_csv(RAW_DIR / fname).to_numpy(dtype=np.float64)
    n = len(arr)
    bounds = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    tr_parts, te_parts = [], []
    for b in range(N_BLOCKS):
        lo, hi = bounds[b], bounds[b + 1]
        cut = int((hi - lo) * EDGE)
        (tr_parts if b % 2 == 0 else te_parts).append(arr[lo + cut: hi - cut])
    tr, te = np.vstack(tr_parts), np.vstack(te_parts)
    label = subtype if botnet == "benign" else f"{botnet}_{subtype}"
    Xtr.append(tr[rng.choice(len(tr), N_TR, replace=False)]); ytr += [label] * N_TR
    Xte.append(te[rng.choice(len(te), N_TE, replace=False)]); yte += [label] * N_TE
Xtr, Xte, ytr, yte = np.vstack(Xtr), np.vstack(Xte), np.array(ytr), np.array(yte)

mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1.0
t0 = time.time()
rf = RandomForestClassifier(n_estimators=30, max_depth=15, min_samples_split=10, min_samples_leaf=5,
                            max_features="sqrt", random_state=42).fit((Xtr - mean) / std, ytr, verbose=False)
pred = rf.predict((Xte - mean) / std)
print(f"train {time.time()-t0:.0f}s | INTERLEAVED-BLOCK TEST ACCURACY: {(pred == yte).mean():.4f}  ({len(yte):,} rows)")
cm = confusion_matrix(yte, pred, rf.classes_)
prf = precision_recall_f1(cm, rf.classes_)
print(prf.round(4).to_string(index=False)); print(f"macro-F1: {prf.f1.mean():.4f}")
errs = [(str(rf.classes_[i]), str(rf.classes_[j]), int(cm[i, j])) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 20]
print("confusions (> 20 rows):", errs)
prf.to_csv(RESULTS / "block_cv_metrics.csv", index=False)
pd.DataFrame(cm, index=rf.classes_, columns=rf.classes_).to_csv(RESULTS / "block_cv_confusion.csv")

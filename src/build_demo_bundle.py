#!/usr/bin/env python3
"""
Builds the small DEMO BUNDLE in demo_data/ from the full N-BaIoT files (run once; needs the dataset in data/raw).

  nbaiot_random.npz + stream_random.npz
      model trained exactly like the main pipeline (random 64/16/20 split of the 5,000-rows-per-class sample);
      stream = 300 rows per class from that split's TEST part (the model never saw them)
  nbaiot_time.npz + stream_time.npz
      model trained on the FIRST 65% of each capture; stream = 300 CONSECUTIVE rows per class from the late part of
      the capture (75% onwards). This is the honest, harder setting: four Gafgyt classes fail here.

Streams hold RAW (unscaled) float64 feature rows (float32 would flip rows that sit exactly on a split threshold); the exported models contain their own scaler. The bundle lets
demo_nbaiot.py run anywhere without the 1 GB dataset (it needs only numpy).
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from load_data import FILE_LABELS, RAW_DIR
from split_and_scale import stratified_split, zscore_fit
from random_forest import RandomForestClassifier
from model_io import export_forest, FlatForest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_data"
OUT.mkdir(exist_ok=True)
N_STREAM = 300
rng = np.random.default_rng(42)

sampled, tX, ty, sX, sy = [], [], [], [], []
for fname, (botnet, subtype) in FILE_LABELS.items():
    label = subtype if botnet == "benign" else f"{botnet}_{subtype}"
    df = pd.read_csv(RAW_DIR / fname)
    n = len(df)
    s = df.sample(n=5000, random_state=42) if n > 5000 else df            # identical to load_data.py
    sampled.append(s.assign(label=label))
    arr = df.to_numpy(dtype=np.float64)
    pool = arr[: int(n * 0.65)]
    tX.append(pool[rng.choice(len(pool), 4000, replace=False)]); ty += [label] * 4000
    start = int(n * 0.75)
    sX.append(arr[start: start + N_STREAM]); sy += [label] * N_STREAM
    print(f"{fname:20s} {n:>8,} rows", flush=True)


def fit(X, y):
    mean, std = zscore_fit(X)
    rf = RandomForestClassifier(n_estimators=30, max_depth=15, min_samples_split=10, min_samples_leaf=5,
                                max_features="sqrt", random_state=42)
    rf.fit((X - mean) / std, y, verbose=False)
    return rf, mean, std


def accuracy(model_path, X, y):
    m = FlatForest(model_path)
    pred = [m.predict_one(x) for x in X.astype(np.float64)]
    return float(np.mean(np.array(pred) == y)), pred


report = []
# ---- random mode (mirrors the main pipeline)
full = pd.concat(sampled, ignore_index=True)
feats = full.columns[:115].tolist()
train_val, test = stratified_split(full, "label", 0.20, 42)
train, _ = stratified_split(train_val, "label", 0.20, 42)
rf, mean, std = fit(train[feats].to_numpy(np.float64), train["label"].to_numpy())
export_forest(rf, OUT / "nbaiot_random.npz", mean, std)
Xs, ys = [], []
for lab in sorted(test["label"].unique()):
    part = test[test["label"] == lab].head(N_STREAM)
    Xs.append(part[feats].to_numpy(np.float64)); ys += [lab] * len(part)
Xs, ys = np.vstack(Xs), np.array(ys)
np.savez_compressed(OUT / "stream_random.npz", X=Xs, y=ys)
acc_r, _ = accuracy(OUT / "nbaiot_random.npz", Xs, ys)
report.append(("random", acc_r, len(ys)))
print(f"random mode: stream accuracy {acc_r:.4f} ({len(ys)} rows)", flush=True)

# ---- time mode (honest)
rf, mean, std = fit(np.vstack(tX), np.array(ty))
export_forest(rf, OUT / "nbaiot_time.npz", mean, std)
Xs, ys = np.vstack(sX).astype(np.float64), np.array(sy)
np.savez_compressed(OUT / "stream_time.npz", X=Xs, y=ys)
acc_t, pred = accuracy(OUT / "nbaiot_time.npz", Xs, ys)
report.append(("time", acc_t, len(ys)))
print(f"time mode:   stream accuracy {acc_t:.4f} ({len(ys)} rows)")
for lab in sorted(set(ys)):
    m = ys == lab
    print(f"   {lab:<16s} recall {(np.array(pred)[m] == lab).mean():.3f}")

(OUT / "README.md").write_text(f"""# Demo bundle

Small, ready-made inputs for `src/demo_nbaiot.py` (replay of REAL N-BaIoT feature rows through the REAL trained model).
Built by `src/build_demo_bundle.py` from the Danmini doorbell files of the public N-BaIoT dataset
(https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot; see that page for the dataset's terms and citation).

| Mode | Model | Stream ({N_STREAM} rows per class) | Accuracy on the stream |
|---|---|---|---|
| random | trained like the main pipeline (random split) | rows from that split's test part | {acc_r:.4f} |
| time | trained on the first 65% of each capture | consecutive rows from the late part of each capture | {acc_t:.4f} |

The random mode shows the headline result and is optimistic; the time mode is the honest setting where four Gafgyt classes fail.
These are replayed feature rows, not live packets. Rebuild with `python src/build_demo_bundle.py` (needs the dataset in `data/raw`).
""")
print("bundle written:", ", ".join(f"{p.name} {p.stat().st_size/1024:.0f} KB" for p in sorted(OUT.iterdir())))

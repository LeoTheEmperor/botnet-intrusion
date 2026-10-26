#!/usr/bin/env bash
# Runs the complete offline pipeline from the repository root.
# Needs the dataset in data/raw (or NBAIOT_DIR set), see data/README.md.
set -euo pipefail
PY=${PYTHON:-python}   # e.g. PYTHON=python3
cd "$(dirname "$0")/.."

"$PY" src/load_data.py          # label + sample the 11 CSVs         (~1-2 min)
"$PY" src/split_and_scale.py    # 64/16/20 stratified split, train-only scaling
"$PY" src/evaluate.py           # final model, confusion matrix, P/R/F1, ROC/AUC -> results/
"$PY" src/visualize.py          # the four dataset figures -> figures/
"$PY" src/overfitting_sweep.py  # depth / forest-size sweeps -> results/       (~4 min)
"$PY" src/time_split_eval.py    # time-ordered evaluation -> results/          (~2 min)
"$PY" src/block_cv_eval.py       # interleaved-block diagnosis -> results/      (~2 min)

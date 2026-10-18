# IoT Botnet Intrusion Detection: a From-Scratch Random Forest on N-BaIoT

Classifies network traffic of an IoT device (a Danmini doorbell) as **benign** or one of **10 Mirai / Gafgyt (BASHLITE) attack types**,
using the 115 statistical traffic features of the [N-BaIoT](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) dataset.

**Every learning component is implemented from scratch with numpy, with no scikit-learn or other ML library:** stratified splitting and scaling,
a CART decision tree, a Random Forest, PCA, and all metrics (confusion matrix, precision/recall/F1, ROC/AUC). The project also re-implements N-BaIoT's
streaming feature extractor so that live traffic can be classified on a Raspberry Pi Zero 2 W.

*7th-semester Machine Learning Project-Based Learning (PBL), Pandit Deendayal Energy University.*

## Results

| Evaluation | Test accuracy |
|---|---|
| Random stratified split (64 / 16 / 20) | **99.90%** (macro-F1 about 0.999; 11 errors in 11,000 rows) |
| Time-ordered split (train on the first 65% of each capture, test on the last 30%) | **82.27%** (family-level benign / Mirai / Gafgyt accuracy 99.98%) |

The gap is the main finding. The features are decayed running statistics, so neighbouring rows are near-duplicates and a random split leaks information into the test set.
On unseen later traffic, four Gafgyt attack types (`junk`, `udp`, `tcp`, `combo`) cannot be reliably separated, while benign traffic, the Mirai types and the attack *family* remain easy.
Details, tables and the overfitting/underfitting analysis are in the report and in `results/`.

![PCA projection](figures/pca_scatter.png)

## Repository structure

```
botnet-intrusion/
├── README.md
├── requirements.txt            # offline pipeline + notebook (numpy, pandas, matplotlib, jupyter)
├── requirements-live.txt       # Raspberry Pi live detection (numpy, scapy)
├── notebooks/
│   └── PBL_ML_Pipeline.ipynb   # the complete ML code in one notebook
├── src/
│   ├── load_data.py            # load + label the 11 CSV files
│   ├── split_and_scale.py      # stratified train/val/test split, train-only z-score scaling
│   ├── decision_tree.py        # from-scratch CART tree (Gini, vectorised split search)
│   ├── random_forest.py        # from-scratch Random Forest (bagging + feature subsampling)
│   ├── pca.py                  # from-scratch PCA
│   ├── evaluate.py             # final model + confusion matrix, P/R/F1, ROC/AUC
│   ├── visualize.py            # class distribution, correlation heatmap, PCA scatter, scree plot
│   ├── overfitting_sweep.py    # depth / forest-size sweeps (train vs validation)
│   ├── time_split_eval.py      # time-ordered evaluation
│   ├── afterimage.py           # damped incremental statistics engine (streaming)
│   ├── extractor.py            # streaming 115-feature extractor (N-BaIoT column order)
│   ├── validate_extractor.py   # numerical validation of the extractor vs the authors' code
│   ├── packet_features.py      # scapy packet -> extractor input
│   ├── model_io.py             # export / load a numpy-only forest for the Pi
│   ├── live_detector.py        # live or pcap-replay: `collect` and `detect` commands
│   ├── train_live_model.py     # train + export a model from your own captured traffic
│   └── make_test_pcaps.py      # synthetic traffic for testing without hardware
├── scripts/
│   ├── run_pipeline.sh         # runs the whole offline pipeline
│   └── fetch_reference.sh      # downloads the authors' reference code (validation only)
├── results/                    # metrics and sweep tables (CSV) from the full run
├── figures/                    # dataset figures
├── docs/                       # report, slides, demo photos
└── data/
    └── README.md               # how to obtain the dataset (data itself is not committed)
```

## Quick start

```bash
git clone https://github.com/LeoTheEmperor/botnet-intrusion.git
cd botnet-intrusion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Get the data** (about 1 GB, not in the repo): follow [`data/README.md`](data/README.md) and put the 11 Danmini CSVs in `data/raw/`.
2. **Run everything** (about 10 minutes): `bash scripts/run_pipeline.sh` (set `PYTHON=python3` if needed), or run the steps one by one:
   `python src/load_data.py`, then `split_and_scale.py`, `evaluate.py`, `visualize.py`, `overfitting_sweep.py`, `time_split_eval.py`.
3. **Or use the notebook:** `jupyter lab notebooks/PBL_ML_Pipeline.ipynb`. Setting the environment variable `PBL_FAST=1` runs a small, quick version for a smoke test.

## Live detection pipeline

Packets -> streaming 115-feature extractor -> scaler -> forest -> label per time window (and a JSON status file for a dashboard or LEDs).

```bash
pip install -r requirements-live.txt

# try it without any hardware, using synthetic traffic
python src/make_test_pcaps.py --out pcaps/train --seed 1
python src/live_detector.py collect --pcap pcaps/train/mixed.pcap --out data/live/session1.csv \
       --schedule "benign:6,syn_flood:6,benign:6,udp_flood:6,benign:6,port_scan:6,benign:6"
python src/train_live_model.py --data data/live --out models/live_model.npz --final
python src/live_detector.py detect --pcap pcaps/train/mixed.pcap --model models/live_model.npz

# on the Raspberry Pi (needs sudo; use only on a network you own)
sudo python3 src/live_detector.py collect --iface wlan0 --schedule "benign:120,syn_flood:30,benign:60" --out data/live/s1.csv
sudo python3 src/live_detector.py detect  --iface wlan0 --model models/live_model.npz
```

Record training data as **continuous sessions labelled by time segment** (`--schedule`), not one file per class: the long-window features remember history,
and cold-start captures taught a model that missed a port scan which followed earlier floods. Synthetic traffic is far easier to separate than real traffic,
so it only shows that the pipeline works.

To check the extractor numerically against the authors' implementation: `bash scripts/fetch_reference.sh && python src/validate_extractor.py`
(max relative deviation about 9e-13 over 4,000 packets x 115 features).

## Demo hardware (planned)

Raspberry Pi Zero 2 W (headless, Wi-Fi), power supply, microSD card, three LEDs with resistors as a physical indicator, and a second machine on an **isolated network you own**
to generate SYN floods, UDP floods and port scans.

## Project status

- [x] Dataset, preprocessing, from-scratch tree / forest / PCA / metrics
- [x] Visualisations, overfitting / underfitting analysis, time-ordered evaluation
- [x] Streaming feature extractor, validated against the reference implementation
- [x] Live capture / detect pipeline, tested on synthetic traffic
- [x] Notebook with the full ML code
- [ ] Raspberry Pi setup, extractor benchmark on the board, live demo with real traffic
- [ ] Web dashboard / LED indicator
- [ ] Final report, slides and demo photos in `docs/`

## Limitations

- The model learns one device's traffic (Danmini doorbell); other devices and networks are untested, and a different live device needs its own training data.
- The balanced 5,000-rows-per-class sample is a design choice; the raw dataset is heavily imbalanced (benign is under 5% of rows).
- Four Gafgyt classes do not generalise to later portions of the captures (see Results); the explanation is a hypothesis that still needs block cross-validation.
- The extractor's stream keying for `HH` / `HpHp` is inferred from the dataset (see `src/extractor.py`), validated for numerics but not against the original N-BaIoT extractor, which is not public.

## Responsible use

Generate attack traffic only on a network you own or are explicitly authorised to test. Do not run the flood or scan tests on a shared, campus or public network.

## References

1. Y. Meidan et al., *N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders*, IEEE Pervasive Computing, 2018 (arXiv:1805.03409).
2. Y. Mirsky et al., *Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection*, NDSS 2018 (arXiv:1802.09089).
3. Kitsune-py reference implementation: https://github.com/ymirsky/Kitsune-py (used only to validate our extractor).
4. UCI Machine Learning Repository, *detection_of_IoT_botnet_attacks_N_BaIoT* (dataset 442).

## Author

Tanishq Nayak, Computer Science and Engineering, Pandit Deendayal Energy University.

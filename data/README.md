# Data

Nothing in this folder is committed to git (the dataset is ~1 GB). Download it and place the files in `data/raw/`.

## N-BaIoT, Danmini doorbell

Source: UCI Machine Learning Repository, *detection_of_IoT_botnet_attacks_N_BaIoT*
(https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot), also mirrored on Kaggle.
Use the files of device 1 (Danmini doorbell) and arrange them as:

```
data/raw/
├── benign_traffic.csv
├── ack.csv        scan_1.csv    syn.csv      udp_1.csv          # Mirai
└── combo.csv      junk.csv      scan.csv     tcp.csv   udp.csv   udpplain.csv   # Gafgyt (BASHLITE)
```

Mirai and Gafgyt both ship a `scan.csv` and a `udp.csv`; the Mirai ones are renamed `scan_1.csv` and `udp_1.csv`
so that all 11 files can live in one folder (this is what `src/load_data.py` expects).

To keep the data somewhere else, set the environment variable `NBAIOT_DIR` to that folder.

## Generated files (also not committed)

`python src/load_data.py` and `python src/split_and_scale.py` create, in this folder: the labelled 55,000-row sample,
the scaled train/validation/test arrays (`X_*.npy`, `y_*.csv`) and the scaler parameters.

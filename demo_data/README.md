# Demo bundle

Small, ready-made inputs for `src/demo_nbaiot.py` (replay of REAL N-BaIoT feature rows through the REAL trained model).
Built by `src/build_demo_bundle.py` from the Danmini doorbell files of the public N-BaIoT dataset
(https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot; see that page for the dataset's terms and citation).

| Mode | Model | Stream (300 rows per class) | Accuracy on the stream |
|---|---|---|---|
| random | trained like the main pipeline (random split) | rows from that split's test part | 0.9988 |
| time | trained on the first 65% of each capture | consecutive rows from the late part of each capture | 0.9055 |

The random mode shows the headline result and is optimistic; the time mode is the honest setting where four Gafgyt classes fail.
These are replayed feature rows, not live packets. Rebuild with `python src/build_demo_bundle.py` (needs the dataset in `data/raw`).

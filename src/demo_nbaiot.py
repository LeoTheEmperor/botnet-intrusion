#!/usr/bin/env python3
"""
Dataset-replay demo: REAL N-BaIoT feature rows run through the REAL trained model, shown on the live dashboard with
the ground truth next to every prediction. No network, no packets, no root and no dataset download needed: it uses the
small bundle in demo_data/ and only numpy, so it also runs on the Raspberry Pi.

    python src/demo_nbaiot.py                  # random-split model: the headline result (optimistic)
    python src/demo_nbaiot.py --mode time      # honest setting: model trained on early traffic, replayed on LATE traffic
    python src/demo_nbaiot.py --plan "benign:10,mirai_syn:10,benign:10,gafgyt_junk:10"

What it shows: classification of real botnet-infected-device traffic FEATURES. What it is not: live packets. Say so.
"""
import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

from dashboard import DashboardState, start_dashboard
from model_io import FlatForest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = {
    "random": "benign:10,mirai_syn:10,benign:8,gafgyt_tcp:10,benign:8,mirai_scan:10,benign:8,gafgyt_combo:10",
    "time": "benign:10,mirai_ack:10,benign:8,gafgyt_junk:10,benign:8,gafgyt_combo:10,benign:8,gafgyt_tcp:10",
}


def parse_plan(text, known):
    plan = []
    for part in text.split(","):
        label, secs = part.rsplit(":", 1)
        label = label.strip()
        if label not in known:
            sys.exit(f"unknown class '{label}'; choose from: {', '.join(sorted(known))}")
        plan.append((label, int(float(secs))))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["random", "time"], default="random")
    ap.add_argument("--plan", default=None, help="timeline 'class:seconds,...' (default depends on the mode)")
    ap.add_argument("--rate", type=int, default=30, help="samples per one-second window")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--lan", action="store_true", help="serve on 0.0.0.0 (other devices on YOUR network can open the page)")
    ap.add_argument("--alert-share", type=float, default=0.5)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-hold", action="store_true", help="exit when the timeline ends")
    a = ap.parse_args()

    bundle = ROOT / "demo_data"
    model = FlatForest(bundle / f"nbaiot_{a.mode}.npz")
    z = np.load(bundle / f"stream_{a.mode}.npz")
    X, y = z["X"].astype(np.float64), z["y"]
    pools = {str(lab): X[y == lab] for lab in np.unique(y)}
    plan = parse_plan(a.plan or DEFAULT_PLAN[a.mode], pools)
    rng, cursor = np.random.default_rng(7), {lab: 0 for lab in pools}
    order = {lab: (rng.permutation(len(p)) if a.mode == "random" else np.arange(len(p))) for lab, p in pools.items()}

    def take(label, n):
        """Next n rows of a class: shuffled cycles in random mode, consecutive rows (restarting) in time mode."""
        idx = [order[label][(cursor[label] + i) % len(pools[label])] for i in range(n)]
        cursor[label] += n
        return pools[label][idx]

    mode_name = {"random": "random-split model", "time": "honest time-ordered setting"}[a.mode]
    dash = DashboardState(classes=model.classes, source=f"N-BaIoT held-out rows, replay ({mode_name})",
                          benign_label="benign", window_s=1.0, unit="samples")
    host = "0.0.0.0" if a.lan else "127.0.0.1"
    try:
        start_dashboard(dash, host, a.port)
    except OSError as e:
        sys.exit(f"cannot start dashboard on {host}:{a.port}: {e}")
    url = f"http://localhost:{a.port}/"
    print(f"dashboard: {url}   ({mode_name}; real N-BaIoT feature rows replayed, not live packets)", flush=True)
    if not a.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()

    t0, w, right, total = time.time(), 0, 0, 0
    try:
        for truth, secs in plan:
            for _ in range(secs):
                preds = [model.predict_one(row) for row in take(truth, a.rate)]
                counts = {}
                for p in preds:
                    counts[p] = counts.get(p, 0) + 1
                label = max(counts, key=counts.get)
                share = counts[label] / len(preds)
                alert = label != "benign" and share >= a.alert_share
                right += sum(p == truth for p in preds); total += len(preds)
                w += 1
                dash.push(w, label, share, alert, counts, len(preds), truth=truth)
                print(f"[t+{w:4d}s] samples={len(preds):3d}  predicted={label:<16s} {share:4.0%}  truth={truth:<16s} "
                      f"{'*** ALERT ***' if alert else 'ok'}", flush=True)
                delay = t0 + w / a.speed - time.time()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass
    print(f"overall: {right / max(total, 1):.1%} of {total:,} samples classified correctly", flush=True)
    dash.finished = True
    if not a.no_hold:
        print("timeline finished - dashboard stays up, press Ctrl+C to quit", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

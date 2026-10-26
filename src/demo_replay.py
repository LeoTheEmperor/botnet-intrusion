#!/usr/bin/env python3
"""
One-command rehearsal of the whole live demo on a laptop: no Raspberry Pi, no hardware, no root.

  1. generates SYNTHETIC traffic (benign, SYN flood, UDP flood, port scan) as pcap files
  2. records a labelled training session from it and trains a model
  3. replays UNSEEN traffic at its recorded speed through the detector, with the web dashboard

    pip install numpy scapy            # scapy is only used to generate the synthetic pcaps
    python src/demo_replay.py          # opens http://localhost:8080

Files go to the git-ignored folders pcaps/, data/live_demo/ and models/. Synthetic traffic is far easier to
separate than real traffic: this shows that the pipeline and the dashboard work, nothing more. Say so when you demo it.
"""
import argparse
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(script, *args):
    subprocess.run([sys.executable, str(SRC / script), *map(str, args)], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", type=float, default=8.0, help="seconds per traffic segment (7 segments in the timeline)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--lan", action="store_true", help="serve on 0.0.0.0 so other devices on YOUR network can open the page")
    ap.add_argument("--regen", action="store_true", help="regenerate traffic and retrain even if files exist")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-hold", action="store_true", help="exit when the replay ends (default: keep the dashboard up)")
    a = ap.parse_args()

    train_dir, test_dir = ROOT / "pcaps" / "demo_train", ROOT / "pcaps" / "demo_test"
    model, session_dir = ROOT / "models" / "demo_model.npz", ROOT / "data" / "live_demo"
    seg = f"{a.segment:g}"
    schedule = ",".join(f"{l}:{seg}" for l in ["benign", "syn_flood", "benign", "udp_flood", "benign", "port_scan", "benign"])

    if a.regen or not (train_dir / "mixed.pcap").exists() or not (test_dir / "mixed.pcap").exists():
        print("== 1/3 generating synthetic traffic ==", flush=True)
        run("make_test_pcaps.py", "--out", train_dir, "--seed", 1, "--seconds", 10, "--segment", seg)
        run("make_test_pcaps.py", "--out", test_dir, "--seed", 2, "--seconds", 10, "--segment", seg)
    if a.regen or not model.exists():
        print("== 2/3 recording a labelled session and training ==", flush=True)
        session = session_dir / "session.csv"
        session.unlink(missing_ok=True)
        run("live_detector.py", "collect", "--pcap", train_dir / "mixed.pcap", "--schedule", schedule, "--out", session)
        run("train_live_model.py", "--data", session_dir, "--out", model, "--trees", 20, "--final")

    print("== 3/3 replaying unseen traffic with the dashboard ==", flush=True)
    url = f"http://localhost:{a.port}/"
    if not a.no_browser:
        threading.Timer(2.0, webbrowser.open, [url]).start()
    cmd = ["detect", "--pcap", test_dir / "mixed.pcap", "--model", model, "--window", 1, "--dashboard", a.port,
           "--realtime", "--speed", a.speed]
    if a.lan:
        cmd += ["--dashboard-host", "0.0.0.0"]
    if a.no_hold:
        cmd += ["--no-hold"]
    run("live_detector.py", *cmd)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Live (or pcap-replay) network-traffic pipeline:  packets -> 115 streaming features -> classifier.

  collect : record labelled feature rows (to build your own training data on the Pi)
      python3 live_detector.py collect --iface wlan0 --label benign --duration 120 --out data/live/benign.csv
      python3 live_detector.py collect --pcap flood.pcap --label syn_flood --out data/live/syn_flood.csv

      # RECOMMENDED: one continuous session, labelled by time segment (label, seconds), so the
      # long-window features see realistic history (attacks that follow normal traffic, etc.)
      python3 live_detector.py collect --iface wlan0 --schedule "benign:60,syn_flood:30,benign:60,udp_flood:30" --out data/live/session1.csv

  detect  : classify traffic continuously and print one summary line per time window
      sudo python3 live_detector.py detect --iface wlan0 --model models/live_model.npz
      python3 live_detector.py detect --pcap pcaps/mixed.pcap --model models/live_model.npz

Live capture needs root (sudo) and scapy. Only capture on networks you own or administer.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from extractor import FeatureExtractor, FEATURE_NAMES
from packet_features import parse_packet


def run_source(args, handler):
    from scapy.all import PcapReader, sniff
    if args.pcap:
        with PcapReader(args.pcap) as reader:
            for pkt in reader:
                handler(pkt)
    else:
        sniff(iface=args.iface, prn=handler, store=False, filter=args.bpf, timeout=args.duration)


def parse_schedule(text):
    """'benign:60,syn_flood:30' -> [(0,60,'benign'), (60,90,'syn_flood')]  (seconds from first packet)"""
    segs, t = [], 0.0
    for part in text.split(","):
        label, dur = part.rsplit(":", 1)
        segs.append((t, t + float(dur), label.strip()))
        t += float(dur)
    return segs


def cmd_collect(args):
    if not args.label and not args.schedule:
        sys.exit("collect needs --label (whole capture is one class) or --schedule (time-segmented session)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists() or not args.append
    fh = open(out, "w" if new_file else "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(FEATURE_NAMES + ["label"])
    segs = parse_schedule(args.schedule) if args.schedule else None
    fx, st, per_label = FeatureExtractor(), {"seen": 0, "rows": 0, "t0": None}, Counter()

    def label_at(ts):
        if segs is None:
            return args.label
        off = ts - st["t0"]
        for a, b, lab in segs:
            if a <= off < b:                       # drop rows near a boundary: label may be wrong there
                return lab if (off - a >= args.trim_edges and b - off >= args.trim_edges) else None
        return None

    def handler(pkt):
        rec = parse_packet(pkt)
        if rec is None:
            return
        if st["t0"] is None:
            st["t0"] = rec[0]
        st["seen"] += 1
        vec = fx.update(*rec)                       # always update: streams need the full history
        if st["seen"] > args.warmup:
            lab = label_at(rec[0])
            if lab is not None:
                writer.writerow([f"{v:.10g}" for v in vec] + [lab])
                st["rows"] += 1
                per_label[lab] += 1

    try:
        run_source(args, handler)
    except KeyboardInterrupt:
        pass
    fh.close()
    print(f"collect: {st['seen']} packets parsed, {st['rows']} rows written to {out}  {dict(per_label)}")


def cmd_detect(args):
    from model_io import FlatForest
    model, fx = FlatForest(args.model), FeatureExtractor()
    st = {"t0": None, "wstart": None, "seen": 0}
    win, total = Counter(), Counter()

    def flush(now):
        n = sum(win.values())
        if n == 0:
            return
        label, c = win.most_common(1)[0]
        share = c / n
        alert = label != args.benign_label and share >= args.alert_share
        print(f"[t+{now - st['t0']:7.2f}s] packets={n:6d}  dominant={label:<14s} {share:5.0%}  "
              f"{'*** ALERT ***' if alert else 'ok'}", flush=True)
        if args.status_file:                    # simple hook for a dashboard / LED script
            Path(args.status_file).write_text(json.dumps(
                {"t": now, "label": label, "share": share, "alert": alert, "counts": dict(win)}))
        win.clear()

    def handler(pkt):
        rec = parse_packet(pkt)
        if rec is None:
            return
        ts = rec[0]
        if st["t0"] is None:
            st["t0"] = st["wstart"] = ts
        st["seen"] += 1
        vec = fx.update(*rec)
        if st["seen"] > args.warmup and st["seen"] % args.every == 0:
            label = model.predict_one(vec)
            win[label] += 1
            total[label] += 1
        if ts - st["wstart"] >= args.window:
            flush(ts)
            st["wstart"] = ts

    try:
        run_source(args, handler)
    except KeyboardInterrupt:
        pass
    flush(st["wstart"] or 0)
    print("overall:", dict(total))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "detect"):
        p = sub.add_parser(name)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--iface", help="network interface to sniff (needs root)")
        src.add_argument("--pcap", help="read packets from a pcap file instead")
        p.add_argument("--duration", type=float, default=None, help="stop live capture after N seconds")
        p.add_argument("--bpf", default=None, help="optional BPF filter, e.g. 'not port 22'")
        p.add_argument("--warmup", type=int, default=100, help="ignore the first N packets")
    c = sub.choices["collect"]
    c.add_argument("--label", default=None, help="class label for the WHOLE capture")
    c.add_argument("--schedule", default=None, help="time-segmented labels, e.g. 'benign:60,syn_flood:30,benign:60'")
    c.add_argument("--trim-edges", type=float, default=0.5, help="drop rows within N s of a segment boundary")
    c.add_argument("--out", required=True)
    c.add_argument("--append", action="store_true")
    d = sub.choices["detect"]
    d.add_argument("--model", required=True)
    d.add_argument("--window", type=float, default=1.0, help="seconds per summary line")
    d.add_argument("--every", type=int, default=1, help="classify every Nth packet (speed knob for the Pi)")
    d.add_argument("--benign-label", default="benign")
    d.add_argument("--alert-share", type=float, default=0.5)
    d.add_argument("--status-file", default=None, help="write latest status as JSON (for a dashboard)")
    args = ap.parse_args()
    {"collect": cmd_collect, "detect": cmd_detect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

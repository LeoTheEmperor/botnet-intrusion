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

  dashboard : add --dashboard 8080 to `detect` to serve a live web page (see dashboard.py)
      sudo python3 live_detector.py detect --iface wlan0 --model models/live_model.npz --dashboard 8080 --dashboard-host 0.0.0.0
      # no hardware? replay a recording at its original speed (--speed 4 = four times faster):
      python3 live_detector.py detect --pcap pcaps/test/mixed.pcap --model models/live_model.npz --dashboard 8080 --realtime

Live capture needs root (sudo). scapy is optional (only for pcapng files, non-Linux live capture, or --engine scapy). Only capture on networks you own or administer.
"""
import argparse
import csv
import itertools
import json
import socket
import sys
import time
from collections import Counter
from pathlib import Path

from extractor import FeatureExtractor, FEATURE_NAMES


def _fast_records(args):
    """Parsed packet records from the dependency-free engine (fastcapture.py)."""
    from fastcapture import iter_pcap, iter_live, parse_frame
    exclude = getattr(args, "exclude_ports", None)
    source = iter_pcap(args.pcap) if args.pcap else iter_live(args.iface, args.duration)
    for ts, buf, wire in source:
        rec = parse_frame(buf, ts, wire)
        if rec is None:
            continue
        if exclude and not args.pcap and (rec[6] in exclude or rec[7] in exclude):
            continue                                  # e.g. the dashboard's own HTTP traffic
        yield rec


def _scapy_pcap_records(path):
    from scapy.all import PcapReader
    from packet_features import parse_packet
    with PcapReader(path) as reader:
        for pkt in reader:
            rec = parse_packet(pkt)
            if rec is not None:
                yield rec


def _feed(records, args, handler):
    """Hand records to the handler, optionally pacing a pcap replay to the recorded timestamps."""
    realtime = args.pcap and getattr(args, "realtime", False)
    speed = max(getattr(args, "speed", 1.0), 1e-3)
    t0 = wall0 = None
    for rec in records:
        if realtime:
            if t0 is None:
                t0, wall0 = rec[0], time.time()
            delay = (rec[0] - t0) / speed - (time.time() - wall0)
            if delay > 0:
                time.sleep(delay)
        handler(rec)


def run_source(args, handler):
    """Feed parsed packet records to handler(rec). Engines: 'fast' (stdlib only, default where
    possible), 'scapy' (pcapng, non-Linux live capture, BPF filters), 'auto' (fast, else scapy)."""
    engine = getattr(args, "engine", "auto")
    if engine == "fast" or (engine == "auto" and (args.pcap or hasattr(socket, "AF_PACKET"))):
        gen = _fast_records(args)
        try:
            first = next(gen)
        except StopIteration:
            return
        except (ValueError, OSError) as e:
            if engine == "fast":
                raise
            print(f"fast engine unavailable ({e}); falling back to scapy", file=sys.stderr)
        else:
            _feed(itertools.chain([first], gen), args, handler)
            return
    if args.pcap:
        _feed(_scapy_pcap_records(args.pcap), args, handler)
    else:
        from scapy.all import sniff
        from packet_features import parse_packet
        sniff(iface=args.iface, prn=lambda p: handler(parse_packet(p)), store=False,
              filter=args.bpf, timeout=args.duration)


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

    def handler(rec):
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

    dash = None
    if args.dashboard:
        from dashboard import DashboardState, start_dashboard
        dash = DashboardState(classes=model.classes, source=args.pcap or args.iface,
                              benign_label=args.benign_label, window_s=args.window)
        if args.iface:                            # keep the dashboard's own HTTP traffic out of the classifier
            args.exclude_ports = {str(args.dashboard)}                         # fast engine
            exclude = f"not tcp port {args.dashboard}"                         # scapy engine (BPF)
            args.bpf = f"({args.bpf}) and {exclude}" if args.bpf else exclude
        try:
            start_dashboard(dash, args.dashboard_host, args.dashboard)
        except OSError as e:
            sys.exit(f"cannot start dashboard on {args.dashboard_host}:{args.dashboard}: {e}")
        shown = "localhost" if args.dashboard_host in ("127.0.0.1", "0.0.0.0") else args.dashboard_host
        print(f"dashboard: http://{shown}:{args.dashboard}/   (bound to {args.dashboard_host})", flush=True)
        if args.dashboard_host == "0.0.0.0":
            print("  note: reachable by every device on this network - use only on a network you own", flush=True)

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
        if dash is not None:
            dash.push(now - st["t0"], label, share, alert, win, n)
        win.clear()

    def handler(rec):
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
    if dash is not None:
        dash.finished = True
        if args.pcap and not args.no_hold:        # a finite replay: keep the page up for viewing
            print("replay finished - dashboard stays up, press Ctrl+C to quit", flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "detect"):
        p = sub.add_parser(name)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--iface", help="network interface to sniff (needs root)")
        src.add_argument("--pcap", help="read packets from a pcap file instead")
        p.add_argument("--duration", type=float, default=None, help="stop live capture after N seconds")
        p.add_argument("--bpf", default=None, help="optional BPF filter, e.g. 'not port 22' (scapy engine only)")
        p.add_argument("--engine", choices=["auto", "fast", "scapy"], default="auto",
                       help="packet engine: fast = stdlib-only parser (default where possible), scapy = fallback")
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
    d.add_argument("--dashboard", type=int, default=None, metavar="PORT", help="serve a live web dashboard on this port")
    d.add_argument("--dashboard-host", default="127.0.0.1", help="bind address (use 0.0.0.0 to view from another device on YOUR network)")
    d.add_argument("--no-hold", action="store_true", help="with --pcap: exit when the replay ends instead of keeping the dashboard up")
    d.add_argument("--realtime", action="store_true", help="with --pcap: replay at the recorded speed")
    d.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier for --realtime")
    args = ap.parse_args()
    {"collect": cmd_collect, "detect": cmd_detect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

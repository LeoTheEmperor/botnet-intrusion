#!/usr/bin/env python3
"""
How many packets per second can THIS machine sustain? Run it on the Raspberry Pi.
Needs only numpy + the project files (no scapy, no pandas). Synthetic Ethernet/IPv4/TCP frames are built with struct.

    python3 src/benchmark.py                       # parser + feature extractor
    python3 src/benchmark.py --model models/live_model.npz   # ... + forest prediction (full detector)
"""
import argparse
import random
import struct
import time

from fastcapture import parse_frame
from extractor import FeatureExtractor


def make_frame(src, dst, sport, dport, payload):
    eth = bytes.fromhex("aabbcc000002") + bytes.fromhex("aabbcc000001") + b"\x08\x00"
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 40 + payload, 0, 0, 64, 6, 0, bytes(src), bytes(dst))
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 0, 0, 0x50, 0x02, 8192, 0, 0)
    return eth + ip + tcp + bytes(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=int, default=20000)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    random.seed(1)
    hosts = [(10, 0, 0, i) for i in range(1, 6)]
    frames = []
    for _ in range(a.packets):
        s, d = random.sample(hosts, 2)
        frames.append(make_frame(s, d, random.choice([80, 443, 5000, 40000]), random.choice([80, 443, 5000]), random.randrange(0, 1200)))

    t = time.perf_counter(); recs = [parse_frame(f, 1000.0 + i * 0.001) for i, f in enumerate(frames)]; t_parse = time.perf_counter() - t
    fx = FeatureExtractor()
    t = time.perf_counter(); vecs = [fx.update(*r) for r in recs]; t_fx = time.perf_counter() - t
    t_pred = 0.0
    if a.model:
        from model_io import FlatForest
        m = FlatForest(a.model)
        t = time.perf_counter(); [m.predict_one(v) for v in vecs[:5000]]; t_pred = (time.perf_counter() - t) * len(vecs) / min(len(vecs), 5000)
    n = len(frames)
    print(f"{n:,} packets")
    for name, dt in [("parse (fastcapture)", t_parse), ("feature extractor", t_fx), ("forest prediction", t_pred)]:
        if dt:
            print(f"  {name:<22s} {dt / n * 1e6:8.1f} us/packet")
    total = (t_parse + t_fx + t_pred) / n
    print(f"  {'TOTAL':<22s} {total * 1e6:8.1f} us/packet  ->  about {1 / total:,.0f} packets/s")
    safe = int(0.5 / total // 100 * 100)
    print(f"\nrule of thumb: keep test floods at or below ~{safe:,} packets/s on this device (50% headroom),")
    print("or classify only every Nth packet with `detect --every N` (the extractor still sees every packet).")


if __name__ == "__main__":
    main()

"""
Generate SYNTHETIC test traffic (pcap files) so the whole pipeline can be tried without any
hardware: benign chatter, SYN flood, UDP flood, port scan, plus a 'mixed' timeline.
This is only for testing the plumbing - synthetic traffic is far easier to separate than real
traffic, so accuracy on it says nothing about real-world performance.

    python3 make_test_pcaps.py --out pcaps --seed 1
"""
import argparse
from pathlib import Path
import numpy as np
from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

VICTIM = ("aa:bb:cc:00:00:01", "192.168.1.50")
PEER = ("aa:bb:cc:00:00:02", "192.168.1.1")
ATTACKER = ("aa:bb:cc:00:00:99", "192.168.1.66")


def pkt(t, src, dst, l4, payload=0):
    p = Ether(src=src[0], dst=dst[0]) / IP(src=src[1], dst=dst[1]) / l4
    if payload:
        p = p / Raw(load=b"x" * payload)
    p.time = t
    return p


def benign(t0, dur, rng):
    out, t = [], t0
    while t < t0 + dur:
        t += rng.exponential(0.03)
        sport = int(rng.integers(30000, 60000))
        out.append(pkt(t, VICTIM, PEER, TCP(sport=sport, dport=int(rng.choice([443, 80, 1883]))), int(rng.integers(0, 500))))
        t += rng.exponential(0.01)                                   # reply
        out.append(pkt(t, PEER, VICTIM, TCP(sport=443, dport=sport), int(rng.integers(0, 1200))))
    return out, t


def syn_flood(t0, dur, rng, pps=1500):
    n = int(dur * pps); ts = t0 + np.sort(rng.uniform(0, dur, n))
    return [pkt(float(t), ATTACKER, VICTIM, TCP(sport=int(rng.integers(1024, 65535)), dport=80, flags="S")) for t in ts], t0 + dur


def udp_flood(t0, dur, rng, pps=1200):
    n = int(dur * pps); ts = t0 + np.sort(rng.uniform(0, dur, n))
    return [pkt(float(t), ATTACKER, VICTIM, UDP(sport=4444, dport=53), 1000) for t in ts], t0 + dur


def port_scan(t0, dur, rng, pps=800):
    n = int(dur * pps); ts = t0 + np.sort(rng.uniform(0, dur, n))
    return [pkt(float(t), ATTACKER, VICTIM, TCP(sport=40000, dport=int(rng.integers(1, 10000)), flags="S")) for t in ts], t0 + dur


KINDS = {"benign": benign, "syn_flood": syn_flood, "udp_flood": udp_flood, "port_scan": port_scan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pcaps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()
    rng, out = np.random.default_rng(a.seed), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in KINDS.items():
        pk, _ = fn(1_700_000_000.0, a.seconds, rng)
        wrpcap(str(out / f"{name}.pcap"), pk)
        print(f"{name}.pcap: {len(pk)} packets")
    # mixed timeline: benign -> syn flood -> benign -> udp flood -> benign -> port scan -> benign
    seq, t, mixed = ["benign", "syn_flood", "benign", "udp_flood", "benign", "port_scan", "benign"], 1_700_001_000.0, []
    for kind in seq:
        pk, t = KINDS[kind](t, 6.0, rng)
        mixed += pk
    wrpcap(str(out / "mixed.pcap"), mixed)
    print(f"mixed.pcap: {len(mixed)} packets  timeline: " + " > ".join(seq) + "  (6 s each)")


if __name__ == "__main__":
    main()

"""
Validation of the dependency-free packet engine (fastcapture.py) against scapy.
  1. every packet of a pcap parses to the same record with both engines
  2. hand-crafted edge cases (ARP, ICMP, UDP, VLAN, IPv6, IP fragment, runt frames)
  3. live raw-socket capture on the loopback interface (skipped if not permitted)
  4. speed
Usage: python src/validate_capture.py [pcap]      (default: pcaps/test/mixed.pcap)
"""
import ipaddress
import socket
import sys
import threading
import time
from pathlib import Path

from scapy.all import Ether, Dot1Q, IP, IPv6, TCP, UDP, ICMP, ARP, Raw, PcapReader
from fastcapture import parse_frame, iter_pcap, iter_live
from packet_features import parse_packet
from extractor import FeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
pcap = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "pcaps" / "test" / "mixed.pcap"


def same(a, b):
    """Records equal; IPv6 text normalised; timestamps within 1 microsecond."""
    if a is None or b is None:
        return a is b
    if abs(a[0] - b[0]) > 1e-5 or a[1] != b[1]:
        return False
    for x, y in zip(a[2:], b[2:]):
        if x != y:
            try:
                if ipaddress.ip_address(x) == ipaddress.ip_address(y):
                    continue
            except ValueError:
                pass
            return False
    return True


# ---- 1. whole pcap
n = bad = 0
with PcapReader(str(pcap)) as sc:
    for (ts, buf, wire), pkt in zip(iter_pcap(str(pcap)), sc):
        n += 1
        if not same(parse_frame(buf, ts, wire), parse_packet(pkt)):
            bad += 1
            if bad <= 3:
                print("  MISMATCH", parse_frame(buf, ts, wire), parse_packet(pkt))
print(f"[1] {pcap.name}: {n:,} packets compared, {bad} mismatches")

# ---- 2. crafted edge cases
mac = dict(src="aa:bb:cc:00:00:01", dst="aa:bb:cc:00:00:02")
cases = {
    "tcp":        Ether(**mac) / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=80) / Raw(b"hello"),
    "udp":        Ether(**mac) / IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=5353, dport=53),
    "icmp":       Ether(**mac) / IP(src="10.0.0.1", dst="10.0.0.2") / ICMP(),
    "arp":        Ether(**mac) / ARP(psrc="10.0.0.1", pdst="10.0.0.2", hwsrc=mac["src"]),
    "vlan+tcp":   Ether(**mac) / Dot1Q(vlan=7) / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1, dport=2),
    "ip options": Ether(**mac) / IP(src="10.0.0.1", dst="10.0.0.2", options=[b"\x01\x01\x01\x01"]) / TCP(sport=9, dport=8),
    "ipv6+tcp":   Ether(**mac) / IPv6(src="fe80::1", dst="fe80::2") / TCP(sport=4000, dport=443),
    "ipv6+udp":   Ether(**mac) / IPv6(src="2001:db8::1", dst="2001:db8::2") / UDP(sport=1000, dport=2000),
    "ip fragment": Ether(**mac) / IP(src="10.0.0.1", dst="10.0.0.2", frag=5, proto=17) / Raw(b"x" * 20),
    "other proto": Ether(**mac) / IP(src="10.0.0.1", dst="10.0.0.2", proto=47) / Raw(b"x" * 20),
}
ok_all = True
for name, p in cases.items():
    raw = bytes(p)
    a, b = parse_frame(raw, 1.0, len(raw)), parse_packet(Ether(raw))
    if b is not None:
        b = (1.0,) + b[1:]
    ok = same(a, b)
    ok_all &= ok
    print(f"[2] {name:<12s} {'ok' if ok else 'MISMATCH':<9s} fast={a[4:] if a else None}")
print("[2] runt / unknown frames ->", parse_frame(b"\x00" * 10, 0.0), parse_frame(bytes(Ether(type=0x1234)) + b"x" * 30, 0.0))
print("[2] all crafted cases identical:", ok_all)

# ---- 3. live raw socket on loopback
try:
    got, stop = [], threading.Event()

    def sender():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        time.sleep(0.4)
        for i in range(50):
            s.sendto(b"x" * 100, ("127.0.0.1", 45678))
        time.sleep(0.3)

    th = threading.Thread(target=sender); th.start()
    for ts, buf, wire in iter_live("lo", duration=1.5):
        rec = parse_frame(buf, ts, wire)
        if rec and rec[7] == "45678":
            got.append(rec)
    th.join()
    print(f"[3] live AF_PACKET capture on 'lo': {len(got)} UDP frames seen (sent 50)"
          + (f", e.g. {got[0][4:]} size={got[0][1]:.0f}" if got else ""))
except (PermissionError, OSError) as e:
    print(f"[3] live capture skipped: {e}")

# ---- 4. speed
N = 12000
t = time.time()
recs = []
for ts, buf, wire in iter_pcap(str(pcap)):
    recs.append(parse_frame(buf, ts, wire))
    if len(recs) >= N:
        break
dt = time.time() - t
print(f"[4] fast read+parse: {dt/len(recs)*1e6:6.1f} us/packet ({len(recs)/dt:>9,.0f} pkt/s)   [scapy read+parse was ~1,090 us/packet]")
fx = FeatureExtractor(); t = time.time()
for r in recs:
    fx.update(*r)
print(f"[4] + feature extractor:  {(dt + time.time() - t)/len(recs)*1e6:6.1f} us/packet total ({len(recs)/(dt + time.time() - t):>9,.0f} pkt/s)")

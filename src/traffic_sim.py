#!/usr/bin/env python3
"""
Rate-limited traffic generator for testing the detector on YOUR OWN lab network.

Ordinary TCP/UDP sockets only: no IP spoofing, no raw sockets, no root needed; runs on Linux, macOS and Windows.
Modes (each capped in rate and duration):
  benign : light, irregular chatter (small UDP datagrams + occasional TCP connection attempts)
  syn    : rapid TCP connection attempts to one port        (SYN-flood-like pressure on the target)
  udp    : fixed-size UDP datagrams to one port             (UDP-flood-like)
  scan   : TCP connect scan across ports 1..N               (port-scan-like)
  session: a labelled timeline of the above; sends a tiny UDP MARKER at the start of every segment so
           `live_detector.py collect --markers ...` can label the Pi's capture without clock sync

Safety rails: the target must be a private or loopback IPv4 address (no hostnames), the rate is capped at
3000/s, one segment lasts at most 10 minutes, and you must pass --i-own-this-network. Use it only against
devices and networks you own or are explicitly authorised to test - never on campus or public networks.

Examples (target = your Raspberry Pi):
  python3 src/traffic_sim.py syn --target 192.168.1.50 --rate 400 --duration 20 --i-own-this-network
  python3 src/traffic_sim.py session --target 192.168.1.50 --i-own-this-network \\
          --plan "benign:90,syn_flood:30,benign:30,udp_flood:30,benign:30,port_scan:30,benign:30"
"""
import argparse
import ipaddress
import random
import socket
import sys
import time

MAX_RATE, MAX_SEGMENT_S, MAX_PLAN_S = 3000, 600, 3600
LABELS = ["benign", "syn_flood", "udp_flood", "port_scan"]          # marker index = position in this list
MODE_OF = {"benign": "benign", "syn_flood": "syn", "udp_flood": "udp", "port_scan": "scan"}
DEFAULT_RATE = {"benign": 30, "syn": 400, "udp": 400, "scan": 400}


def check_target(host):
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        sys.exit("target must be an IP address (hostnames are not accepted)")
    if ip.version != 4 or not (ip.is_private or ip.is_loopback) or ip.is_multicast or ip.is_unspecified:
        sys.exit(f"refusing target {host}: only private or loopback IPv4 addresses are allowed")
    return str(ip)


def pace(rate, duration):
    """Yield event numbers at about `rate` per second for `duration` seconds (catches up in bursts if late)."""
    start = time.perf_counter()
    n = 0
    while time.perf_counter() - start < duration:
        delay = start + n / rate - time.perf_counter()
        if delay > 0.002:
            time.sleep(delay)
        yield n
        n += 1


def tcp_attempt(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(False)
    try:
        s.connect_ex((ip, port))                    # the SYN is sent immediately; we do not wait for the answer
    finally:
        s.close()


def run_mode(mode, ip, port, rate, duration, scan_ports=10000):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = bytes(1000)
    sent = 0
    for n in pace(rate, duration):
        try:
            if mode == "syn":
                tcp_attempt(ip, port)
            elif mode == "udp":
                udp.sendto(payload, (ip, port))
            elif mode == "scan":
                tcp_attempt(ip, 1 + n % scan_ports)
            else:                                   # benign: mostly small UDP, sometimes a TCP attempt
                if random.random() < 0.9:
                    udp.sendto(bytes(random.randrange(50, 400)), (ip, random.choice([5683, 1883, 5353])))
                else:
                    tcp_attempt(ip, random.choice([80, 443, 8883]))
            sent += 1
        except OSError:
            pass                                    # e.g. transient buffer errors: just carry on
    udp.close()
    return sent


def send_marker(ip, base, idx, repeats=3):
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(repeats):
        u.sendto(b"seg:" + LABELS[idx].encode(), (ip, base + idx))
        time.sleep(0.05)
    u.close()


def parse_plan(text):
    plan, total = [], 0
    for part in text.split(","):
        label, dur = part.rsplit(":", 1)
        label, dur = label.strip(), float(dur)
        if label not in LABELS:
            sys.exit(f"unknown label '{label}' in plan; use one of {LABELS}")
        if not 0 < dur <= MAX_SEGMENT_S:
            sys.exit(f"segment '{part}': duration must be 1..{MAX_SEGMENT_S} seconds")
        plan.append((label, dur)); total += dur
    if total > MAX_PLAN_S:
        sys.exit(f"plan too long ({total:.0f}s > {MAX_PLAN_S}s)")
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["benign", "syn", "udp", "scan", "session"])
    ap.add_argument("--target", required=True, help="private/loopback IPv4 address of the device under test")
    ap.add_argument("--i-own-this-network", action="store_true", required=True, help="confirm you are authorised to test here")
    ap.add_argument("--duration", type=float, default=10, help="seconds (single modes)")
    ap.add_argument("--rate", type=float, default=None, help=f"events per second (max {MAX_RATE}); default depends on the mode")
    ap.add_argument("--port", type=int, default=None, help="destination port for syn/udp (default 80 / 5001)")
    ap.add_argument("--plan", default=None, help="session timeline, e.g. 'benign:60,syn_flood:30,...'")
    ap.add_argument("--marker-base", type=int, default=54300)
    ap.add_argument("--no-markers", action="store_true")
    a = ap.parse_args()
    ip = check_target(a.target)

    def rate_for(mode):
        r = a.rate if a.rate else DEFAULT_RATE[mode]
        if not 0 < r <= MAX_RATE:
            sys.exit(f"rate must be between 0 and {MAX_RATE} per second")
        return r

    try:
        if a.mode == "session":
            if not a.plan:
                sys.exit("session needs --plan")
            plan = parse_plan(a.plan)
            print(f"session of {sum(d for _, d in plan):.0f} s against {ip}: " + " > ".join(f"{l}({d:.0f}s)" for l, d in plan), flush=True)
            for label, dur in plan:
                if not a.no_markers:
                    send_marker(ip, a.marker_base, LABELS.index(label))
                mode = MODE_OF[label]
                t0 = time.time()
                n = run_mode(mode, ip, a.port or (5001 if mode == "udp" else 80), rate_for(mode), dur)
                print(f"  {label:<10s} {dur:>5.0f}s  {n:>7,} events  ({n / max(time.time() - t0, 1e-9):,.0f}/s)", flush=True)
        else:
            r = rate_for(a.mode)
            dur = min(a.duration, MAX_SEGMENT_S)
            port = a.port or (5001 if a.mode == "udp" else 80)
            print(f"{a.mode} -> {ip}:{port} at <= {r:g}/s for {dur:g}s (Ctrl+C stops)", flush=True)
            t0 = time.time()
            n = run_mode(a.mode, ip, port, r, dur)
            print(f"done: {n:,} events in {time.time() - t0:.1f}s ({n / max(time.time() - t0, 1e-9):,.0f}/s)")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()

"""
Dependency-free packet capture + header parsing (Python standard library only).

Why this exists: scapy fully dissects every packet into Python objects, which cost about 1 ms
per packet in our benchmark - far more than the whole machine-learning part (feature extractor
~0.12 ms, forest prediction ~0.007 ms) - and would swamp a Raspberry Pi during a flood. Here only
the handful of header fields the extractor needs are decoded with `struct`.

  parse_frame(buf, ts, wire_len)  Ethernet frame -> (ts, size, src_mac, dst_mac, src_ip, dst_ip,
                                  src_proto, dst_proto)   (same tuple packet_features.parse_packet returns)
  iter_pcap(path)                 classic pcap files (micro/nanosecond, either endianness, Ethernet)
  iter_live(iface, duration)      raw AF_PACKET socket (Linux only, needs root / CAP_NET_RAW)

Supported: Ethernet II (+ one 802.1Q VLAN tag), IPv4, IPv6 without extension headers, TCP, UDP,
ICMP, ARP. Anything else returns None. pcapng and non-Ethernet link types are not supported
(live_detector falls back to scapy for those; convert with `tcpdump -r in.pcapng -w out.pcap`).
"""
import socket
import struct
import time

_ETH = struct.Struct("!6s6sH")
_CACHE_MAX = 50000                 # spoofed-source floods must not grow the caches without bound
_mac_cache, _ip4_cache, _ip6_cache = {}, {}, {}


def _mac(b):
    s = _mac_cache.get(b)
    if s is None:
        s = ":".join(f"{x:02x}" for x in b)
        if len(_mac_cache) < _CACHE_MAX:
            _mac_cache[b] = s
    return s


def _ip4(b):
    s = _ip4_cache.get(b)
    if s is None:
        s = socket.inet_ntoa(b)
        if len(_ip4_cache) < _CACHE_MAX:
            _ip4_cache[b] = s
    return s


def _ip6(b):
    s = _ip6_cache.get(b)
    if s is None:
        s = socket.inet_ntop(socket.AF_INET6, b)
        if len(_ip6_cache) < _CACHE_MAX:
            _ip6_cache[b] = s
    return s


def parse_frame(buf, ts, wire_len=None):
    n = len(buf)
    if n < 14:
        return None
    dst, src, ethertype = _ETH.unpack_from(buf, 0)
    off = 14
    if ethertype == 0x8100 and n >= 18:                     # 802.1Q VLAN tag
        ethertype = (buf[16] << 8) | buf[17]
        off = 18
    size = float(wire_len if wire_len else n)
    smac, dmac = _mac(src), _mac(dst)

    if ethertype == 0x0800:                                  # IPv4
        if n < off + 20:
            return None
        ihl = (buf[off] & 0x0F) * 4
        proto = buf[off + 9]
        sip, dip = _ip4(buf[off + 12:off + 16]), _ip4(buf[off + 16:off + 20])
        l4 = off + ihl
        frag_offset = ((buf[off + 6] & 0x1F) << 8) | buf[off + 7]
        if (proto == 6 or proto == 17) and frag_offset == 0 and n >= l4 + 4:
            sp = str((buf[l4] << 8) | buf[l4 + 1])
            dp = str((buf[l4 + 2] << 8) | buf[l4 + 3])
        elif proto == 1:
            sp = dp = "icmp"
        else:
            sp = dp = str(proto)
        return ts, size, smac, dmac, sip, dip, sp, dp

    if ethertype == 0x86DD:                                  # IPv6 (extension headers not followed)
        if n < off + 40:
            return None
        nh = buf[off + 6]
        sip, dip = _ip6(buf[off + 8:off + 24]), _ip6(buf[off + 24:off + 40])
        l4 = off + 40
        if (nh == 6 or nh == 17) and n >= l4 + 4:
            sp = str((buf[l4] << 8) | buf[l4 + 1])
            dp = str((buf[l4 + 2] << 8) | buf[l4 + 3])
        elif nh == 58:
            sp = dp = "icmp"
        else:
            sp = dp = str(nh)
        return ts, size, smac, dmac, sip, dip, sp, dp

    if ethertype == 0x0806:                                  # ARP
        if n < off + 28:
            return None
        return ts, size, smac, dmac, _ip4(buf[off + 14:off + 18]), _ip4(buf[off + 24:off + 28]), "arp", "arp"

    return None


def iter_pcap(path):
    """Yield (timestamp, frame_bytes, wire_length) from a classic pcap file."""
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) < 24:
            raise ValueError("file too short to be a pcap")
        magic = head[:4]
        if magic == b"\xd4\xc3\xb2\xa1":
            end, div = "<", 1e6
        elif magic == b"\xa1\xb2\xc3\xd4":
            end, div = ">", 1e6
        elif magic == b"\x4d\x3c\xb2\xa1":
            end, div = "<", 1e9
        elif magic == b"\xa1\xb2\x3c\x4d":
            end, div = ">", 1e9
        else:
            raise ValueError("not a classic pcap file (pcapng?)")
        linktype = struct.unpack(end + "I", head[20:24])[0]
        if linktype != 1:
            raise ValueError(f"unsupported link type {linktype} (only Ethernet = 1)")
        rec = struct.Struct(end + "IIII")
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            sec, frac, incl, orig = rec.unpack(h)
            data = f.read(incl)
            if len(data) < incl:
                return
            yield sec + frac / div, data, orig


def iter_live(iface, duration=None):
    """Yield (timestamp, frame_bytes, wire_length) from a raw socket on `iface` until `duration` s pass."""
    if not hasattr(socket, "AF_PACKET"):
        raise OSError("raw capture needs Linux (AF_PACKET); use --engine scapy on this system")
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))   # ETH_P_ALL
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)       # fewer drops in bursts
        sock.bind((iface, 0))
        sock.settimeout(0.5)                                                         # lets Ctrl+C / duration be checked
        end = time.time() + duration if duration else None
        while end is None or time.time() < end:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            yield time.time(), data, len(data)
    finally:
        sock.close()

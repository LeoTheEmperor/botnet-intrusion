"""
Turn a scapy packet into the tuple the FeatureExtractor expects:
    (timestamp, frame_size, src_mac, dst_mac, src_ip, dst_ip, src_proto, dst_proto)
src_proto/dst_proto are the TCP/UDP port as a string, 'icmp', the IP protocol number, or 'arp'.
Returns None for frames we cannot describe (e.g. non-IP, non-ARP).
"""
from scapy.all import Ether, IP, IPv6, TCP, UDP, ICMP, ARP


def parse_packet(pkt):
    ts = float(pkt.time)
    size = float(len(pkt))
    eth = pkt.getlayer(Ether)
    src_mac = eth.src if eth is not None else ""
    dst_mac = eth.dst if eth is not None else ""

    if pkt.haslayer(IP):
        ip = pkt[IP]
        src_ip, dst_ip, proto_no = ip.src, ip.dst, str(ip.proto)
    elif pkt.haslayer(IPv6):
        ip = pkt[IPv6]
        src_ip, dst_ip, proto_no = ip.src, ip.dst, str(ip.nh)
    elif pkt.haslayer(ARP):
        a = pkt[ARP]
        return ts, size, src_mac or a.hwsrc, dst_mac or a.hwdst, a.psrc, a.pdst, "arp", "arp"
    else:
        return None

    if pkt.haslayer(TCP):
        l4 = pkt[TCP]
        sp, dp = str(l4.sport), str(l4.dport)
    elif pkt.haslayer(UDP):
        l4 = pkt[UDP]
        sp, dp = str(l4.sport), str(l4.dport)
    elif pkt.haslayer(ICMP):
        sp = dp = "icmp"
    else:
        sp = dp = proto_no
    return ts, size, src_mac, dst_mac, src_ip, dst_ip, sp, dp

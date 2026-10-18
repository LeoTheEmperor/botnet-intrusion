"""
Streaming 115-feature extractor in the N-BaIoT column layout:
  blocks  MI_dir | H | HH | HH_jit | HpHp ,  inside each block: lambda L5,L3,L1,L0.1,L0.01,
  inside each lambda: the block's statistics.

  MI_dir  src MAC+IP stream            -> weight, mean, variance
  H       src IP stream                -> weight, mean, variance
  HH      src->dst IP channel (+ reverse direction as partner)
                                       -> weight, mean, std, magnitude, radius, covariance, pcc
  HH_jit  inter-arrival times on the src->dst channel -> weight, mean, variance
  HpHp    src ip:port -> dst ip:port socket (+ reverse) -> same 7 stats as HH

Feature keying of HH/HpHp (per directional channel/socket) is inferred from the dataset
(HH_weight <= H_weight, equal on ~21% of rows); see validate_extractor.py.
"""
import numpy as np
from afterimage import StatDB

LAMBDAS = (5, 3, 1, 0.1, 0.01)
LNAMES = ("L5", "L3", "L1", "L0.1", "L0.01")
STATS_1D = ("weight", "mean", "variance")
STATS_2D = ("weight", "mean", "std", "magnitude", "radius", "covariance", "pcc")
BLOCKS = (("MI_dir", STATS_1D), ("H", STATS_1D), ("HH", STATS_2D),
          ("HH_jit", STATS_1D), ("HpHp", STATS_2D))

FEATURE_NAMES = [f"{b}_{ln}_{s}" for b, stats in BLOCKS for ln in LNAMES for s in stats]
assert len(FEATURE_NAMES) == 115


class FeatureExtractor:
    def __init__(self, prune_every=5000):
        self.db = {name: StatDB() for name, _ in BLOCKS}
        self._n = 0
        self._prune_every = prune_every

    def update(self, ts, size, src_mac, dst_mac, src_ip, dst_ip, src_proto, dst_proto):
        """Process one packet, return the 115-dim feature vector (np.ndarray).
        size = frame length in bytes; src_proto/dst_proto = port (str) or 'arp'."""
        out = np.empty(115)
        pos = 0
        mi, h, hh, jit, hphp = (self.db[n] for n, _ in BLOCKS)
        mi_id, h_id = src_mac + src_ip, src_ip
        hh_a, hh_b = f"{src_ip}>{dst_ip}", f"{dst_ip}>{src_ip}"
        jit_id = src_ip + dst_ip
        if src_proto == "arp":
            sk_a, sk_b = src_mac, dst_mac
        else:
            sk_a, sk_b = f"{src_ip}:{src_proto}>{dst_ip}:{dst_proto}", f"{dst_ip}:{dst_proto}>{src_ip}:{src_proto}"

        blocks_out = {name: [] for name, _ in BLOCKS}   # collect per block, then lay out in CSV order
        for lam in LAMBDAS:
            blocks_out["MI_dir"].extend(mi.update_1d(mi_id, ts, size, lam))
            blocks_out["H"].extend(h.update_1d(h_id, ts, size, lam))
            blocks_out["HH"].extend(hh.update_1d2d(hh_a, hh_b, ts, size, lam))
            blocks_out["HH_jit"].extend(jit.update_1d(jit_id, ts, 0.0, lam, is_diff=True))
            blocks_out["HpHp"].extend(hphp.update_1d2d(sk_a, sk_b, ts, size, lam))
        for name, _ in BLOCKS:
            vals = blocks_out[name]
            out[pos:pos + len(vals)] = vals
            pos += len(vals)

        self._n += 1
        if self._n % self._prune_every == 0:
            for d in self.db.values():
                d.prune(ts)
        return out

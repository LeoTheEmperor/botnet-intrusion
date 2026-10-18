"""
Validation of our AfterImage port + 115-feature extractor.
  1. Numerical equivalence vs the authors' reference implementation (random packet stream)
  2. Feature names / order == N-BaIoT CSV columns
  3. Semantic checks on the real dataset (magnitude/radius labels, HH vs H streams)
  4. Throughput benchmark
"""
import importlib.util, math, time
import numpy as np
import pandas as pd
from pathlib import Path
from extractor import FeatureExtractor, FEATURE_NAMES, LAMBDAS, BLOCKS

BASE = Path(__file__).resolve().parents[1]
csv = pd.read_csv(BASE / "data" / "danmini_labeled_sample.csv")

# ---------- 2. names/order ----------
cols = csv.columns[:115].tolist()
print("[2] feature names/order identical to CSV columns:", cols == FEATURE_NAMES)

# ---------- synthetic packet stream ----------
rng = np.random.default_rng(7)
hosts = [("aa:00:00:00:00:0%d" % i, "10.0.0.%d" % (i + 1)) for i in range(5)]
def make_packets(n):
    t, pk = 1000.0, []
    for _ in range(n):
        r = rng.random()
        t += 0.0 if r < 0.10 else (rng.exponential(0.05) if r < 0.97 else rng.uniform(5, 30))
        if rng.random() < 0.01:
            t -= rng.uniform(0, 0.2)                       # occasional out-of-order timestamp
        a, b = rng.choice(len(hosts), 2, replace=False)
        size = float(rng.integers(60, 1500))
        sp, dp = str(rng.choice([80, 443, 5000, 23])), str(rng.choice([80, 443, 5000, 23]))
        pk.append((t, size, hosts[a][0], hosts[b][0], hosts[a][1], hosts[b][1], sp, dp))
    return pk
packets = make_packets(4000)

# ---------- 1. equivalence vs reference ----------
spec = importlib.util.spec_from_file_location("af_ref", BASE / "reference" / "AfterImage_ref.py")
af = importlib.util.module_from_spec(spec); spec.loader.exec_module(af)
ref = {n: af.incStatDB(limit=10**9) for n, _ in BLOCKS}

def ref_vector(p):
    ts, size, smac, dmac, sip, dip, sp, dp = p
    parts = {n: [] for n, _ in BLOCKS}
    for lam in LAMBDAS:
        parts["MI_dir"] += ref["MI_dir"].update_get_1D_Stats(smac + sip, ts, size, lam)
        parts["H"] += ref["H"].update_get_1D_Stats(sip, ts, size, lam)
        w, m, var, rad, mag, cov, pcc = ref["HH"].update_get_1D2D_Stats(f"{sip}>{dip}", f"{dip}>{sip}", ts, size, lam)
        parts["HH"] += [w, m, math.sqrt(var), mag, rad, cov, pcc]
        parts["HH_jit"] += ref["HH_jit"].update_get_1D_Stats(sip + dip, ts, 0, lam, isTypeDiff=True)
        w, m, var, rad, mag, cov, pcc = ref["HpHp"].update_get_1D2D_Stats(
            f"{sip}:{sp}>{dip}:{dp}", f"{dip}:{dp}>{sip}:{sp}", ts, size, lam)
        parts["HpHp"] += [w, m, math.sqrt(var), mag, rad, cov, pcc]
    return np.concatenate([parts[n] for n, _ in BLOCKS])

fx = FeatureExtractor(prune_every=10**9)    # pruning off for exact comparison
worst = 0.0
for p in packets:
    mine, theirs = fx.update(*p), ref_vector(p)
    worst = max(worst, float(np.max(np.abs(mine - theirs) / np.maximum(1.0, np.abs(theirs)))))
print(f"[1] max relative deviation vs reference over {len(packets)} packets x 115 features: {worst:.3e}")

# ---------- 3. dataset semantics ----------
d = csv
for L in ("L5", "L0.01"):
    mag_eq_mean = np.isclose(d[f"HH_{L}_magnitude"], d[f"HH_{L}_mean"].abs(), rtol=1e-6).mean()
    rad_eq_var = np.isclose(d[f"HH_{L}_radius"], d[f"HH_{L}_std"] ** 2, rtol=1e-6).mean()
    swapped = np.isclose(d[f"HH_{L}_radius"], d[f"HH_{L}_mean"].abs(), rtol=1e-6).mean()
    print(f"[3] {L}: magnitude==|mean| on {mag_eq_mean:.1%} of rows; radius==std^2 on {rad_eq_var:.1%}; "
          f"(swapped-label check radius==|mean|: {swapped:.1%})")
print(f"[3] HH_weight <= H_weight on {(d['HH_L5_weight'] <= d['H_L5_weight'] + 1e-9).mean():.1%} of rows")

# ---------- 4. benchmark ----------
bench = make_packets(20000)
fx2 = FeatureExtractor()
t0 = time.time()
for p in bench:
    fx2.update(*p)
dt = time.time() - t0
print(f"[4] throughput on this sandbox: {len(bench)/dt:,.0f} packets/s  ({dt/len(bench)*1e6:.0f} us/packet)")

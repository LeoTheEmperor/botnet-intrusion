"""
Damped incremental statistics (the "AfterImage" engine from Kitsune, Mirsky et al.,
NDSS 2018) - our own implementation, validated numerically against the authors'
reference code (see validate_extractor.py).

Each stream keeps exponentially-decayed sums:  w (weight), CF1 (sum), CF2 (sum of squares)
Decay before each insertion:  gamma = 2^(-lambda * dt)   (lambda in 1/seconds)
  mean = CF1/w          var = |CF2/w - mean^2|          std = sqrt(var)
Two streams can be linked by an IncStatCov, which keeps a decayed sum of residual
products (CF3, w3) -> approximate covariance cov = CF3/w3 and pcc = cov/(std_i*std_j).
A packet updates ONLY its own direction's stream; the partner stream is merely decayed
to the same time and its last residual is reused (this is what makes it streaming-safe).

Numerical fidelity: like the reference, mean/var/std are CACHED and invalidated only when a
new value is inserted. Decay scales CF1, CF2 and w together so mean/var are mathematically
unchanged, but recomputing var = |CF2/w - mean^2| after decay adds floating-point noise that
blows up pcc for near-constant streams (the dataset shows pcc == 0 for 75-99% of rows, i.e. the
exact-zero behaviour of the cached version).

Engineering additions for long-running use (do not change results in normal operation):
  * weights are floored at 1e-300 to avoid ZeroDivisionError after a very long idle gap
  * StatDB.prune() removes dead streams so memory stays bounded on a Raspberry Pi
"""
import math

_W_FLOOR = 1e-300


class IncStat:
    """One damped 1-D stream (e.g. packet sizes sent by one host)."""
    __slots__ = ("id", "lam", "cf1", "cf2", "w", "is_diff", "last_t", "covs",
                 "_mean", "_var", "_std")

    def __init__(self, lam, stream_id, init_time=0.0, is_diff=False):
        self.id = stream_id
        self.lam = lam
        self.cf1 = 0.0
        self.cf2 = 0.0
        self.w = 1e-20            # tiny non-zero weight avoids divide-by-zero on first read
        self.is_diff = is_diff    # jitter stream: the inserted value is the inter-arrival time
        self.last_t = init_time
        self.covs = []            # IncStatCov trackers this stream participates in
        self._mean = self._var = self._std = None   # caches, invalidated by insert()

    def decay(self, t):
        dt = t - self.last_t
        if dt > 0:                # out-of-order / identical timestamps do not decay
            f = 2.0 ** (-self.lam * dt)
            self.cf1 *= f
            self.cf2 *= f
            self.w = max(self.w * f, _W_FLOOR)
            self.last_t = t       # (cached mean/var/std stay valid: decay scales all sums equally)

    def insert(self, v, t):
        if self.is_diff:
            d = t - self.last_t   # time since this stream's previous packet
            v = d if d > 0 else 0.0
        self.decay(t)
        self.cf1 += v
        self.cf2 += v * v
        self.w += 1.0
        self._mean = self._var = self._std = None   # force recomputation
        for c in self.covs:       # keep linked covariance trackers in sync
            c.update(self.id, v, t)

    def mean(self):
        if self._mean is None:
            self._mean = self.cf1 / self.w
        return self._mean

    def var(self):
        if self._var is None:
            self._var = abs(self.cf2 / self.w - self.mean() ** 2)
        return self._var

    def std(self):
        if self._std is None:
            self._std = math.sqrt(self.var())
        return self._std

    def stats_1d(self):
        """[weight, mean, variance] - refreshes mean/var caches like the reference's allstats_1D()."""
        self._mean = self.cf1 / self.w
        self._var = abs(self.cf2 / self.w - self._mean ** 2)
        return self.w, self._mean, self._var


class IncStatCov:
    """Links two streams; tracks decayed sum of residual products (approx. covariance)."""
    __slots__ = ("s", "last_res", "cf3", "w3", "last_t")

    def __init__(self, s0, s1, init_time=0.0):
        self.s = (s0, s1)
        self.last_res = [0.0, 0.0]
        self.cf3 = 0.0
        self.w3 = 1e-20
        self.last_t = init_time

    def update(self, stream_id, v, t):
        """Call AFTER stream `stream_id` has already absorbed (v, t)."""
        if stream_id == self.s[0].id:
            k = 0
        elif stream_id == self.s[1].id:
            k = 1
        else:
            raise ValueError("stream id not part of this covariance tracker")
        other = 1 - k
        self.s[other].decay(t)            # bring partner stream to the current time
        self._decay(t, k)
        res = v - self.s[k].mean()        # residual of this packet vs its stream mean
        self.cf3 += res * self.last_res[other]
        self.w3 += 1.0
        self.last_res[k] = res

    def _decay(self, t, k):
        dt = t - self.last_t
        if dt > 0:
            f = 2.0 ** (-self.s[k].lam * dt)
            self.cf3 *= f
            self.w3 = max(self.w3 * f, _W_FLOOR)
            self.last_t = t
            self.last_res[k] *= f

    def cov(self):
        return self.cf3 / self.w3

    def pcc(self):
        ss = self.s[0].std() * self.s[1].std()
        return self.cov() / ss if ss != 0 else 0.0


class StatDB:
    """Hash table of streams keyed by (stream_id, lambda)."""

    def __init__(self):
        self.ht = {}

    def _get(self, sid, lam, t, is_diff=False):
        key = (sid, lam)
        s = self.ht.get(key)
        if s is None:
            s = self.ht[key] = IncStat(lam, sid, t, is_diff)
        return s

    def update_1d(self, sid, t, v, lam, is_diff=False):
        """Insert v, return [weight, mean, variance]."""
        s = self._get(sid, lam, t, is_diff)
        s.insert(v, t)
        return s.stats_1d()

    def update_1d2d(self, id1, id2, t, v, lam):
        """Insert v into stream id1 (partner id2 = opposite direction); return
        [weight, mean, std, magnitude, radius, covariance, pcc]."""
        s1 = self._get(id1, lam, t)
        s1.insert(v, t)
        s2 = self._get(id2, lam, t)
        cov = None
        for c in s1.covs:
            if c.s[0].id == id2 or c.s[1].id == id2:
                cov = c
                break
        if cov is None:
            cov = IncStatCov(s1, s2, t)
            s1.covs.append(cov)
            s2.covs.append(cov)
        cov.update(id1, v, t)   # (reference quirk kept: tracker is updated here as well)
        w1, m1, v1 = s1.stats_1d()                      # refresh s1's caches (as reference does)
        magnitude = math.sqrt(s1.mean() ** 2 + s2.mean() ** 2)
        radius = math.sqrt(s1.var() ** 2 + s2.var() ** 2)
        return w1, m1, math.sqrt(v1), magnitude, radius, cov.cov(), cov.pcc()

    def prune(self, now, min_weight=1e-3):
        """Drop streams whose decayed weight (at time `now`) is negligible."""
        dead = [k for k, s in self.ht.items()
                if s.w * 2.0 ** (-s.lam * max(0.0, now - s.last_t)) < min_weight]
        for k in dead:
            s = self.ht.pop(k)
            for c in s.covs:                       # unlink from partner streams
                for o in c.s:
                    if o is not s and c in o.covs:
                        o.covs.remove(c)
        return len(dead)

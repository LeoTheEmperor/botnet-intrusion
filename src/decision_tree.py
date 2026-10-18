"""
From-scratch CART Decision Tree Classifier (Gini impurity, binary numeric splits).
Only numpy is used. Split search is vectorised: samples are sorted once per
feature and class counts for every candidate threshold come from a cumulative sum.
"""
import numpy as np


class TreeNode:
    __slots__ = ["feature_idx", "threshold", "left", "right", "prediction", "is_leaf"]

    def __init__(self):
        self.feature_idx = self.threshold = self.left = self.right = self.prediction = None
        self.is_leaf = False


def gini_impurity(y):
    """Gini = 1 - sum(p_c^2)."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


class DecisionTreeClassifier:
    """max_features: None | int | float | "sqrt" -> features tried per split
    (the source of extra randomness when used inside a Random Forest)."""

    def __init__(self, max_depth=10, min_samples_split=10, min_samples_leaf=5,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.classes_ = None
        self._rng = np.random.default_rng(random_state)

    def _n_features_to_use(self, n):
        if self.max_features is None:
            return n
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n))
        return min(self.max_features, n)

    def _best_split(self, X, y, feature_indices):
        """Return (feature, threshold, gini) minimising weighted Gini, or None.
        Candidate thresholds: midpoints if <=32 unique values, else 32 quantiles."""
        best_gini, best_feat, best_thr = np.inf, None, None
        n = X.shape[0]
        _, y_int = np.unique(y, return_inverse=True)
        n_classes = y_int.max() + 1

        for f in feature_indices:
            values = X[:, f]
            uniq = np.unique(values)
            if len(uniq) <= 1:
                continue
            if len(uniq) > 32:
                cands = np.unique(np.quantile(values, np.linspace(0, 1, 33)[1:-1]))
            else:
                cands = (uniq[:-1] + uniq[1:]) / 2.0
            if len(cands) == 0:
                continue

            order = np.argsort(values)
            sorted_vals, sorted_y = values[order], y_int[order]
            one_hot = np.zeros((n, n_classes), dtype=np.int64)
            one_hot[np.arange(n), sorted_y] = 1
            cum = np.cumsum(one_hot, axis=0)          # left-side class counts
            total = cum[-1]
            positions = np.searchsorted(sorted_vals, cands, side="right")

            for pos, thr in zip(positions, cands):
                n_l, n_r = pos, n - pos
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                lc = cum[pos - 1]
                rc = total - lc
                g = (n_l / n) * (1 - np.sum((lc / n_l) ** 2)) + (n_r / n) * (1 - np.sum((rc / n_r) ** 2))
                if g < best_gini:
                    best_gini, best_feat, best_thr = g, f, thr

        return None if best_feat is None else (best_feat, best_thr, best_gini)

    @staticmethod
    def _majority(y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _build(self, X, y, depth):
        node = TreeNode()
        n, n_feat = X.shape
        if (len(np.unique(y)) == 1 or n < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            node.is_leaf, node.prediction = True, self._majority(y)
            return node

        feats = self._rng.choice(n_feat, size=self._n_features_to_use(n_feat), replace=False)
        split = self._best_split(X, y, feats)
        if split is None:
            node.is_leaf, node.prediction = True, self._majority(y)
            return node

        node.feature_idx, node.threshold, _ = split
        mask = X[:, node.feature_idx] <= node.threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.root = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.prediction

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xd = rng.normal(size=(200, 4))
    yd = (Xd[:, 0] + Xd[:, 1] > 0).astype(int)
    t = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xd, yd)
    print(f"Smoke test train accuracy: {(t.predict(Xd) == yd).mean():.3f} (expect > 0.9)")

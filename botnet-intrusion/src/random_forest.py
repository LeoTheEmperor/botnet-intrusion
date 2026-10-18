"""
From-scratch Random Forest: bootstrap aggregating (bagging) over rows +
per-split feature subsampling (inside DecisionTreeClassifier), majority vote.
Uses only numpy and our own DecisionTreeClassifier.
"""
import numpy as np
from collections import Counter
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=10,
                 min_samples_leaf=5, max_features="sqrt", bootstrap_frac=1.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap_frac = bootstrap_frac
        self.random_state = random_state
        self.trees = []
        self.classes_ = None

    def fit(self, X, y, verbose=True):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        n_boot = int(round(n * self.bootstrap_frac))
        self.classes_ = np.unique(y)
        self.trees = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n_boot)              # sample WITH replacement
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)))
            self.trees.append(tree.fit(X[idx], y[idx]))
            if verbose and (i + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"  Trained tree {i + 1}/{self.n_estimators}")
        return self

    def predict(self, X):
        votes = np.array([t.predict(X) for t in self.trees])    # (n_trees, n_samples)
        return np.array([Counter(votes[:, c]).most_common(1)[0][0] for c in range(votes.shape[1])])

    def predict_proba(self, X):
        """Fraction of trees voting for each class, columns ordered by classes_."""
        votes = np.array([t.predict(X) for t in self.trees])
        idx = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for c in range(X.shape[0]):
            for cls, k in Counter(votes[:, c]).items():
                proba[c, idx[cls]] = k / self.n_estimators
        return proba


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xd = rng.normal(size=(300, 5))
    yd = (Xd[:, 0] + Xd[:, 1] - Xd[:, 2] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=0).fit(Xd, yd, verbose=False)
    print(f"Smoke test train accuracy: {(rf.predict(Xd) == yd).mean():.3f}")

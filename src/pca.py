"""
Principal Component Analysis from scratch (eigendecomposition of the covariance
matrix with numpy; no sklearn.decomposition). Fit on TRAIN only, then applied to val/test.
"""
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = self.components_ = None
        self.explained_variance_ = self.explained_variance_ratio_ = None
        self.all_eigvals_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X - self.mean_, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)          # symmetric matrix -> eigh
        order = np.argsort(eigvals)[::-1]                # largest variance first
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        self.components_ = eigvecs[:, :self.n_components].T
        self.explained_variance_ = eigvals[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigvals.sum()
        self.all_eigvals_ = eigvals
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    b = rng.normal(size=(500, 1))
    X = np.hstack([3 * b + rng.normal(scale=.1, size=(500, 1)),
                   -2 * b + rng.normal(scale=.1, size=(500, 1)),
                   rng.normal(scale=.1, size=(500, 1))])
    print("Explained variance ratio:", PCA(2).fit(X).explained_variance_ratio_, "(PC1 expected > 0.9)")

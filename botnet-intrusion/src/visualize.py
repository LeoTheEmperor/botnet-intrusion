"""
Dataset visualisations required by the assignment (saved to ./figures):
class distribution, feature-correlation heatmap, PCA 2-D scatter, PCA scree plot.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pca import PCA

BASE = Path(__file__).resolve().parents[1]
DATA, FIG = BASE / "data", BASE / "figures"
FIG.mkdir(exist_ok=True)

raw = pd.read_csv(DATA / "danmini_labeled_sample.csv")
X_train = np.load(DATA / "X_train.npy")
y_train = pd.read_csv(DATA / "y_train.csv")["label"].to_numpy()
feature_cols = raw.columns[:115].tolist()
CLASSES = sorted(np.unique(y_train))
COLOR = dict(zip(CLASSES, plt.cm.tab20(np.linspace(0, 1, len(CLASSES)))))


def class_distribution():
    counts = raw["label"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=[COLOR[c] for c in counts.index])
    ax.set(title="Class Distribution - Danmini Doorbell (balanced sample)", ylabel="Samples", xlabel="Class")
    plt.xticks(rotation=45, ha="right")
    for b in bars:
        ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "class_distribution.png", dpi=150); plt.close()


def correlation_heatmap(n_features=30):
    idx = np.linspace(0, len(feature_cols) - 1, n_features).astype(int)   # spread over all blocks
    cols = [feature_cols[i] for i in idx]
    corr = raw[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n_features)); ax.set_yticks(range(n_features))
    ax.set_xticklabels(cols, rotation=90, fontsize=6); ax.set_yticklabels(cols, fontsize=6)
    ax.set_title(f"Feature Correlation Heatmap ({n_features}-feature sample of 115)")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    plt.tight_layout(); plt.savefig(FIG / "correlation_heatmap.png", dpi=150); plt.close()


def pca_scatter():
    pca = PCA(2); X2 = pca.fit_transform(X_train)
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in CLASSES:
        m = y_train == c
        ax.scatter(X2[m, 0], X2[m, 1], s=6, alpha=.5, color=COLOR[c], label=c)
    r = pca.explained_variance_ratio_
    ax.set(title=f"PCA Projection of Training Data ({r.sum()*100:.1f}% variance in 2 components)",
           xlabel=f"PC1 ({r[0]*100:.1f}%)", ylabel=f"PC2 ({r[1]*100:.1f}%)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=3)
    plt.tight_layout(); plt.savefig(FIG / "pca_scatter.png", dpi=150); plt.close()


def scree(k=50):
    pca = PCA(k).fit(X_train)
    cum = np.cumsum(pca.all_eigvals_[:k]) / pca.all_eigvals_.sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, k + 1), cum, marker="o", markersize=3)
    ax.axhline(.95, color="red", ls="--", lw=1, label="95% variance")
    ax.set(xlabel="Principal components", ylabel="Cumulative explained variance",
           title="PCA Scree Plot"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "pca_scree.png", dpi=150); plt.close()


if __name__ == "__main__":
    class_distribution(); correlation_heatmap(); pca_scatter(); scree()
    print(f"Figures saved to {FIG}")

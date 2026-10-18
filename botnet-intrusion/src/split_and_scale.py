"""
Stratified train / validation / test split + z-score scaling (all from scratch).

Split: 80% train+val / 20% test, then 80/20 of the 80% -> 64% / 16% / 20%.
Scaling: mean/std fitted on TRAIN only, then applied to val and test (no leakage).
No sklearn.model_selection / sklearn.preprocessing is used.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
N_FEATURES = 115
LABEL_COL = "label"
RANDOM_STATE = 42


def stratified_split(df, label_col, test_frac, random_state=42):
    """Per class: shuffle indices, send test_frac to the right split, rest to left."""
    rng = np.random.default_rng(random_state)
    left, right = [], []
    for _, group in df.groupby(label_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_right = int(round(len(idx) * test_frac))
        right.append(df.loc[idx[:n_right]])
        left.append(df.loc[idx[n_right:]])
    shuffle = lambda parts: pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return shuffle(left), shuffle(right)


def zscore_fit(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0  # guard constant columns
    return mean, std


def zscore_apply(X, mean, std):
    return (X - mean) / std


def main():
    df = pd.read_csv(DATA / "danmini_labeled_sample.csv")
    feats = df.columns[:N_FEATURES].tolist()

    train_val, test = stratified_split(df, LABEL_COL, 0.20, RANDOM_STATE)
    train, val = stratified_split(train_val, LABEL_COL, 0.20, RANDOM_STATE)
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name}: {len(part):,} rows ({len(part)/len(df):.1%})")

    X_tr, X_va, X_te = (p[feats].to_numpy(dtype=np.float64) for p in (train, val, test))
    mean, std = zscore_fit(X_tr)
    print(f"train mean after scaling: {zscore_apply(X_tr, mean, std).mean():.6f}")

    np.save(DATA / "X_train.npy", zscore_apply(X_tr, mean, std))
    np.save(DATA / "X_val.npy", zscore_apply(X_va, mean, std))
    np.save(DATA / "X_test.npy", zscore_apply(X_te, mean, std))
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[[LABEL_COL]].to_csv(DATA / f"y_{name}.csv", index=False)
    np.save(DATA / "scaler_mean.npy", mean)   # needed again on the Pi
    np.save(DATA / "scaler_std.npy", std)
    print("Saved arrays, labels and scaler params.")


if __name__ == "__main__":
    main()

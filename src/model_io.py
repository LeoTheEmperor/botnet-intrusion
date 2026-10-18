"""
Export a trained from-scratch Random Forest (+ scaler) to a single .npz file and load it
back as a lightweight predictor that needs ONLY numpy (ideal for the Raspberry Pi: no pickle,
no training code). Trees are flattened into parallel arrays; leaves have cls >= 0.
"""
import numpy as np


def _flatten(root, class_index, offset):
    feat, thr, left, right, cls = [], [], [], [], []

    def add(node):
        i = len(feat)
        feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); cls.append(-1)
        if node.is_leaf:
            cls[i] = class_index[node.prediction]
        else:
            feat[i] = int(node.feature_idx)
            thr[i] = float(node.threshold)
            left[i] = add(node.left) + offset
            right[i] = add(node.right) + offset
        return i

    add(root)
    return feat, thr, left, right, cls


def export_forest(rf, path, scaler_mean, scaler_std):
    class_index = {c: i for i, c in enumerate(rf.classes_)}
    F, T, L, R, C, roots = [], [], [], [], [], []
    for tree in rf.trees:
        off = len(F)
        roots.append(off)
        f, t, l, r, c = _flatten(tree.root, class_index, off)
        F += f; T += t; L += l; R += r; C += c
    np.savez_compressed(path, feat=np.array(F, np.int32), thr=np.array(T), left=np.array(L, np.int32),
                        right=np.array(R, np.int32), cls=np.array(C, np.int32),
                        roots=np.array(roots, np.int32), classes=np.array(rf.classes_, dtype=str),
                        mean=np.asarray(scaler_mean), std=np.asarray(scaler_std))


class FlatForest:
    """Inference-only forest. predict_one() takes a RAW (unscaled) 115-feature vector."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        # plain Python lists: scalar indexing is much faster than on numpy arrays
        self.feat, self.thr = z["feat"].tolist(), z["thr"].tolist()
        self.left, self.right, self.cls = z["left"].tolist(), z["right"].tolist(), z["cls"].tolist()
        self.roots = z["roots"].tolist()
        self.classes = [str(c) for c in z["classes"]]
        self.mean, self.std = z["mean"], z["std"]

    def predict_proba_one(self, x_raw):
        x = ((x_raw - self.mean) / self.std).tolist()
        votes = [0] * len(self.classes)
        feat, thr, left, right, cls = self.feat, self.thr, self.left, self.right, self.cls
        for i in self.roots:
            while cls[i] < 0:
                i = left[i] if x[feat[i]] <= thr[i] else right[i]
            votes[cls[i]] += 1
        n = len(self.roots)
        return [v / n for v in votes]

    def predict_one(self, x_raw):
        p = self.predict_proba_one(x_raw)
        return self.classes[max(range(len(p)), key=p.__getitem__)]

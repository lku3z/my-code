# -*- coding: utf-8 -*-
"""knn.py —— 手写 KNN 核心

包含：
    距离函数       euclidean_distance / manhattan_distance / euclidean_distances / manhattan_distances
    最近邻搜索     find_nearest_neighbors
    KNN 分类器     KNNClassifier（多数投票，含平票处理）
    KNN 回归       KNNRegressor（均值 / 距离加权）
    工具函数       train_test_split（分层划分）、StandardScaler（Z-score）、
                   accuracy_score / confusion_matrix / classification_report /
                   mean_absolute_error / r2_score
"""

import numpy as np

RANDOM_SEED = 42          # 全局随机种子
EPS = 1e-12               # 距离加权时防止除零


# ====================== 距离函数（第 4 题） ======================
def euclidean_distance(x, y):
    """两个样本的欧氏距离"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("样本维度不一致：%d 维 vs %d 维" % (x.size, y.size))
    diff = x - y
    return float(np.sqrt(np.dot(diff, diff)))


def manhattan_distance(x, y):
    """两个样本的曼哈顿距离"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("样本维度不一致：%d 维 vs %d 维" % (x.size, y.size))
    return float(np.abs(x - y).sum())


def _as_2d(X, name="X"):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("%s 为空：至少需要一条样本" % name)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("%s 的维度应为 1 或 2，实际为 %d" % (name, X.ndim))
    if not np.all(np.isfinite(X)):
        raise ValueError("%s 中含有 NaN 或 Inf，请先处理缺失值" % name)
    return X


def euclidean_distances(X, Y):
    """欧氏距离矩阵，返回 shape=(len(X), len(Y))"""
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("特征维度不一致：%d 维 vs %d 维" % (X.shape[1], Y.shape[1]))
    # (n,1,d) - (1,m,d) 广播，再求平方和
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def manhattan_distances(X, Y):
    """曼哈顿距离矩阵，返回 shape=(len(X), len(Y))"""
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("特征维度不一致：%d 维 vs %d 维" % (X.shape[1], Y.shape[1]))
    diff = X[:, None, :] - Y[None, :, :]
    return np.abs(diff).sum(axis=2)


def distance_matrix(X, Y, metric="euclidean"):
    if metric == "euclidean":
        return euclidean_distances(X, Y)
    if metric == "manhattan":
        return manhattan_distances(X, Y)
    raise ValueError("不支持的距离度量：%r" % (metric,))


# ====================== 最近邻搜索（第 5 题） ======================
def find_nearest_neighbors(X_train, X_test, k=5, metric="euclidean"):
    """返回 (距离, 索引)，形状都是 (len(X_test), k)，按距离升序"""
    X_train = _as_2d(X_train, "X_train")
    X_test = _as_2d(X_test, "X_test")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k 必须是正整数，实际为 %r" % (k,))
    if k > X_train.shape[0]:
        raise ValueError("k=%d 大于训练样本数 %d，无法取到这么多邻居"
                         % (k, X_train.shape[0]))
    D = distance_matrix(X_test, X_train, metric)
    order = np.argsort(D, axis=1, kind="stable")   # stable 保证等距时顺序固定
    idx = order[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    return dist, idx


# ====================== KNN 分类器（第 6 题） ======================
class KNNClassifier:
    """k 近邻分类器

    n_neighbors : k
    weights     : 'uniform' 等权投票 / 'distance' 按 1/d 加权
    tie_break   : 'smallest' 平票取标签最小的类 / 'nearest' 平票取最近邻居的类
    """

    def __init__(self, n_neighbors=5, weights="uniform",
                 metric="euclidean", tie_break="smallest"):
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors <= 0:
            raise ValueError("n_neighbors 必须是正整数，实际为 %r" % (n_neighbors,))
        if weights not in ("uniform", "distance"):
            raise ValueError("weights 只能是 'uniform' 或 'distance'")
        if tie_break not in ("smallest", "nearest"):
            raise ValueError("tie_break 只能是 'smallest' 或 'nearest'")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.tie_break = tie_break
        self.tie_count = 0      # 预测中发生平票的次数

    def fit(self, X, y):
        X = _as_2d(X, "X_train")
        y = np.asarray(y).ravel()
        if y.size != X.shape[0]:
            raise ValueError("样本数与标签数不一致：X %d 行，y %d 个"
                             % (X.shape[0], y.size))
        if self.n_neighbors > X.shape[0]:
            raise ValueError("k=%d 大于训练样本数 %d"
                             % (self.n_neighbors, X.shape[0]))
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        self.fitted = True
        return self

    def vote(self, labels, dists):
        """一个样本的 k 个邻居投票，返回 (预测标签, 各类得分)"""
        labels = np.asarray(labels).ravel()
        dists = np.asarray(dists, dtype=float).ravel()
        if labels.size != dists.size:
            raise ValueError("邻居标签数与距离数不一致")
        if labels.size == 0:
            raise ValueError("没有可投票的邻居")

        if self.weights == "uniform":
            w = np.ones(len(dists))
        else:
            w = 1.0 / np.where(dists <= EPS, EPS, dists)

        scores = {}
        for lab, wt in zip(labels, w):
            scores[lab] = scores.get(lab, 0.0) + float(wt)

        best = max(scores.values())
        tol = 1e-12 * max(1.0, best)
        top = sorted([lab for lab, s in scores.items() if abs(s - best) <= tol])

        if len(top) > 1:                       # 平票
            self.tie_count += 1
            if self.tie_break == "nearest":
                for lab, d in zip(labels, dists):
                    if lab in top:
                        return lab, scores        # 最近的邻居
            return top[0], scores                 # 标签最小的类
        return top[0], scores

    def predict(self, X):
        if not getattr(self, "fitted", False):
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        dist, idx = find_nearest_neighbors(
            self.X_train, X, k=self.n_neighbors, metric=self.metric)
        preds = []
        for r in range(dist.shape[0]):
            lab, _ = self.vote(self.y_train[idx[r]], dist[r])
            preds.append(lab)
        return np.array(preds)

    def predict_proba(self, X):
        if not getattr(self, "fitted", False):
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        dist, idx = find_nearest_neighbors(
            self.X_train, X, k=self.n_neighbors, metric=self.metric)
        proba = np.zeros((dist.shape[0], self.classes.size))
        for r in range(dist.shape[0]):
            _, scores = self.vote(self.y_train[idx[r]], dist[r])
            total = sum(scores.values())
            for c, i in zip(self.classes, range(self.classes.size)):
                proba[r, i] = scores.get(c, 0.0) / total
        return proba


# ====================== KNN 回归（第 7 题） ======================
class KNNRegressor:
    """k 近邻回归：uniform 取均值，distance 取 1/d 加权平均"""

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean"):
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors <= 0:
            raise ValueError("n_neighbors 必须是正整数，实际为 %r" % (n_neighbors,))
        if weights not in ("uniform", "distance"):
            raise ValueError("weights 只能是 'uniform' 或 'distance'")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        X = _as_2d(X, "X_train")
        y = np.asarray(y, dtype=float).ravel()
        if y.size != X.shape[0]:
            raise ValueError("样本数与标签数不一致：X %d 行，y %d 个"
                             % (X.shape[0], y.size))
        if self.n_neighbors > X.shape[0]:
            raise ValueError("k=%d 大于训练样本数 %d"
                             % (self.n_neighbors, X.shape[0]))
        self.X_train = X
        self.y_train = y
        self.fitted = True
        return self

    def predict(self, X):
        if not getattr(self, "fitted", False):
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        dist, idx = find_nearest_neighbors(
            self.X_train, X, k=self.n_neighbors, metric=self.metric)
        y = self.y_train[idx]
        if self.weights == "uniform":
            return y.mean(axis=1)
        w = 1.0 / np.where(dist <= EPS, EPS, dist)
        return (y * w).sum(axis=1) / w.sum(axis=1)


# ====================== 数据划分 + 标准化（第 3 题） ======================
def train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=True):
    """分层划分训练集/测试集，返回 X_train, X_test, y_train, y_test"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = X.shape[0]
    if n == 0:
        raise ValueError("样本为空，无法划分")
    if X.ndim != 2:
        raise ValueError("X 应为二维数组")
    n_test = int(round(n * test_size)) if isinstance(test_size, float) else int(test_size)
    n_test = max(1, min(n - 1, n_test))

    rng = np.random.default_rng(random_state)
    if stratify:
        train_idx, test_idx = [], []
        for cls in np.unique(y):
            idx = rng.permutation(np.where(y == cls)[0])
            n_cls_test = int(round(idx.size * n_test / n))
            test_idx.extend(idx[:n_cls_test])
            train_idx.extend(idx[n_cls_test:])
        train_idx = rng.permutation(np.array(train_idx))
        test_idx = rng.permutation(np.array(test_idx))
    else:
        perm = rng.permutation(n)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Z-score 标准化：x' = (x - mean) / std，只在训练集上 fit"""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std < EPS, 1.0, std)   # 常数列防止除零
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.mean.size:
            raise ValueError("特征数不一致：fit 时 %d，现在 %d"
                             % (self.mean.size, X.shape[1]))
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ====================== 评估指标（第 8 题） ======================
def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("长度不一致：%d vs %d" % (y_true.size, y_pred.size))
    if y_true.size == 0:
        raise ValueError("空数组无法计算准确率")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """返回 (labels, 矩阵)，matrix[i, j] = 真实 i 被判成 j 的个数"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    pos = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return labels, cm


def classification_report(y_true, y_pred, labels=None):
    """逐类 precision / recall / f1 / support 的文本报告"""
    labels, cm = confusion_matrix(y_true, y_pred, labels)
    lines = ["%s %10s %10s %10s %8s" % ("类别", "precision", "recall", "f1", "support")]
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        support = cm[i, :].sum()
        pred_n = cm[:, i].sum()
        p = tp / pred_n if pred_n else 0.0
        r = tp / support if support else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        lines.append("%-8s %10.4f %10.4f %10.4f %8d" % (lab, p, r, f1, support))
    lines.append("准确率: %.4f" % accuracy_score(y_true, y_pred))
    return "\n".join(lines)


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else 0.0

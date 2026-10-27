# -*- coding: utf-8 -*-
"""main.py —— KNN 主程序

流程：数据加载 -> 数据探索 -> 预处理 -> 距离函数 -> 最近邻 -> 训练 -> 评估
      -> 回归 -> 交叉验证选 k -> 绘图 -> sklearn 对照 -> 异常与性能 -> 自测

运行：python main.py
输出：控制台结果；图片保存到 plots/
"""

import os
import platform
import sys
import time
import unittest

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knn import (KNNClassifier, KNNRegressor, RANDOM_SEED, StandardScaler,
                 accuracy_score, classification_report, confusion_matrix,
                 euclidean_distance, euclidean_distances, find_nearest_neighbors,
                 manhattan_distance, mean_absolute_error, r2_score,
                 train_test_split)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "dataset.csv")
PLOTS = os.path.join(HERE, "plots")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
COLORS = ["#2E86AB", "#E4572E", "#17B890"]

os.makedirs(PLOTS, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def title(n, text):
    print("\n" + "=" * 62)
    print("第 %d 题  %s" % (n, text))
    print("=" * 62)


def save(fig, name):
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   图片已保存: plots/%s" % name)


# ==================================================================
# 1 工程环境
# ==================================================================
def q1():
    title(1, "工程环境与代码规范")
    print("Python 版本 :", platform.python_version())
    print("NumPy/pandas:", np.__version__, "/", pd.__version__)
    print("随机种子     : RANDOM_SEED =", RANDOM_SEED)
    print("\n项目结构：")
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        depth = root[len(HERE):].count(os.sep)
        print("   " + "    " * depth + (os.path.basename(root) or "knn_project") + "/")
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            print("   " + "    " * (depth + 1) + f)


# ==================================================================
# 2 数据加载与探索
# ==================================================================
def q2():
    title(2, "数据加载与探索")
    if not os.path.exists(DATA):
        raise FileNotFoundError("找不到数据文件: " + DATA)
    df = pd.read_csv(DATA)

    print("文件      : data/dataset.csv")
    print("样本数    :", df.shape[0])
    print("特征数    :", len(FEATURES), FEATURES)
    print("缺失值总数:", int(df.isna().sum().sum()))

    print("\n类别分布：")
    counts = df["species"].value_counts().sort_index()
    for name, cnt in counts.items():
        print("   %-12s %3d 条  %5.2f%%" % (name, cnt, cnt / len(df) * 100))

    print("\n特征范围：")
    for c in FEATURES:
        print("   %-14s min=%.1f  max=%.1f  mean=%.2f  std=%.2f"
              % (c, df[c].min(), df[c].max(), df[c].mean(), df[c].std()))

    # 图1 类别分布
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(list(map(str, counts.index)), counts.values, color=COLORS)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    ax.set_title("类别分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    ax.legend(["各类样本数"])
    save(fig, "01_类别分布.png")

    # 图2 二维散点图
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for i, name in enumerate(CLASS_NAMES):
        m = df["target"] == i
        ax.scatter(df.loc[m, "petal_length"], df.loc[m, "petal_width"],
                   s=42, color=COLORS[i], edgecolor="white", label=name)
    ax.set_title("花瓣长度 vs 花瓣宽度")
    ax.set_xlabel("petal_length (cm)")
    ax.set_ylabel("petal_width (cm)")
    ax.legend(title="类别")
    save(fig, "02_特征散点图.png")

    print("\n结论：150 样本、4 特征、3 类各 50 条，无缺失值。")
    return df


# ==================================================================
# 3 预处理
# ==================================================================
def q3(df):
    title(3, "数据预处理：划分 + Z-score 标准化")
    X = df[FEATURES].to_numpy(float)
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=True)

    scaler = StandardScaler().fit(X_train)          # 只在训练集 fit
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("训练集 / 测试集:", X_train.shape[0], "/", X_test.shape[0])
    print("训练集类别分布 :", dict(zip(*np.unique(y_train, return_counts=True))))
    print("测试集类别分布 :", dict(zip(*np.unique(y_test, return_counts=True)))
          if len(np.unique(y_test)) > 1 else "-")
    print("\n标准化参数（来自训练集）：")
    for i, c in enumerate(FEATURES):
        print("   %-14s mean=%7.3f std=%6.3f" % (c, scaler.mean[i], scaler.std[i]))
    print("\n训练集标准化后 mean:", np.round(X_train_s.mean(axis=0), 6))
    print("训练集标准化后 std :", np.round(X_train_s.std(axis=0), 6))
    print("测试集只 transform，未使用测试集统计量 -> 无信息泄漏")

    a = train_test_split(X, y, random_state=RANDOM_SEED)
    b = train_test_split(X, y, random_state=RANDOM_SEED)
    print("同种子重复划分一致:", all(np.array_equal(u, v) for u, v in zip(a, b)))

    return X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler


# ==================================================================
# 4 距离函数
# ==================================================================
def q4(X_train):
    title(4, "距离函数：欧氏 / 曼哈顿")
    print("%-18s %-18s %8s %8s %8s %8s"
          % ("x", "y", "欧氏", "numpy", "曼哈顿", "numpy"))
    for a, b in [([0, 0], [3, 4]), ([1, 2, 3], [4, 6, 3]), ([-1, 2.5], [2, -1.5])]:
        a, b = np.array(a, float), np.array(b, float)
        print("%-18s %-18s %8.4f %8.4f %8.4f %8.4f"
              % (a.tolist(), b.tolist(),
                 euclidean_distance(a, b), np.linalg.norm(a - b),
                 manhattan_distance(a, b), np.abs(a - b).sum()))

    D = euclidean_distances(X_train, X_train[:5])
    print("\n矩阵形式 euclidean_distances ->", D.shape)
    err = max(abs(D[i, j] - euclidean_distance(X_train[i], X_train[j]))
              for i in range(0, len(X_train), 5) for j in range(5))
    print("矩阵结果 vs 逐对计算，最大误差: %.2e" % err)


# ==================================================================
# 5 最近邻搜索
# ==================================================================
def q5(X_train_s, X_test_s, y_train):
    title(5, "最近邻搜索")
    dist, idx = find_nearest_neighbors(X_train_s, X_test_s, k=5)
    for r in range(3):
        print("\n测试样本 %d 的前 5 个邻居：" % r)
        print("   序号  训练集索引     距离   类别")
        for j in range(5):
            print("   %2d     %5d     %7.4f   %d"
                  % (j + 1, idx[r, j], dist[r, j], y_train[idx[r, j]]))
        d_all = np.linalg.norm(X_train_s - X_test_s[r], axis=1)
        print("   距离升序: %s | 线性扫描复核: %d | 一致: %s"
              % (bool(np.all(np.diff(dist[r]) >= 0)), int(np.argmin(d_all)),
                 int(np.argmin(d_all)) == int(idx[r, 0])))


# ==================================================================
# 6 分类器
# ==================================================================
def q6(X_train_s, X_test_s, y_train, y_test):
    title(6, "KNN 分类器：多数投票 + 平票处理")
    clf = KNNClassifier(n_neighbors=5).fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    print("k = 5")
    print("测试集准确率: %.4f (%d/%d)"
          % (accuracy_score(y_test, pred), int((y_test == pred).sum()), len(y_test)))
    print("预测各类数量:", dict(zip(*np.unique(pred, return_counts=True))))
    print("本次平票次数:", clf.tie_count)

    c1 = KNNClassifier(n_neighbors=4, tie_break="smallest").fit(
        X_train_s[:10], y_train[:10])
    lab, scores = c1.vote(["B", "A", "A", "B"], [1.0, 1.0, 1.0, 1.0])
    print("\n平票演示：标签 ['B','A','A','B'] 等权得分", scores,
          "-> smallest 取", lab)
    c2 = KNNClassifier(n_neighbors=4, tie_break="nearest").fit(
        X_train_s[:10], y_train[:10])
    lab2, _ = c2.vote(["B", "A", "A", "B"], [0.5, 0.9, 1.2, 1.5])
    print("nearest 取最近邻居的类 ->", lab2)

    print("\n前 3 个测试样本的类别概率:")
    print(clf.predict_proba(X_test_s[:3]).round(3))
    return pred


# ==================================================================
# 7 回归
# ==================================================================
def q7(df):
    title(7, "KNN 回归：均值 / 距离加权")
    feats = ["sepal_length", "sepal_width", "petal_length"]
    X = df[feats].to_numpy(float)
    y = df["petal_width"].to_numpy(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    print("输入特征:", feats, " 目标: petal_width")
    print("\n  k   weights         MAE        R2")
    best = None
    for k in (3, 5, 8):
        for w in ("uniform", "distance"):
            reg = KNNRegressor(n_neighbors=k, weights=w).fit(X_train_s, y_train)
            p = reg.predict(X_test_s)
            mae, r2 = mean_absolute_error(y_test, p), r2_score(y_test, p)
            print("%3d   %-10s %8.4f  %8.4f" % (k, w, mae, r2))
            if best is None or mae < best[2]:
                best = (k, w, mae, r2, y_test, p)
    print("\n最优: k=%d, weights=%s, MAE=%.4f, R2=%.4f"
          % (best[0], best[1], best[2], best[3]))

    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(best[4], best[5], s=40, color=COLORS[0], label="测试样本")
    lim = [min(best[4].min(), best[5].min()) - 0.05,
           max(best[4].max(), best[5].max()) + 0.05]
    ax.plot(lim, lim, "--", color="#C1121F", label="y=x")
    ax.set_title("KNN 回归：预测值 vs 真实值（k=%d）" % best[0])
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.legend()
    save(fig, "07_回归对比.png")


# ==================================================================
# 8 评估
# ==================================================================
def q8(y_test, pred):
    title(8, "模型评估：准确率、混淆矩阵、分类报告")
    print("准确率: %.4f" % accuracy_score(y_test, pred))
    labels, cm = confusion_matrix(y_test, pred)

    print("\n混淆矩阵（行=真实，列=预测）：")
    print("        " + "".join("%8s" % l for l in labels) + "%8s" % "合计")
    for i, l in enumerate(labels):
        print("   %-5s" % l + "".join("%8d" % v for v in cm[i]) + "%8d" % cm[i].sum())
    print("   合计  " + "".join("%8d" % v for v in cm.sum(axis=0)) + "%8d" % cm.sum())

    print("\n分类报告：")
    print(classification_report(y_test, pred))

    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([CLASS_NAMES[l] for l in labels], rotation=15)
    ax.set_yticklabels([CLASS_NAMES[l] for l in labels])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    save(fig, "04_混淆矩阵.png")


# ==================================================================
# 9 交叉验证选 k
# ==================================================================
def kfold(y, n_splits=5, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for i, j in enumerate(idx):
            buckets[i % n_splits].append(j)
    return [np.array(sorted(b)) for b in buckets]


def q9(X_train_s, y_train, X_test_s, y_test):
    title(9, "k 值选择与交叉验证")
    folds = kfold(y_train, 5)
    print("候选 k = 1~30，训练集内部 5 折分层交叉验证（测试集不参与选参）")
    print("\n  k    CV均值   CV标准差   训练集    测试集")
    ks, cv, sd, tr, te = [], [], [], [], []
    for k in range(1, 31):
        scores = []
        for f in folds:
            mask = np.ones(len(y_train), bool)
            mask[f] = False
            m = KNNClassifier(n_neighbors=k).fit(X_train_s[mask], y_train[mask])
            scores.append(accuracy_score(y_train[f], m.predict(X_train_s[f])))
        full = KNNClassifier(n_neighbors=k).fit(X_train_s, y_train)
        ks.append(k)
        cv.append(np.mean(scores))
        sd.append(np.std(scores))
        tr.append(accuracy_score(y_train, full.predict(X_train_s)))
        te.append(accuracy_score(y_test, full.predict(X_test_s)))

    best_i = int(np.argmax(cv))
    for i, k in enumerate(ks):
        if k <= 10 or k % 5 == 0 or i == best_i:
            mark = "   <- 最优" if i == best_i else ""
            print("%3d  %7.4f  %8.4f  %7.4f  %7.4f%s"
                  % (k, cv[i], sd[i], tr[i], te[i], mark))
    best_k = ks[best_i]
    print("\n最优 k = %d，CV 准确率 %.4f，测试集准确率 %.4f"
          % (best_k, cv[best_i], te[best_i]))
    print("理由：k 太小过拟合（训练接近 1、CV 波动大）；k 太大欠拟合（边界过平滑）。")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ks, np.array(tr) * 100, "o-", markersize=4, label="训练集准确率")
    ax.plot(ks, np.array(cv) * 100, "s-", markersize=4, label="5 折交叉验证准确率")
    ax.axvline(best_k, ls="--", color="gray")
    ax.annotate("最优 k=%d" % best_k, xy=(best_k, cv[best_i] * 100),
                xytext=(12, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_title("k 值与准确率的关系")
    ax.set_xlabel("k")
    ax.set_ylabel("准确率 (%)")
    ax.set_xticks(ks[::2])
    ax.legend()
    save(fig, "03_k准确率曲线.png")
    return best_k, cv[best_i]


# ==================================================================
# 10 绘图：决策边界
# ==================================================================
def q10(df, best_k):
    title(10, "可视化分析：决策边界 + 误分类标注")
    X = df[["petal_length", "petal_width"]].to_numpy(float)
    y = df["target"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)
    scaler = StandardScaler().fit(X_train)
    model = KNNClassifier(n_neighbors=best_k).fit(scaler.transform(X_train), y_train)

    pred_all = model.predict(scaler.transform(X))
    wrong = np.where(pred_all != y)[0]
    print("k=%d，全部 150 个样本中误分类 %d 个: %s"
          % (best_k, len(wrong), wrong.tolist()))

    xx, yy = np.meshgrid(
        np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 0.02),
        np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 0.02))
    zz = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.contourf(xx, yy, zz, alpha=0.22, colors=COLORS)
    ax.contour(xx, yy, zz, colors="k", linewidths=0.6, levels=[0.5, 1.5])
    for i, name in enumerate(CLASS_NAMES):
        m = y == i
        ax.scatter(X[m, 0], X[m, 1], s=38, color=COLORS[i],
                   edgecolor="white", label=name)
    ax.scatter(X[wrong, 0], X[wrong, 1], s=150, facecolors="none",
               edgecolors="red", linewidths=1.8, label="误分类 (%d)" % len(wrong))
    for i in wrong:
        ax.annotate(str(i), xy=(X[i, 0], X[i, 1]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color="red")
    ax.set_title("KNN 决策边界（k=%d）" % best_k)
    ax.set_xlabel("petal_length (cm)")
    ax.set_ylabel("petal_width (cm)")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "05_决策边界.png")


# ==================================================================
# 11 sklearn 对照
# ==================================================================
def q11(X_train_s, X_test_s, y_train, y_test, best_k):
    title(11, "sklearn 对照实验")
    from sklearn.neighbors import KNeighborsClassifier

    mine = KNNClassifier(n_neighbors=best_k).fit(X_train_s, y_train)
    p_mine = mine.predict(X_test_s)
    sk = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_s, y_train)
    p_sk = sk.predict(X_test_s)

    print("k =", best_k)
    print("手写准确率    : %.4f" % accuracy_score(y_test, p_mine))
    print("sklearn 准确率: %.4f" % accuracy_score(y_test, p_sk))
    print("预测不同的样本数:", int((p_mine != p_sk).sum()), "/", len(y_test))

    d1, i1 = find_nearest_neighbors(X_train_s, X_test_s, k=best_k)
    d2, i2 = sk.kneighbors(X_test_s)
    print("近邻索引一致率: %.2f%%" % (np.mean(i1 == i2) * 100))
    print("近邻距离最大差: %.2e" % np.abs(d1 - d2).max())
    print("\n差异原因：1) 浮点运算顺序不同；2) 等距邻居排序不稳定；"
          "3) 平票裁决规则不同；4) sklearn 默认 KD-Tree，只影响速度。")


# ==================================================================
# 12 异常与性能
# ==================================================================
def q12(X_train, y_train):
    title(12, "异常处理与性能优化")
    print("一、异常处理")

    def check(name, func):
        try:
            func()
            print("   %-14s 没有报错" % name)
        except Exception as e:
            print("   %-14s %s: %s" % (name, type(e).__name__, e))

    check("空训练集", lambda: KNNClassifier(n_neighbors=3).fit(np.empty((0, 4)), []))
    check("维度不一致", lambda: euclidean_distances(np.zeros((3, 4)), np.zeros((3, 5))))
    check("k 大于样本数", lambda: KNNClassifier(n_neighbors=999).fit(X_train[:5], y_train[:5]))
    check("特征含 NaN", lambda: KNNClassifier(n_neighbors=3).fit(
        np.array([[1.0, np.nan]]), [0]))
    check("未 fit 就预测", lambda: KNNClassifier(n_neighbors=3).predict(X_train[:2]))
    check("标签数不一致", lambda: KNNClassifier(n_neighbors=3).fit(X_train, y_train[:-1]))
    check("非法 k", lambda: KNNClassifier(n_neighbors=0))
    check("空测试集", lambda: KNNClassifier(n_neighbors=3).fit(X_train, y_train)
          .predict(np.empty((0, 4))))

    rng = np.random.default_rng(RANDOM_SEED)
    A, B = rng.normal(size=(600, 4)), rng.normal(size=(200, 4))

    t0 = time.perf_counter()
    D_vec = euclidean_distances(A, B)
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    D_loop = np.empty((600, 200))
    for i in range(600):
        for j in range(200):
            D_loop[i, j] = euclidean_distance(A[i], B[j])
    t_loop = time.perf_counter() - t0

    print("\n二、性能对照（600 x 200 x 4 维）")
    print("   纯 Python 循环 : %.4f 秒" % t_loop)
    print("   NumPy 向量化   : %.4f 秒" % t_vec)
    print("   加速比         : %.1f 倍" % (t_loop / t_vec))
    print("   结果是否一致   :", np.allclose(D_vec, D_loop))

    fig, ax = plt.subplots(figsize=(5.6, 4))
    bars = ax.bar(["纯 Python 循环", "NumPy 向量化"], [t_loop, t_vec], color=COLORS[:2])
    for b, t in zip(bars, [t_loop, t_vec]):
        ax.text(b.get_x() + b.get_width() / 2, t, "%.3fs" % t,
                ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_title("距离计算耗时对比（加速 %.0f 倍）" % (t_loop / t_vec))
    ax.set_ylabel("耗时 (秒)")
    save(fig, "06_性能对比.png")


# ==================================================================
# 13 自测
# ==================================================================
class TestDistance(unittest.TestCase):
    def test_euclidean(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)

    def test_manhattan(self):
        self.assertAlmostEqual(manhattan_distance([0, 0], [3, 4]), 7.0)

    def test_matrix(self):
        D = euclidean_distances(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
        self.assertAlmostEqual(D[0, 0], 5.0)

    def test_dimension_error(self):
        with self.assertRaises(ValueError):
            euclidean_distance([1, 2], [1, 2, 3])


class TestVote(unittest.TestCase):
    def setUp(self):
        self.clf = KNNClassifier(n_neighbors=3).fit(
            np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))

    def test_majority(self):
        label, scores = self.clf.vote(["A", "A", "B"], [1.0, 2.0, 3.0])
        self.assertEqual(label, "A")
        self.assertEqual(scores["A"], 2.0)

    def test_tie_smallest(self):
        clf = KNNClassifier(n_neighbors=4, tie_break="smallest").fit(
            np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 1, 0, 1]))
        label, _ = clf.vote(["B", "A", "A", "B"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(label, "A")
        self.assertEqual(clf.tie_count, 1)


class TestPredict(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(6, 1, (20, 2))])
        self.y = np.array([0] * 20 + [1] * 20)

    def test_perfect(self):
        clf = KNNClassifier(n_neighbors=3).fit(self.X, self.y)
        self.assertEqual(accuracy_score(self.y, clf.predict(self.X)), 1.0)

    def test_k_too_large(self):
        with self.assertRaises(ValueError):
            KNNClassifier(n_neighbors=41).fit(self.X, self.y)

    def test_not_fitted(self):
        with self.assertRaises(RuntimeError):
            KNNClassifier(n_neighbors=3).predict(self.X[:2])

    def test_regressor(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        reg = KNNRegressor(n_neighbors=2).fit(X, y)
        self.assertAlmostEqual(reg.predict(np.array([[0.1]]))[0], 1.5)


def q13():
    title(13, "测试与复现")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDistance)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVote))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPredict))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\n测试用例数: %d，失败: %d，错误: %d"
          % (result.testsRun, len(result.failures), len(result.errors)))


# ==================================================================
def main():
    print("第 8 周实践一 · KNN")
    print("Python", platform.python_version(), "| 数据: data/dataset.csv")

    q1()
    df = q2()
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = q3(df)
    q4(X_train)
    q5(X_train_s, X_test_s, y_train)
    pred = q6(X_train_s, X_test_s, y_train, y_test)
    q7(df)
    q8(y_test, pred)
    best_k, best_cv = q9(X_train_s, y_train, X_test_s, y_test)
    q10(df, best_k)
    q11(X_train_s, X_test_s, y_train, y_test, best_k)
    q12(X_train, y_train)
    q13()

    print("\n" + "=" * 62)
    print("全部完成：结果见控制台，7 张图见 plots/")
    print("=" * 62)


if __name__ == "__main__":
    main()

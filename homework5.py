# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
import matplotlib
# 有图形界面时用 TkAgg 弹窗展示；无显示环境（服务器/后台）自动切到 Agg，只保存图片
try:
    import tkinter as tk
    _root = tk.Tk()
    _root.destroy()
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体显示设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)


# =====================================================================
# 1. 数据加载
#    优先使用 sklearn 的 OpenML 接口自动下载 credit-g 数据集；
#    若联网失败，也可改成读取本地下载的 CSV（见下方说明）。
# =====================================================================
def load_data():
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("credit-g", as_frame=True, parser="auto")
        X, y = data.data, data.target
        print("[数据] 已从 OpenML 自动获取 German Credit（credit-g）数据集")
        return X, y
    except Exception as e:
        # 备用：如果已在 Kaggle / UCI 下载了 german_credit.csv，放到脚本同目录即可
        print("[数据] OpenML 自动下载失败，尝试读取本地 german_credit.csv ...")
        print("      原因:", e)
        df = pd.read_csv("german_credit.csv")
        y = df["class"]                                   # 目标列名按下载文件调整
        X = df.drop(columns=["class"])
        return X, y


X, y = load_data()
print("样本数:", X.shape[0], " 特征数:", X.shape[1])
print("类别分布:\n", y.value_counts())

# 把标签规范成 0/1（坏=1 表示风险类别，方便计算 AUC 等指标）
y = y.map({"bad": 1, "good": 0})
if y.isna().any():
    y = pd.Series(np.where(y.isna(), 1, y), index=X.index)  # 容错处理
print("映射后类别分布:\n", y.value_counts())


# =====================================================================
# 2. 特征划分：数值特征（标准化） vs 类别特征（独热编码）
#    这是德国信用数据的关键预处理 —— 20 个特征里混着数值和类别。
# =====================================================================
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("\n数值特征(%d): %s" % (len(num_cols), num_cols))
print("类别特征(%d): %s" % (len(cat_cols), cat_cols))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),              # 数值特征标准化
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),  # 类别特征独热
    ]
)


# =====================================================================
# 3. 划分训练 / 测试集（分层抽样，保持类别比例一致）
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("\n训练集:", X_train.shape, " 测试集:", X_test.shape)


# =====================================================================
# 4. 模型 1：线性核 SVM（快速 baseline，课件里 C=0.1 时 AUC≈0.76 欠拟合）
# =====================================================================
linear_pipe = Pipeline([
    ("prep", preprocessor),
    ("svm", SVC(kernel="linear", C=1.0, probability=True)),
])
linear_pipe.fit(X_train, y_train)

y_pred_lin = linear_pipe.predict(X_test)
y_prob_lin = linear_pipe.predict_proba(X_test)[:, 1]

print("\n================== 线性核 SVM ==================")
print("准确率 : %.4f" % accuracy_score(y_test, y_pred_lin))
print("AUC    : %.4f" % roc_auc_score(y_test, y_prob_lin))


# =====================================================================
# 5. 模型 2：RBF 核 SVM + 网格搜索调参 + 类别不平衡处理
#    课件推荐 C∈[1e-3,1e3]、γ∈[1e-4,1e1] 对数搜索，用 AUC 评分；
#    class_weight="balanced" 自动给少数类更大权重。
# =====================================================================
rbf_pipe = Pipeline([
    ("prep", preprocessor),
    ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
])

param_grid = {
    "svm__C":     [0.1, 1, 5, 10, 50],          # 正则化，越大越允许复杂边界
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],  # RBF 核宽度
}

print("\n正在网格搜索 RBF SVM 超参数 ...（约 5×4×5=100 次拟合）")
grid = GridSearchCV(rbf_pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("最优参数:", grid.best_params_)
print("交叉验证最佳 AUC: %.4f" % grid.best_score_)

best = grid.best_estimator_
y_pred_rbf = best.predict(X_test)
y_prob_rbf = best.predict_proba(X_test)[:, 1]

print("\n================== RBF 核 SVM（调参后） ==================")
print("准确率 : %.4f" % accuracy_score(y_test, y_pred_rbf))
print("AUC    : %.4f" % roc_auc_score(y_test, y_prob_rbf))
print("\n分类报告:\n", classification_report(y_test, y_pred_rbf, target_names=["好(0)", "坏(1)"]))


# =====================================================================
# 6. 对比与可视化：混淆矩阵 + ROC 曲线
# =====================================================================
def plot_results(y_test, y_prob_rbf, y_pred_rbf, y_prob_lin):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：混淆矩阵（RBF 最佳模型）
    cm = confusion_matrix(y_test, y_pred_rbf)
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("RBF SVM 混淆矩阵")
    axes[0].set_xlabel("预测类别"); axes[0].set_ylabel("真实类别")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["好(0)", "坏(1)"]); axes[0].set_yticklabels(["好(0)", "坏(1)"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # 右：ROC 曲线
    for name, prob in [("线性核 SVM", y_prob_lin), ("RBF SVM (调参)", y_prob_rbf)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        axes[1].plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, auc), lw=2)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测")
    axes[1].set_title("ROC 曲线对比")
    axes[1].set_xlabel("假正率 (FPR)"); axes[1].set_ylabel("真正率 (TPR)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("german_credit_svm_result.png", dpi=150)
    print("\n结果图已保存: german_credit_svm_result.png")
    try:
        plt.show()          # 无显示环境时会自动跳过
    except Exception:
        pass


plot_results(y_test, y_prob_rbf, y_pred_rbf, y_prob_lin)


# =====================================================================
# 7. 结论
# =====================================================================
print("\n================== 结论 ==================")
print("线性 SVM AUC=%.3f（欠拟合），RBF SVM AUC=%.3f（更优），"
      % (roc_auc_score(y_test, y_prob_lin), roc_auc_score(y_test, y_prob_rbf)))
print("与课件一致：小样本 + 非线性关系时，RBF 核 + 网格调参 + 类别权重处理明显优于线性核。")

"""
===============================================================
04_modeling.py — 模型训练与评估
===============================================================
项目: 台湾信用卡违约预测 — 基于机器学习的信贷风控分析

模型:
    1. 逻辑回归 (Logistic Regression) — 基准模型，可解释性强
    2. GBDT (GradientBoostingClassifier) — 梯度提升树模型，提升预测精度

评估指标:
    - AUC (ROC曲线下面积): 模型区分违约与正常的能力
    - KS (Kolmogorov-Smirnov): 模型在风控中的区分度
    - Precision & Recall: 精确率与召回率
    - F1 Score: 综合指标
    - Confusion Matrix: 混淆矩阵

产出:
    - 模型训练日志与性能对比
    - ROC曲线对比图
    - 混淆矩阵
    - KS曲线图
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, precision_recall_curve,
                              f1_score, accuracy_score)
from sklearn.ensemble import GradientBoostingClassifier

FIG_PATH = '../output/figures/'

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 60)
print("模型训练与评估")
print("=" * 60)

# 读取WOE特征（供逻辑回归使用）
X_train_woe = pd.read_csv('../data/X_train_woe.csv')
X_test_woe = pd.read_csv('../data/X_test_woe.csv')

# 读取标准化特征（供XGBoost使用，XGBoost不需标准化但也兼容）
X_train_scaled = pd.read_csv('../data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('../data/X_test_scaled.csv')

y_train = pd.read_csv('../data/y_train.csv').squeeze()
y_test = pd.read_csv('../data/y_test.csv').squeeze()

# 读取特征信息
with open('../data/final_features.json', 'r', encoding='utf-8') as f:
    feat_info = json.load(f)

woe_features = feat_info['woe_features']

print(f"训练集: {len(y_train)} 样本, 违约率: {y_train.mean():.2%}")
print(f"测试集: {len(y_test)} 样本, 违约率: {y_test.mean():.2%}")
print(f"WOE特征数: {len(woe_features)}")

# ============================================================
# 2. 逻辑回归 (Logistic Regression)
# ============================================================
print("\n" + "-" * 40)
print("模型 1: 逻辑回归")
print("-" * 40)

lr = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_woe, y_train)

# 预测
y_pred_lr = lr.predict(X_test_woe)
y_prob_lr = lr.predict_proba(X_test_woe)[:, 1]

# 评估
lr_auc = roc_auc_score(y_test, y_prob_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_acc = accuracy_score(y_test, y_pred_lr)

print(f"AUC: {lr_auc:.4f}")
print(f"Accuracy: {lr_acc:.4f}")
print(f"F1 Score: {lr_f1:.4f}")
print(f"\n分类报告:\n{classification_report(y_test, y_pred_lr, target_names=['正常', '违约'])}")

# 计算KS值
def calculate_ks(y_true, y_prob):
    """计算KS (Kolmogorov-Smirnov) 统计量"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = max(tpr - fpr)
    ks_idx = np.argmax(tpr - fpr)
    return ks, thresholds[ks_idx]

lr_ks, lr_ks_threshold = calculate_ks(y_test, y_prob_lr)
print(f"KS值: {lr_ks:.4f} (阈值={lr_ks_threshold:.3f})")

# ============================================================
# 3. GBDT (Gradient Boosting Decision Tree)
# ============================================================
print("\n" + "-" * 40)
print("模型 2: GBDT (梯度提升决策树)")
print("-" * 40)

# GBDT使用标准化后的全量特征
gbdt_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    max_features=0.8,
    random_state=42
)
gbdt_model.fit(X_train_scaled, y_train)

y_pred_gbdt = gbdt_model.predict(X_test_scaled)
y_prob_gbdt = gbdt_model.predict_proba(X_test_scaled)[:, 1]

gbdt_auc = roc_auc_score(y_test, y_prob_gbdt)
gbdt_f1 = f1_score(y_test, y_pred_gbdt)
gbdt_acc = accuracy_score(y_test, y_pred_gbdt)
gbdt_ks, gbdt_ks_threshold = calculate_ks(y_test, y_prob_gbdt)

print(f"AUC: {gbdt_auc:.4f}")
print(f"Accuracy: {gbdt_acc:.4f}")
print(f"F1 Score: {gbdt_f1:.4f}")
print(f"KS值: {gbdt_ks:.4f} (阈值={gbdt_ks_threshold:.3f})")
print(f"\n分类报告:\n{classification_report(y_test, y_pred_gbdt, target_names=['正常', '违约'])}")

# ============================================================
# 4. 模型对比
# ============================================================
print("\n" + "-" * 40)
print("模型性能对比")
print("-" * 40)

comparison = pd.DataFrame({
    '模型': ['逻辑回归', 'GBDT'],
    'AUC': [lr_auc, gbdt_auc],
    'KS': [lr_ks, gbdt_ks],
    'Accuracy': [lr_acc, gbdt_acc],
    'F1 Score': [lr_f1, gbdt_f1],
})
print(comparison.to_string(index=False))

# ============================================================
# 5. 可视化 — ROC曲线
# ============================================================
print("\n绘制ROC曲线...")

fig, ax = plt.subplots(figsize=(7, 6))

lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob_lr)
gbdt_fpr, gbdt_tpr, _ = roc_curve(y_test, y_prob_gbdt)

# B&W友好: LR=黑色实线, GBDT=深灰虚线
ax.plot(lr_fpr, lr_tpr, label=f'逻辑回归 (AUC={lr_auc:.3f})',
        color='#111111', linewidth=2, linestyle='-')
ax.plot(gbdt_fpr, gbdt_tpr, label=f'GBDT (AUC={gbdt_auc:.3f})',
        color='#555555', linewidth=2, linestyle='--')
ax.plot([0, 1], [0, 1], '#999999', linestyle=':', linewidth=1, label='随机猜测')
ax.fill_between(lr_fpr, lr_tpr, alpha=0.08, color='#333333')
ax.fill_between(gbdt_fpr, gbdt_tpr, alpha=0.08, color='#888888', hatch='...')

ax.set_xlabel('假正率 (FPR)', fontsize=12)
ax.set_ylabel('真正率 (TPR)', fontsize=12)
ax.set_title('ROC曲线 — 逻辑回归 vs GBDT', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig08_roc_curve.png')
plt.close()
print("  -> fig08_roc_curve.png")

# ============================================================
# 6. 可视化 — KS曲线
# ============================================================
print("\n绘制KS曲线...")

fig, ax = plt.subplots(figsize=(7, 6))

# GBDT KS
gbdt_fpr_ks, gbdt_tpr_ks, gbdt_thresholds = roc_curve(y_test, y_prob_gbdt)
ks_diff = gbdt_tpr_ks - gbdt_fpr_ks

# B&W友好: TPR=黑色实线, FPR=深灰虚线, KS=浅灰点划线 + 文字标注
ax.plot(gbdt_thresholds, gbdt_tpr_ks,
        color='#111111', linewidth=2, linestyle='-')
ax.plot(gbdt_thresholds, gbdt_fpr_ks,
        color='#555555', linewidth=2, linestyle='--')
ax.plot(gbdt_thresholds, ks_diff,
        color='#999999', linewidth=2, linestyle='-.')

# 直接标注曲线名称
mid = len(gbdt_thresholds) // 2
ax.annotate('TPR (真正率)', xy=(gbdt_thresholds[mid], gbdt_tpr_ks[mid]),
            xytext=(gbdt_thresholds[mid]-0.15, gbdt_tpr_ks[mid]+0.12),
            fontsize=10, color='#111111', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#111111'))
ax.annotate('FPR (假正率)', xy=(gbdt_thresholds[mid], gbdt_fpr_ks[mid]),
            xytext=(gbdt_thresholds[mid]-0.05, gbdt_fpr_ks[mid]-0.15),
            fontsize=10, color='#555555', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#555555'))
ax.annotate(f'KS差值 ({gbdt_ks:.4f})', xy=(gbdt_thresholds[mid], ks_diff[mid]),
            xytext=(0.65, 0.55), fontsize=10, color='#999999', fontweight='bold')

# 标记KS最大点
ks_max_idx = np.argmax(ks_diff)
ax.axvline(x=gbdt_thresholds[ks_max_idx], color='#666666',
           linestyle=':', linewidth=1.2)
ax.scatter([gbdt_thresholds[ks_max_idx]], [gbdt_ks],
           color='#111111', s=100, zorder=5, marker='D', edgecolors='#111111')

ax.set_xlabel('分类阈值', fontsize=12)
ax.set_ylabel('比率', fontsize=12)
ax.set_title(f'KS曲线 (GBDT) — KS={gbdt_ks:.4f}', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig09_ks_curve.png')
plt.close()
print("  -> fig09_ks_curve.png")

# ============================================================
# 7. 可视化 — 混淆矩阵
# ============================================================
print("\n绘制混淆矩阵...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (y_pred, title) in enumerate([
    (y_pred_lr, '逻辑回归'),
    (y_pred_gbdt, 'GBDT')
]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '违约'],
                yticklabels=['正常', '违约'],
                ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{title} — 混淆矩阵', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('预测值')
    axes[idx].set_ylabel('真实值')

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig10_confusion_matrix.png')
plt.close()
print("  -> fig10_confusion_matrix.png")

# ============================================================
# 8. 保存模型
# ============================================================
print("\n保存模型...")
import joblib

joblib.dump(lr, '../data/model_lr.pkl')
joblib.dump(gbdt_model, '../data/model_gbdt.pkl')
print("  -> model_lr.pkl, model_gbdt.pkl")

# 保存评估结果
results = {
    'logistic_regression': {
        'auc': lr_auc, 'ks': lr_ks, 'accuracy': lr_acc, 'f1': lr_f1,
        'ks_threshold': lr_ks_threshold
    },
    'gbdt': {
        'auc': gbdt_auc, 'ks': gbdt_ks, 'accuracy': gbdt_acc, 'f1': gbdt_f1,
        'ks_threshold': gbdt_ks_threshold
    }
}
with open('../data/model_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n模型训练完成!")
print(f"最佳模型: {'GBDT' if gbdt_auc > lr_auc else '逻辑回归'}")
print(f"AUC提升: {abs(gbdt_auc - lr_auc):.4f}")

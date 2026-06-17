"""
===============================================================
05_insights.py — 模型可解释性与业务洞察
===============================================================
项目: 台湾信用卡违约预测 — 基于机器学习的信贷风控分析

功能:
    1. GBDT 特征重要性分析 (sklearn内置)
    2. 逻辑回归特征系数分析
    3. 评分卡构建与信用评分分布
    4. 风控策略分析 (阈值选择、抓获率、误拒率)
    5. 业务指标汇总

注意: 由于网络限制无法安装SHAP库，本脚本使用sklearn内置的
      feature_importances_ 和 permutation importance 作为替代。
      两种方法均能有效反映特征对模型的贡献程度。
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.inspection import permutation_importance

FIG_PATH = '../output/figures/'

# ============================================================
# 1. 加载模型与数据
# ============================================================
print("=" * 60)
print("模型可解释性与业务洞察")
print("=" * 60)

gbdt_model = joblib.load('../data/model_gbdt.pkl')
lr_model = joblib.load('../data/model_lr.pkl')

X_train_scaled = pd.read_csv('../data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('../data/X_test_scaled.csv')
X_train_woe = pd.read_csv('../data/X_train_woe.csv')
X_test_woe = pd.read_csv('../data/X_test_woe.csv')
y_test = pd.read_csv('../data/y_test.csv').squeeze()

with open('../data/model_results.json', 'r', encoding='utf-8') as f:
    model_results = json.load(f)
with open('../data/final_features.json', 'r', encoding='utf-8') as f:
    feat_info = json.load(f)

print(f"测试集: {len(y_test)} 样本")

# ============================================================
# 2. GBDT 特征重要性分析
# ============================================================
print("\n" + "-" * 40)
print("GBDT 特征重要性分析")
print("-" * 40)

# 2a. 内置特征重要性 (基于基尼不纯度的平均减少量)
print("\nsklearn 内置特征重要性 TOP 15:")
feat_imp = pd.DataFrame({
    '特征': X_train_scaled.columns,
    '重要性': gbdt_model.feature_importances_
}).sort_values('重要性', ascending=False)

for _, row in feat_imp.head(15).iterrows():
    bar = '█' * int(row['重要性'] * 200)
    print(f"  {row['特征'][:30]:<30s} {row['重要性']:.4f} {bar}")

# 2b. Permutation Importance (基于打乱特征后模型性能下降程度, 更稳健)
print("\n计算 Permutation Importance (测试集)...")
sample_test = X_test_scaled.sample(min(3000, len(X_test_scaled)), random_state=42)
sample_y = y_test.iloc[sample_test.index]

perm_imp = permutation_importance(
    gbdt_model, sample_test, sample_y,
    n_repeats=5, random_state=42, scoring='roc_auc'
)

perm_imp_df = pd.DataFrame({
    '特征': X_test_scaled.columns,
    '重要性': perm_imp.importances_mean,
    '标准差': perm_imp.importances_std
}).sort_values('重要性', ascending=False)

# ============================================================
# 2c. 可视化 — GBDT 内置特征重要性
# ============================================================
print("\n绘制GBDT特征重要性图...")

fig, ax = plt.subplots(figsize=(10, 8))
top_n = 15
top_features = feat_imp.head(top_n)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, top_n))

bars = ax.barh(range(top_n), top_features['重要性'].values[::-1], color=colors[::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['特征'].values[::-1], fontsize=9)
ax.set_xlabel('特征重要性 (Gini Importance)', fontsize=12)
ax.set_title('GBDT 特征重要性排名 (TOP 15)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, top_features['重要性'].values[::-1])):
    ax.text(bar.get_width() + 0.001, i, f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig11_feature_importance.png', dpi=150)
plt.close()
print("  -> fig11_feature_importance.png")

# ============================================================
# 2d. 可视化 — Permutation Importance
# ============================================================
print("\n绘制Permutation Importance图...")

fig, ax = plt.subplots(figsize=(10, 8))
top_n_perm = 15
top_perm = perm_imp_df.head(top_n_perm)

ax.barh(range(top_n_perm), top_perm['重要性'].values[::-1],
        xerr=top_perm['标准差'].values[::-1],
        color=plt.cm.Oranges(np.linspace(0.3, 0.9, top_n_perm)),
        capsize=3)
ax.set_yticks(range(top_n_perm))
ax.set_yticklabels(top_perm['特征'].values[::-1], fontsize=9)
ax.set_xlabel('AUC下降幅度 (越大越重要)', fontsize=12)
ax.set_title('Permutation Importance (打乱特征后AUC下降)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig12_permutation_importance.png', dpi=150)
plt.close()
print("  -> fig12_permutation_importance.png")

# ============================================================
# 3. 逻辑回归 — 特征系数分析
# ============================================================
print("\n" + "-" * 40)
print("逻辑回归 特征系数分析")
print("-" * 40)

lr_coef = pd.DataFrame({
    '特征': X_train_woe.columns,
    '系数': lr_model.coef_[0]
})
lr_coef['系数绝对值'] = np.abs(lr_coef['系数'])
lr_coef = lr_coef.sort_values('系数绝对值', ascending=False)

print("\n逻辑回归 WOE系数 TOP 12:")
for _, row in lr_coef.iterrows():
    direction = '高风险↑' if row['系数'] > 0 else '低风险↓'
    print(f"  {row['特征'][:30]:<30s} 系数={row['系数']:+.4f}  {direction}")

# 逻辑回归系数可视化
fig, ax = plt.subplots(figsize=(10, 6))
colors_lr = ['#F44336' if c > 0 else '#4CAF50' for c in lr_coef['系数'].values[::-1]]
bars = ax.barh(range(len(lr_coef)), lr_coef['系数'].values[::-1], color=colors_lr)
ax.set_yticks(range(len(lr_coef)))
ax.set_yticklabels(lr_coef['特征'].values[::-1], fontsize=9)
ax.set_xlabel('WOE系数', fontsize=12)
ax.set_title('逻辑回归 WOE特征系数 (正=高风险, 负=低风险)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, lr_coef['系数'].values[::-1])):
    ax.text(val + (0.02 if val >= 0 else -0.08), i,
            f'{val:+.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig13_lr_coefficients.png', dpi=150)
plt.close()
print("  -> fig13_lr_coefficients.png")

# ============================================================
# 4. 评分卡构建
# ============================================================
print("\n" + "-" * 40)
print("信用评分卡构建")
print("-" * 40)

def make_scorecard(lr_model, feature_names, base_score=600, pdo=50, base_odds=19):
    """
    将逻辑回归模型转换为信用评分卡。
    评分标准:
      base_score: 基准分 600 (对应 odds=19:1, 违约率~5%)
      PDO: 每降低50分, odds翻倍
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    coef = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]

    scorecard = pd.DataFrame({
        '特征': feature_names,
        '系数': coef,
        '分值权重': -factor * coef
    })
    return scorecard, factor, offset, intercept

def calculate_credit_score(df_woe, scorecard, factor, offset, intercept):
    base_part = offset - factor * intercept
    score = base_part + df_woe.dot(scorecard['分值权重'].values)
    return score.clip(300, 900)

scorecard, factor, offset, intercept = make_scorecard(lr_model, list(X_train_woe.columns))
test_scores = calculate_credit_score(X_test_woe, scorecard, factor, offset, intercept)

print(f"\n评分卡 TOP 5 特征分值权重:")
scorecard_sorted = scorecard.iloc[np.abs(scorecard['分值权重']).argsort()[::-1]]
for _, row in scorecard_sorted.head(5).iterrows():
    print(f"  {row['特征'][:30]:<30s} 权重={row['分值权重']:+.1f}分")

print(f"\n信用评分统计:")
print(f"  均值: {test_scores.mean():.1f}  中位数: {test_scores.median():.1f}")
print(f"  标准差: {test_scores.std():.1f}  范围: [{test_scores.min():.0f}, {test_scores.max():.0f}]")

# ============================================================
# 5. 评分分布可视化
# ============================================================
print("\n绘制信用评分分布...")

scores_default = test_scores[y_test.values == 1]
scores_normal = test_scores[y_test.values == 0]

fig, ax = plt.subplots(figsize=(10, 5))
# B&W友好: 正常=浅灰填充, 违约=深灰+hatch斜线
n1, bins1, patches1 = ax.hist(scores_normal, bins=40, alpha=0.8,
        label=f'正常还款 (均值={scores_normal.mean():.0f})',
        color='#CCCCCC', density=True, edgecolor='#666666', linewidth=0.5)
n2, bins2, patches2 = ax.hist(scores_default, bins=40, alpha=0.9,
        label=f'违约 (均值={scores_default.mean():.0f})',
        color='#555555', density=True, edgecolor='#111111', linewidth=0.5, hatch='///')
ax.axvline(x=test_scores.median(), color='gray', linestyle='--',
           alpha=0.7, label=f'中位数={test_scores.median():.0f}')
ax.set_xlabel('信用评分', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('信用评分分布 — 违约 vs 正常还款', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_PATH + 'fig14_credit_score_distribution.png', dpi=150)
plt.close()
print("  -> fig14_credit_score_distribution.png")

# ============================================================
# 6. 风控策略分析
# ============================================================
print("\n" + "=" * 60)
print("风控策略分析")
print("=" * 60)

y_prob_gbdt = gbdt_model.predict_proba(X_test_scaled)[:, 1]

def analyze_risk_strategy(y_true, y_prob, thresholds):
    results = []
    total_bad = y_true.sum()
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        n_reject = y_pred.sum()
        if n_reject > 0:
            n_bad_caught = ((y_pred == 1) & (y_true == 1)).sum()
            catch_rate = n_bad_caught / max(total_bad, 1)
            n_false_reject = ((y_pred == 1) & (y_true == 0)).sum()
            false_reject_rate = n_false_reject / n_reject
        else:
            catch_rate = 0
            false_reject_rate = 0
        results.append({
            'threshold': thresh,
            '拒绝率': n_reject / len(y_pred),
            '抓获率': catch_rate,
            '误拒率': false_reject_rate,
            '通过率': 1 - n_reject / len(y_pred)
        })
    return pd.DataFrame(results)

thresholds = np.arange(0.05, 0.95, 0.05)
strategy_df = analyze_risk_strategy(y_test.values, y_prob_gbdt, thresholds)

print("\n不同阈值下的风控指标 (GBDT):")
print(f"{'阈值':>8s}  {'拒绝率':>8s}  {'抓获率':>8s}  {'误拒率':>8s}  {'通过率':>8s}")
print("-" * 48)
for _, row in strategy_df.iterrows():
    print(f"{row['threshold']:8.2f}  {row['拒绝率']:7.1%}  {row['抓获率']:7.1%}  "
          f"{row['误拒率']:7.1%}  {row['通过率']:7.1%}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(strategy_df['threshold'], strategy_df['抓获率'] * 100,
        label='抓获率 (Bad Capture Rate)', color='#4CAF50', linewidth=2, marker='o')
ax.plot(strategy_df['threshold'], strategy_df['误拒率'] * 100,
        label='误拒率 (False Reject Rate)', color='#F44336', linewidth=2, marker='s')
ax.plot(strategy_df['threshold'], strategy_df['通过率'] * 100,
        label='通过率 (Approval Rate)', color='#2196F3', linewidth=2, marker='^')
ax.set_xlabel('违约概率阈值', fontsize=12)
ax.set_ylabel('百分比 (%)', fontsize=12)
ax.set_title('风控阈值选择 — 抓获率 vs 误拒率 vs 通过率', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='center right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# 标注推荐区间
mid_idx = len(strategy_df) // 3
ax.axvspan(strategy_df.iloc[mid_idx]['threshold'],
           strategy_df.iloc[mid_idx*2]['threshold'],
           alpha=0.1, color='green', label='推荐阈值区间')
ax.legend(fontsize=10, loc='center right')
plt.tight_layout()
plt.savefig(FIG_PATH + 'fig15_risk_strategy.png', dpi=150)
plt.close()
print("  -> fig15_risk_strategy.png")

# ============================================================
# 7. 业务指标汇总
# ============================================================
print("\n" + "=" * 60)
print("业务指标汇总")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════╗
║         信用卡违约预测模型 — 业务指标总结              ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  【模型性能】                                         ║
║    逻辑回归: AUC={model_results['logistic_regression']['auc']:.4f}, KS={model_results['logistic_regression']['ks']:.4f}              ║
║    GBDT   : AUC={model_results['gbdt']['auc']:.4f}, KS={model_results['gbdt']['ks']:.4f}              ║
║                                                      ║
║  【信用评分】                                         ║
║    违约样本均分: {scores_default.mean():.0f}分                              ║
║    正常样本均分: {scores_normal.mean():.0f}分                               ║
║    分差: {scores_normal.mean() - scores_default.mean():.0f}分                                        ║
║                                                      ║
║  【关键特征 (GBDT重要性)】                             ║
║    TOP1: {feat_imp.iloc[0]['特征'][:25]:25s} ({feat_imp.iloc[0]['重要性']:.4f})               ║
║    TOP2: {feat_imp.iloc[1]['特征'][:25]:25s} ({feat_imp.iloc[1]['重要性']:.4f})               ║
║    TOP3: {feat_imp.iloc[2]['特征'][:25]:25s} ({feat_imp.iloc[2]['重要性']:.4f})               ║
║                                                      ║
║  【风控建议】                                         ║
║    * 推荐采用GBDT模型 (AUC={model_results['gbdt']['auc']:.3f})                    ║
║    * 还款状态类特征为核心监控指标                       ║
║    * 建议设置信用评分580分为审批参考线                  ║
║    * 额度使用率>80%且还款率<0.3的客户需重点关注         ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

print("可解释性分析完成! 共生成 5 张图表")

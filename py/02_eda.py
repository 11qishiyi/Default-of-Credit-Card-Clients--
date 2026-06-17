"""
===============================================================
02_eda.py — 探索性数据分析 (Exploratory Data Analysis)
===============================================================
项目: 台湾信用卡违约预测 — 基于机器学习的信贷风控分析

功能:
    1. 违约率总体分析与可视化
    2. 类别特征与违约率的关系 (性别、教育、婚姻)
    3. 数值特征分布与违约对比 (年龄、额度等)
    4. 还款状态与违约率的关系
    5. 特征相关性热力图
    6. 衍生特征有效性验证
    7. 所有图表保存到 ../output/figures/
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

FIG_PATH = '../output/figures/'

# ============================================================
# 1. 读取数据 (使用原始数据+one-hot编码的组合)
# ============================================================
print("读取处理后的数据...")
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv').squeeze()
y_test = pd.read_csv('../data/y_test.csv').squeeze()

df_train = X_train.copy()
df_train['次月违约'] = y_train.values
df_test = X_test.copy()
df_test['次月违约'] = y_test.values
df = pd.concat([df_train, df_test])

# 重建类别标签 (从one-hot反推)
df['性别_label'] = df['性别_男'].map({1: '男', 0: '女'})

def get_edu(row):
    if row['教育程度_研究生'] == 1: return '研究生'
    if row['教育程度_大学'] == 1: return '大学'
    if row['教育程度_高中'] == 1: return '高中'
    return '其他'

def get_marriage(row):
    if row['婚姻状况_已婚'] == 1: return '已婚'
    if row['婚姻状况_单身'] == 1: return '单身'
    return '其他'

df['教育程度_label'] = df.apply(get_edu, axis=1)
df['婚姻状况_label'] = df.apply(get_marriage, axis=1)

print(f"总数据: {len(df)} 样本, 违约率: {df['次月违约'].mean():.2%}")

# ============================================================
# 2. 违约率总体分析
# ============================================================
print("\n绘制违约率总体分布...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

default_counts = df['次月违约'].value_counts()
axes[0].pie(default_counts.values,
            labels=['正常还款', '发生违约'],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            explode=(0, 0.05),
            startangle=90)
axes[0].set_title('样本违约率分布', fontsize=14, fontweight='bold')

train_rate = df_train['次月违约'].mean() * 100
test_rate = df_test['次月违约'].mean() * 100
bars = axes[1].bar(['训练集', '测试集'], [train_rate, test_rate],
                   color=['#2196F3', '#FF9800'], width=0.5)
axes[1].set_title('训练集与测试集违约率对比', fontsize=14, fontweight='bold')
axes[1].set_ylabel('违约率 (%)')
axes[1].set_ylim(0, 30)
for bar, val in zip(bars, [train_rate, test_rate]):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_PATH + 'fig01_default_rate_distribution.png')
plt.close()
print("  -> fig01_default_rate_distribution.png")

# ============================================================
# 3. 类别特征与违约率
# ============================================================
print("\n绘制类别特征分析...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 性别
for ax, (col_label, title, colors) in zip(axes, [
    ('性别_label', '性别', ['#E91E63', '#2196F3']),
    ('教育程度_label', '教育程度', ['#9C27B0']),
    ('婚姻状况_label', '婚姻状况', ['#FF9800'])
]):
    grouped = df.groupby(col_label)['次月违约'].mean().sort_values(ascending=False) * 100
    bars = ax.bar(grouped.index, grouped.values, color=colors[0] if len(colors)==1
                  else [colors[1] if x=='男' else colors[0] for x in grouped.index])
    ax.set_ylabel('违约率 (%)')
    ax.set_ylim(0, 35)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=10)

plt.suptitle('不同类别特征的违约率对比', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH + 'fig02_categorical_features.png')
plt.close()
print("  -> fig02_categorical_features.png")

# ============================================================
# 4. 数值特征分布 — 按违约分组对比
# ============================================================
print("\n绘制数值特征分布对比...")

num_features = [
    ('信用额度(NT$)', '信用额度'),
    ('年龄', '年龄'),
    ('月均还款金额', '月均还款金额'),
    ('额度使用率', '额度使用率'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (col, title) in enumerate(num_features):
    ax = axes[idx // 2][idx % 2]
    if col in df.columns:
        default_data = df[df['次月违约'] == 1][col]
        non_default_data = df[df['次月违约'] == 0][col]
        ax.hist(non_default_data, bins=40, alpha=0.6, label='正常还款',
                color='#4CAF50', density=True)
        ax.hist(default_data, bins=40, alpha=0.6, label='违约',
                color='#F44336', density=True)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylabel('密度')

plt.suptitle('数值特征分布 — 违约 vs 正常还款', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_PATH + 'fig03_numeric_distribution.png')
plt.close()
print("  -> fig03_numeric_distribution.png")

# ============================================================
# 5. 还款状态与违约率关系
# ============================================================
print("\n绘制还款状态分析...")

pay_cols = [f'{m}月还款状态' for m in [4, 5, 6, 7, 8, 9]]
month_labels = ['4月', '5月', '6月', '7月', '8月', '9月']
available_pay_cols = [c for c in pay_cols if c in df.columns]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图
pay_data = []
for col in available_pay_cols:
    grouped = df.groupby(col)['次月违约'].mean() * 100
    pay_data.append(grouped)

# B&W友好的线型和标记循环
linestyles = ['-', '--', '-.', ':', '-', '--']
markers = ['o', 's', '^', 'D', 'v', 'p']
line_colors = ['#111111', '#333333', '#555555', '#777777', '#999999', '#BBBBBB']

for idx, state in enumerate([-2, -1, 0, 1, 2, 3]):
    rates = []
    valid = []
    for m_idx, g in enumerate(pay_data):
        if state in g.index:
            rates.append(g[state])
            valid.append(m_idx)
    if valid:
        label_map = {-2: '无消费', -1: '按时还款', 0: '循环信用',
                     1: '逾期1月', 2: '逾期2月', 3: '逾期3月+'}
        axes[0].plot(valid, rates,
                     linestyle=linestyles[idx % len(linestyles)],
                     marker=markers[idx % len(markers)],
                     color=line_colors[idx % len(line_colors)],
                     label=label_map.get(state, str(state)),
                     linewidth=1.8, markersize=5)

axes[0].set_xticks(range(len(available_pay_cols)))
axes[0].set_xticklabels(month_labels[:len(available_pay_cols)])
axes[0].set_xlabel('月份')
axes[0].set_ylabel('违约率 (%)')
axes[0].set_title('各还款状态下不同月份的违约率', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(True, alpha=0.3)

# 右图
if '最大逾期月数' in df.columns:
    bins = [-3, -1, 0, 1, 2, 3, 9]
    labels = ['无消费/按时', '循环信用', '逾期1月', '逾期2月', '逾期3-8月', '严重逾期']
    df_temp = df.copy()
    df_temp['逾期等级'] = pd.cut(df_temp['最大逾期月数'], bins=bins,
                                  labels=labels, right=False)
    grouped = df_temp.groupby('逾期等级', observed=True)['次月违约'].mean() * 100
    bars = axes[1].bar(range(len(grouped)), grouped.values,
                       color=plt.cm.Reds(np.linspace(0.3, 0.9, len(grouped))))
    axes[1].set_xticks(range(len(grouped)))
    axes[1].set_xticklabels(grouped.index, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('违约率 (%)')
    axes[1].set_title('最大逾期等级与违约率', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, grouped.values)):
        axes[1].text(i, bar.get_height() + 1, f'{val:.1f}%',
                     ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig04_payment_status.png')
plt.close()
print("  -> fig04_payment_status.png")

# ============================================================
# 6. 特征相关性热力图
# ============================================================
print("\n绘制特征相关性热力图...")

corr_features = ['信用额度(NT$)', '年龄', '额度使用率', '还款率',
                 '最大逾期月数', '逾期总次数', '次月违约']
corr_features = [c for c in corr_features if c in df.columns]
corr_matrix = df[corr_features].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('关键特征相关性矩阵', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_PATH + 'fig05_correlation_heatmap.png')
plt.close()
print("  -> fig05_correlation_heatmap.png")

# ============================================================
# 7. 额度使用率与还款率 — 违约风险边界
# ============================================================
print("\n绘制风险边界散点图...")

if '额度使用率' in df.columns and '还款率' in df.columns:
    df_sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))

    normal = df_sample[df_sample['次月违约'] == 0]
    default = df_sample[df_sample['次月违约'] == 1]
    ax.scatter(normal['额度使用率'], normal['还款率'],
               c='#4CAF50', alpha=0.3, s=10, label='正常还款')
    ax.scatter(default['额度使用率'], default['还款率'],
               c='#F44336', alpha=0.5, s=15, label='违约')
    ax.set_xlabel('额度使用率', fontsize=12)
    ax.set_ylabel('还款率', fontsize=12)
    ax.set_title('额度使用率 vs 还款率 — 违约风险分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.05, 1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH + 'fig06_risk_boundary.png')
    plt.close()
    print("  -> fig06_risk_boundary.png")

print("\nEDA 完成! 共生成 6 张图表到 output/figures/")

"""
===============================================================
03_feature_eng.py — 特征工程 (Feature Engineering)
===============================================================
项目: 台湾信用卡违约预测 — 基于机器学习的信贷风控分析

功能:
    1. WOE (Weight of Evidence) 分箱 — 将连续变量转为信用风险度量
    2. IV (Information Value) 计算 — 评估每个特征对违约的预测能力
    3. 特征筛选 — 根据IV值选择有效特征
    4. 特征重要性可视化
    5. 导出WOE转换后的特征供逻辑回归使用

原理:
    WOE = ln((违约样本占比) / (正常样本占比))
    IV = Σ((违约% - 正常%) × WOE)
    IV值越大，特征对违约的区分能力越强:
      <0.02: 无预测力, 0.02~0.1: 弱, 0.1~0.3: 中等, >0.3: 强
===============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import json
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

FIG_PATH = '../output/figures/'

# ============================================================
# 1. 读取数据
# ============================================================
print("读取数据...")
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv').squeeze()

# 读取特征名列表
with open('../data/feature_names.json', 'r', encoding='utf-8') as f:
    feature_info = json.load(f)
numeric_features = feature_info['numeric_cols']

print(f"训练集: {X_train.shape}")
print(f"数值特征数: {len(numeric_features)}")

# ============================================================
# 2. WOE分箱函数
# ============================================================
print("\n执行WOE分箱与IV计算...")

def calculate_woe_iv(feature, target, n_bins=5):
    """
    对数值特征进行等频分箱，计算各箱的WOE和整体IV值。

    Parameters:
        feature: pd.Series, 特征值
        target: pd.Series, 目标变量 (0/1)
        n_bins: int, 分箱数

    Returns:
        woe_df: DataFrame, 包含各箱的边界、样本数、WOE值
        iv: float, Information Value
    """
    # 等频分箱
    try:
        # 去重后的值如果少于n_bins，则减少分箱数
        unique_vals = feature.nunique()
        actual_bins = min(n_bins, max(2, unique_vals))

        feature_binned, bin_edges = pd.qcut(
            feature, q=actual_bins, duplicates='drop', retbins=True
        )
    except (ValueError, TypeError):
        return None, 0

    # 统计每个箱
    grouped = target.groupby(feature_binned, observed=True)
    total_good = (target == 0).sum()  # 正常
    total_bad = target.sum()          # 违约

    woe_data = []
    iv_total = 0

    for bin_name, group in grouped:
        n_total = len(group)
        n_bad = group.sum()
        n_good = n_total - n_bad

        # 防止除以零
        pct_good = max(n_good, 0.5) / max(total_good, 1)
        pct_bad = max(n_bad, 0.5) / max(total_bad, 1)

        woe = np.log(pct_bad / pct_good)
        iv_bin = (pct_bad - pct_good) * woe
        iv_total += iv_bin

        woe_data.append({
            'bin': str(bin_name),
            'n_samples': n_total,
            'n_bad': int(n_bad),
            'bad_rate': n_bad / n_total if n_total > 0 else 0,
            'woe': woe,
            'iv_bin': iv_bin
        })

    woe_df = pd.DataFrame(woe_data)
    return woe_df, iv_total

# 计算所有数值特征的IV值
iv_results = {}
woe_mappings = {}  # 存储每个特征的WOE映射，供特征转换使用

# 候选数值特征（过滤掉ID和衍生特征中的目标相关信息）
candidate_features = [
    '信用额度(NT$)', '年龄',
    '额度使用率', '还款率',
    '最大逾期月数', '逾期总次数',
    '月均还款金额', '月均账单金额',
]

# 也包括原始的账单和还款金额
for col in numeric_features:
    if col not in candidate_features and '还款状态' not in col and '曾有' not in col:
        candidate_features.append(col)

for col in candidate_features:
    if col not in X_train.columns:
        continue
    woe_df, iv = calculate_woe_iv(X_train[col], y_train, n_bins=5)
    if woe_df is not None:
        iv_results[col] = iv
        woe_mappings[col] = woe_df
        print(f"  {col[:15]:<15s}  IV={iv:.4f}", end='')
        if iv > 0.3: print(" ★★★ 强")
        elif iv > 0.1: print(" ★★  中等")
        elif iv > 0.02: print(" ★   弱")
        else: print("  无预测力")

# ============================================================
# 3. IV值可视化
# ============================================================
print("\n绘制IV值排名图...")

# 按IV降序排列
iv_sorted = sorted(iv_results.items(), key=lambda x: x[1], reverse=True)
iv_names = [x[0][:20] for x in iv_sorted]
iv_values = [x[1] for x in iv_sorted]

# B&W友好: 灰度 + hatch 模式
# 强: 深灰+///, 中: 中灰+\\\, 弱: 浅灰+..., 无: 白+无
colors_bw = []
hatches = []
labels = []
for v in iv_values:
    if v > 0.3:
        colors_bw.append('#444444')
        hatches.append('///')
        labels.append('强')
    elif v > 0.1:
        colors_bw.append('#888888')
        hatches.append('\\\\')
        labels.append('中')
    elif v > 0.02:
        colors_bw.append('#BBBBBB')
        hatches.append('...')
        labels.append('弱')
    else:
        colors_bw.append('#DDDDDD')
        hatches.append('')
        labels.append('无')

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(iv_names)), iv_values, color=colors_bw)

# 添加 hatch 图案
for bar, hatch in zip(bars, hatches):
    if hatch:
        bar.set_hatch(hatch)

ax.set_yticks(range(len(iv_names)))
ax.set_yticklabels(iv_names, fontsize=9)
ax.set_xlabel('Information Value (IV)', fontsize=12)
ax.set_title('特征预测能力排名 (IV值)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# B&W友好的参考线: 不同线型
ax.axvline(x=0.3, color='#333333', linestyle='-', linewidth=1, alpha=0.7, label='IV=0.3 (强)')
ax.axvline(x=0.1, color='#555555', linestyle='--', linewidth=1, alpha=0.7, label='IV=0.1 (中等)')
ax.axvline(x=0.02, color='#777777', linestyle=':', linewidth=1, alpha=0.7, label='IV=0.02 (弱)')
ax.legend(loc='lower right', fontsize=9)

# 在bar上标注IV值和等级
for i, (bar, val, lbl) in enumerate(zip(bars, iv_values, labels)):
    ax.text(bar.get_width() + 0.005, i, f'{val:.3f} ({lbl})',
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_PATH + 'fig07_iv_ranking.png')
plt.close()
print("  -> fig07_iv_ranking.png")

# ============================================================
# 4. WOE转换 — 选取IV>0.02的特征
# ============================================================
print("\n执行WOE特征转换...")

selected_features = [col for col, iv in iv_results.items() if iv > 0.02]
print(f"筛选前特征数: {len(candidate_features)}")
print(f"IV>0.02的特征数: {len(selected_features)}")
print(f"入选特征: {selected_features}")

# 对训练集和测试集执行WOE转换
def apply_woe_transform(df, woe_mappings, selected_features):
    """将原始特征转换为WOE值"""
    df_woe = pd.DataFrame(index=df.index)

    for col in selected_features:
        if col not in woe_mappings or col not in df.columns:
            continue
        woe_df = woe_mappings[col]

        # 为每个样本分配WOE值 (根据其所属分箱)
        woe_values = np.zeros(len(df))
        for _, row in woe_df.iterrows():
            bin_str = row['bin']
            # 解析分箱区间 "(a, b]"
            try:
                left = float(bin_str.split(',')[0].strip('(['))
                right = float(bin_str.split(',')[1].strip('])'))
                mask = (df[col] > left) & (df[col] <= right)
                woe_values[mask] = row['woe']
            except (ValueError, IndexError):
                continue
        df_woe[f'{col}_WOE'] = woe_values

    return df_woe

X_train_woe = apply_woe_transform(X_train, woe_mappings, selected_features)
X_test = pd.read_csv('../data/X_test.csv')
X_test_woe = apply_woe_transform(X_test, woe_mappings, selected_features)

print(f"WOE转换后训练集维度: {X_train_woe.shape}")
print(f"WOE转换后测试集维度: {X_test_woe.shape}")

# ============================================================
# 5. 保存WOE特征
# ============================================================
X_train_woe.to_csv('../data/X_train_woe.csv', index=False)
X_test_woe.to_csv('../data/X_test_woe.csv', index=False)

# 保存最终特征列表
final_features = list(X_train_woe.columns)
with open('../data/final_features.json', 'w', encoding='utf-8') as f:
    json.dump({
        'selected_features': selected_features,
        'woe_features': final_features,
        'iv_results': {k: round(v, 4) for k, v in iv_results.items()}
    }, f, ensure_ascii=False, indent=2)

print(f"\n特征工程完成! 最终使用 {len(final_features)} 个WOE特征")
print("已保存: X_train_woe.csv, X_test_woe.csv, final_features.json")

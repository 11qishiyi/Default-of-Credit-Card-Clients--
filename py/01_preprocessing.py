"""
===============================================================
01_preprocessing.py — 数据预处理
===============================================================
项目: 台湾信用卡违约预测 — 基于机器学习的信贷风控分析
数据: UCI Default of Credit Card Clients (Yeh & Lien, 2009)
      30,000名信用卡持卡人, 23个特征 + 1个目标变量

功能:
    1. 读取原始XLS数据
    2. 特征中文命名与含义映射
    3. 异常类别值检测与修正 (EDUCATION, MARRIAGE)
    4. 添加衍生特征 (还款率、额度使用率等)
    5. 划分训练集与测试集
    6. 导出清洗后数据供后续分析使用
===============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据读取
# ============================================================
print("=" * 60)
print("步骤 1: 读取原始数据")
print("=" * 60)

# 原始XLS文件第1行为变量说明，第2行起为数据
df_raw = pd.read_excel('../data/data.xls', header=1)
print(f"原始数据维度: {df_raw.shape}")
print(f"列名: {list(df_raw.columns)}")

# ============================================================
# 2. 特征中文命名
# ============================================================
print("\n" + "=" * 60)
print("步骤 2: 特征中文命名")
print("=" * 60)

# 构建中英文列名映射表
COLUMN_MAPPING = {
    'ID':                             '客户ID',
    'LIMIT_BAL':                      '信用额度(NT$)',
    'SEX':                            '性别',
    'EDUCATION':                      '教育程度',
    'MARRIAGE':                       '婚姻状况',
    'AGE':                            '年龄',
    'PAY_0':                          '9月还款状态',
    'PAY_2':                          '8月还款状态',
    'PAY_3':                          '7月还款状态',
    'PAY_4':                          '6月还款状态',
    'PAY_5':                          '5月还款状态',
    'PAY_6':                          '4月还款状态',
    'BILL_AMT1':                      '9月账单金额(NT$)',
    'BILL_AMT2':                      '8月账单金额(NT$)',
    'BILL_AMT3':                      '7月账单金额(NT$)',
    'BILL_AMT4':                      '6月账单金额(NT$)',
    'BILL_AMT5':                      '5月账单金额(NT$)',
    'BILL_AMT6':                      '4月账单金额(NT$)',
    'PAY_AMT1':                       '9月还款金额(NT$)',
    'PAY_AMT2':                       '8月还款金额(NT$)',
    'PAY_AMT3':                       '7月还款金额(NT$)',
    'PAY_AMT4':                       '6月还款金额(NT$)',
    'PAY_AMT5':                       '5月还款金额(NT$)',
    'PAY_AMT6':                       '4月还款金额(NT$)',
    'default payment next month':     '次月违约',
}

df = df_raw.rename(columns=COLUMN_MAPPING)
print(f"已重命名 {len(COLUMN_MAPPING)} 个特征")

# ============================================================
# 3. 类别变量取值说明
# ============================================================
print("\n" + "=" * 60)
print("步骤 3: 类别变量清理与编码")
print("=" * 60)

# 3.1 性别: 1=男性, 2=女性
df['性别'] = df['性别'].map({1: '男', 2: '女'})

# 3.2 教育程度: 1=研究生, 2=大学, 3=高中, 4=其他
# 数据中存在 0, 5, 6 等异常值，合并到 4(其他)
EDU_MAP = {1: '研究生', 2: '大学', 3: '高中', 4: '其他',
           0: '其他', 5: '其他', 6: '其他'}
df['教育程度'] = df['教育程度'].map(EDU_MAP)

# 3.3 婚姻状况: 1=已婚, 2=单身, 3=其他
# 数据中存在 0 异常值，合并到 3(其他)
MARRIAGE_MAP = {1: '已婚', 2: '单身', 3: '其他', 0: '其他'}
df['婚姻状况'] = df['婚姻状况'].map(MARRIAGE_MAP)

# 3.4 还款状态 PAY_X 编码说明:
# -2 = 无消费, -1 = 按时还款, 0 = 最低还款额使用循环信用
# 1~8 = 逾期月数 (1=逾期1月, 8=逾期8月+)
print("还款状态 (PAY_X) 取值说明: -2=无消费, -1=按时, 0=使用循环信用, 1~8=逾期月数")
print(f"还款状态分布 (9月):\n{df['9月还款状态'].value_counts().sort_index()}")

print("\n类别变量处理完成:")
print(f"  性别分布:\n{df['性别'].value_counts()}\n")
print(f"  教育程度分布:\n{df['教育程度'].value_counts()}\n")
print(f"  婚姻状况分布:\n{df['婚姻状况'].value_counts()}")

# ============================================================
# 4. 添加衍生特征
# ============================================================
print("\n" + "=" * 60)
print("步骤 4: 衍生特征构造")
print("=" * 60)

# 4.1 6个月平均还款金额
pay_amt_cols = [f'{m}月还款金额(NT$)' for m in [4, 5, 6, 7, 8, 9]]
df['月均还款金额'] = df[pay_amt_cols].mean(axis=1)

# 4.2 6个月平均账单金额
bill_cols = [f'{m}月账单金额(NT$)' for m in [4, 5, 6, 7, 8, 9]]
df['月均账单金额'] = df[bill_cols].mean(axis=1)

# 4.3 额度使用率 = 月均账单 / 信用额度 (避免除零)
df['额度使用率'] = df['月均账单金额'] / df['信用额度(NT$)'].clip(lower=1)

# 4.4 还款率 = 月均还款 / 月均账单 (衡量还款意愿)
df['还款率'] = np.where(
    df['月均账单金额'] > 100,
    df['月均还款金额'] / df['月均账单金额'].clip(lower=100),
    1.0
)
# 还款率截断到 [0, 2] 范围 (超过2视为完全还清)
df['还款率'] = df['还款率'].clip(0, 2)

# 4.5 最大逾期月数 (最近6个月)
pay_cols = [f'{m}月还款状态' for m in [4, 5, 6, 7, 8, 9]]
df['最大逾期月数'] = df[pay_cols].max(axis=1)

# 4.6 逾期总次数 (最近6个月)
df['逾期总次数'] = (df[pay_cols] > 0).sum(axis=1)

# 4.7 是否有严重逾期 (逾期>=3个月)
df['曾有严重逾期'] = (df['最大逾期月数'] >= 3).astype(int)

print(f"已添加 7 个衍生特征，当前总特征数: {df.shape[1]}")
print(f"\n衍生特征摘要:")
print(f"  额度使用率: 均值={df['额度使用率'].mean():.3f}, 中位数={df['额度使用率'].median():.3f}")
print(f"  还款率: 均值={df['还款率'].mean():.3f}, 中位数={df['还款率'].median():.3f}")
print(f"  最大逾期月数: 众数={df['最大逾期月数'].mode().values[0]}")
print(f"  曾有严重逾期比率: {df['曾有严重逾期'].mean():.2%}")

# ============================================================
# 5. 划分训练集与测试集
# ============================================================
print("\n" + "=" * 60)
print("步骤 5: 划分训练集与测试集")
print("=" * 60)

# 删除ID列，分离特征与目标
df = df.drop(columns=['客户ID'])
target_col = '次月违约'

# 对类别变量做独热编码
categorical_cols = ['性别', '教育程度', '婚姻状况']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

# 分层抽样，保持违约率一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本, 违约率 {y_train.mean():.2%}")
print(f"测试集: {X_test.shape[0]} 样本, 违约率 {y_test.mean():.2%}")

# ============================================================
# 6. 数值特征标准化 (供逻辑回归使用)
# ============================================================
print("\n" + "=" * 60)
print("步骤 6: 数值特征标准化")
print("=" * 60)

# 识别数值列 (排除独热编码的列)
onehot_prefixes = ['性别_', '教育程度_', '婚姻状况_']
numeric_cols = [c for c in X_train.columns
                if not any(c.startswith(p) for p in onehot_prefixes)]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"已标准化 {len(numeric_cols)} 个数值特征")

# ============================================================
# 7. 保存处理后的数据
# ============================================================
print("\n" + "=" * 60)
print("步骤 7: 保存处理后的数据")
print("=" * 60)

# 保存前需确保所有列名不含特殊字符（为后续读取兼容）
# 已通过get_dummies自动处理

X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)
X_train_scaled.to_csv('../data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('../data/X_test_scaled.csv', index=False)

# 保存特征名列表供后续使用
import json
with open('../data/feature_names.json', 'w', encoding='utf-8') as f:
    json.dump({
        'numeric_cols': numeric_cols,
        'all_cols': list(X_train.columns)
    }, f, ensure_ascii=False, indent=2)

print("已保存到 data/ 目录:")
print("  X_train.csv, X_test.csv — 原始训练/测试特征")
print("  y_train.csv, y_test.csv — 训练/测试标签")
print("  X_train_scaled.csv, X_test_scaled.csv — 标准化特征(供LR使用)")
print("  feature_names.json — 特征列表")
print("\n预处理完成!")

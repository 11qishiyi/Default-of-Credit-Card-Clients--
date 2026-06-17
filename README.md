# 台湾信用卡违约预测 — 基于机器学习的信贷风控分析

## 项目简介

本项目是《人工智能与机器学习》课程期末项目，聚焦金融科技场景，基于 UCI 台湾信用卡客户数据集（Default of Credit Card Clients, Yeh & Lien, 2009），构建逻辑回归与梯度提升决策树（GBDT）两种违约预测模型，并输出可落地的风控策略与业务建议。

## 数据来源

- **数据集**: Default of Credit Card Clients (UCI Machine Learning Repository)
- **原始论文**: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480
- **规模**: 30,000 名信用卡持卡人，23 个特征 + 1 个目标变量
- **获取**: 下载链接见数据说明文件

## 项目结构

```
credit-card-default/
├── data/
│   ├── 数据文件说明.txt      # 所有数据文件用途说明
│   ├── feature_names.json    # 特征元数据
│   ├── final_features.json   # 最终入选特征与IV值
│   └── model_results.json    # 模型评估结果
├── py/
│   ├── 01_preprocessing.py   # 数据预处理
│   ├── 02_eda.py             # 探索性数据分析
│   ├── 03_feature_eng.py     # 特征工程 (WOE/IV)
│   ├── 04_modeling.py        # 模型训练 (LR + GBDT)
│   ├── 05_insights.py        # 可解释性与风控策略
│   └── 脚本说明.txt          # 脚本详细说明
├── .gitignore
├── LICENSE
└── README.md
```

## 运行环境

- Python 3.8+
- 依赖包:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xlrd joblib python-docx
```

## 运行步骤

1. 下载原始数据: 将 `default of credit card clients.xls` 放入 `data/` 目录
2. 按顺序执行脚本:

```bash
cd py
python 01_preprocessing.py     # 数据预处理 (~30s)
python 02_eda.py               # 探索性分析 (~20s)
python 03_feature_eng.py       # 特征工程 (~30s)
python 04_modeling.py          # 模型训练 (~60s)
python 05_insights.py          # 可解释性与风控建议 (~30s)
```

## 分析方法

### 特征工程
- **WOE (Weight of Evidence) 分箱**: 将连续变量转换为信用风险度量
- **IV (Information Value) 筛选**: 评估特征预测能力，筛选 IV > 0.02 的 12 个特征

### 模型
- **逻辑回归** (L2 正则化): 基准模型，系数可解释，可直接构建信用评分卡
- **GBDT 梯度提升决策树** (200 棵基学习器): 提升预测精度，捕捉非线性交互效应

### 评估
- AUC、KS 统计量、F1 Score、混淆矩阵
- 特征重要性排名、排列重要性 (Permutation Importance)
- 信用评分卡构建 (评分范围 300–900)

## 主要结果

| 模型 | AUC | KS | F1 Score | 准确率 |
|------|-----|-----|----------|--------|
| 逻辑回归 | 0.7503 | 0.3894 | 0.3772 | 79.01% |
| GBDT | 0.7788 | 0.4275 | 0.4649 | 81.61% |

## 关键发现

1. **还款状态是最强预测因子**: 9 月还款状态重要度 0.4126，最大逾期月数 IV = 0.7648
2. **GBDT 全面优于逻辑回归**: AUC 提升 2.86%，KS 提升 3.81%
3. **评分卡有效区分风险**: 正常客户均分 509 vs 违约客户均分 445，分差 64 分
4. **风控阈值可灵活权衡**: 可根据抓获率与通过率之间的业务偏好动态调整

# Team 1 → Team 2 数据交接 / Data Hand-off

**From：** lian（Team 1 — 数据处理与 EDA）
**To：** wang、tang（Team 2）
**Date：** 2026-10-28
**Dataset：** Ames Housing（De Cock, 2011），2,930 条

---

## 1. 交接内容

Team 1 的数据处理部分已完成并**冻结**。包里的两个 CSV 是清洗、编码、划分之后的成品：
`train_encoded.csv` 与 `test_encoded.csv` 已经是切好的两个子集，直接读入即可使用，
不需要再做清洗、插补或编码。

---

## 2. 文件清单

| 文件 | 内容 | 大小 |
|---|---|---:|
| `train_encoded.csv` | 2,344 × 211：`row_id` + 208 特征 + `SalePrice` + `fold` | 2.0 MB |
| `test_encoded.csv` | 586 × 210：`row_id` + 208 特征 + `SalePrice` | 0.5 MB |
| `feature_names.json` | 208 个特征名，与 CSV 列顺序一致 | 6 KB |
| `split_assignments.csv` | 2,930 行：`row_id, split, fold, SalePrice` | 58 KB |
| `sale_condition_lookup.csv` | 2,930 行：`row_id, SaleCondition, SaleType` | 48 KB |
| `eda_summary.json` | 数据侧关键数字（报告引用用） | 2 KB |

---

## 3. 数据说明

### 3.1 208 个特征怎么来的

| 前缀 | 数量 | 编码方式 |
|---|---:|---|
| `ord__` | 22 | 有序类别按真实等级映射为整数（如 `Exter Qual`：`Po`=0 … `Ex`=4） |
| `nom__` | 153 | 无序类别 one-hot，`drop='first'` |
| `num__` | 33 | 原始数值列，未变换 |

这些列**已经是编码后的数值**，无法从数值反推原始语义。需要说明某个特征时，
用 `feature_names.json` 里的列名；需要查看某套房子的原始取值时，向 Team 1 索取人类可读版。

> ⚠️ 两点语义提醒：`ord__` 列是**等级编码**，不是物理量，不宜表述为"每增加 1 单位"；
> 类别经 one-hot 后，单个 `nom__` 列只表示"是否属于该类别"。

### 3.2 目标变量

`SalePrice`，单位美元。原始分布**右偏（偏度 1.744）**，取 `log1p` 后偏度降到 **−0.015**，
因此建模通常在 `log1p` 空间进行，再 `expm1` 还原。

### 3.3 `row_id`

等于原始文件 `Order` 列的 1 基行号（1–2930），用于把任何结果回溯到具体房屋。
它是标识符，不是预测变量。

### 3.4 `fold`

`train_encoded.csv` 里的 5 折标记（1–5），按价格五分位分层生成，各折样本数 469/469/469/469/468。

---

## 4. 读取示例

```python
import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train_encoded.csv")
test  = pd.read_csv("test_encoded.csv")

NON_FEATURES = ["row_id", "SalePrice", "fold"]
FEATURES = [c for c in train.columns if c not in NON_FEATURES]   # 208 列

X_train, y_train = train[FEATURES], train["SalePrice"]
X_test,  y_test  = test[FEATURES],  test["SalePrice"]

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, np.log1p(y_train))

pred_log = model.predict(X_test)     # log 空间
pred     = np.expm1(pred_log)        # 美元空间
```

这段代码的输出可作为校核基准：**MAE = \$16,563，R²(log) = 0.8537**（特征为 208 列）。

用 `fold` 做交叉验证时，它是折标签而非划分定义，需经 `PredefinedSplit` 传入：

```python
from sklearn.model_selection import cross_val_score, PredefinedSplit
ps = PredefinedSplit(test_fold=train["fold"].values)
scores = cross_val_score(model, X_train, np.log1p(y_train), cv=ps,
                         scoring="neg_mean_absolute_error")   # 5 折 -MAE(log) ≈ 0.0855
```

---

## 5. 数据侧参考数字

同一套冻结划分上的结果，供判断量级是否合理：

| 模型 | MAE | R²(log) |
|---|---:|---:|
| 常数预测（log 均值） | $58,089 | −0.0002 |
| Ridge（特征标准化，α=1） | $16,563 | 0.8537 |
| Decision Tree | $22,393 | 0.8318 |
| Random Forest | $15,636 | 0.8950 |

`Overall Qual` 单特征即解释目标方差的 **63.9%**，因此这个数据集的 R² 起点较高。

### 5.1 指标空间会影响数字

同一个 Ridge 模型：

| 空间 | MAE | RMSE | R² |
|---|---:|---:|---:|
| log 空间 | 0.0849 | 0.1562 | 0.8537 |
| 美元空间 | $16,563 | $72,986 | 0.2306 |

R² 在两个空间差别很大（0.85 vs 0.23），引用时需要注明是按哪个空间计算的，
否则不同人报的数字会互相矛盾。

### 5.2 一个极端样本

测试集中 `row_id = 1499` 的实际成交价是 $160,000，而参考模型的预测是 $1,850,896。
这一套样本单独占全部平方误差的 **91.6%**，也是美元空间 RMSE（$72,986）远大于 MAE（$16,563）的原因。

原始记录显示这是**未完工即售出的期房**（`Sale Type = New`，`Sale Condition = Partial`，
`Year Built` 与 `Yr Sold` 同为 2008），同时有 5 个特征处于训练集的极端分位
（`Gr Liv Area` 5,642、`Total Bsmt SF` 6,110、`Garage Area` 1,418、`Lot Area` 63,887、
`Overall Qual` 10）。成交价反映的是半成品，而不是这些规格对应成的成品房。

按交易性质分解测试集误差：

| Sale Condition | n | MAE | 价格中位数 |
|---|---:|---:|---:|
| Normal | 467 | **$12,793** | $156,000 |
| Partial（期房） | 67 | $40,156 | $225,000 |
| Abnorml | 35 | $18,062 | $129,000 |
| Family | 10 | $29,243 | $130,500 |
| Alloca / AdjLand | 7 | ≈ $16,000 | — |

非正常交易占测试样本 **20.3%**，但贡献了 **94.3%** 的总平方误差。
即：误差的主要来源是交易性质，而不是房屋特征。

`Sale Condition` 属于交易元数据，Team 1 在清洗时将其剔除以避免泄漏，因此模型无法利用该信息。
若需按此维度分解误差，可用包里的 `sale_condition_lookup.csv`：

```python
lookup = pd.read_csv("sale_condition_lookup.csv").set_index("row_id")["SaleCondition"]
test_eval = test.assign(pred=pred,
                        abs_err=lambda d: (d.pred - d.SalePrice).abs())
test_eval["SaleCondition"] = test_eval.row_id.map(lookup)
print(test_eval.groupby("SaleCondition")["abs_err"].agg(["size", "mean"]))
```

---

## 6. 可追溯性

| 想查什么 | 在哪 |
|---|---|
| 13 项清洗决定及理由 | 项目内 `03_Docs/decision_log.md` |
| 原始数据与引用格式 | 项目内 `01_Data/raw/DATA_SOURCE.md` |
| 缺失值、重复值、相关性等统计 | 本包 `eda_summary.json`；细节在 `02_EDA/data_analysis.ipynb` |
| 某套房子的原始特征值 | 向 Team 1 索取 `train.csv` / `test.csv`（人类可读版） |

数据侧的编码规则、特征含义、异常情况，直接找 lian。

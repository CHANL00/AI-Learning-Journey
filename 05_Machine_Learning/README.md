# 05 机器学习入门 | Machine Learning

[返回总路线](../README.md) · [上一阶段的数据分析笔记](../04_Data%20analysis)

从 NumPy 数组创建接着学。先把一张原始表变成能回答问题的分析，再用同一份数据完成一次可复现的机器学习实验。以下按 **6 周、每周 4–6 小时**安排，可以按每周产出调整节奏。实践项目使用 UCI 共享单车数据，无需 GPU。

## 每周任务

| 周次 | 学什么 | 动手做什么 | 当周产出 |
| --- | --- | --- | --- |
| 第 1 周 | NumPy 索引、布尔筛选、`axis` 聚合。Pandas 读 CSV、检查类型和缺失、处理日期与重复行。 | 读取 `day.csv`，按日期排序并先划出末尾 20% 作为最终测试集。用前 80% 画每日租借趋势，比较工作日与非工作日的租借量。 | 一份 Notebook，包含清洗记录、一张趋势图和一张分组汇总表。 |
| 第 2 周 | `groupby`、`agg`、`merge` 和可视化。结合图表复习均值、中位数和分布。 | 按月份和工作日属性汇总开发数据。构造一张日期特征表，按日期合并，并检查合并前后行数与键是否唯一。 | 一份简短的探索报告，给出两条有图表支持的发现、一条局限，以及预测前能拿到的字段清单。 |
| 第 3 周 | 特征 `X`、标签 `y`、训练和验证、线性回归、平均绝对误差（MAE）。随练习理解损失和过拟合。 | 把前 80% 再按时间切成训练和验证两段。先用训练集租借量的中位数作基线，再用节假日和工作日两个二元特征训练线性回归，在同一验证集上比较。 | 一张基线与线性回归的验证 MAE 对照表，能用“平均差多少次租借”解释结果。 |
| 第 4 周 | 分类、逻辑回归、混淆矩阵、准确率和宏平均 F1。 | 用 scikit-learn 自带的鸢尾花数据做小练习。固定随机种子并分层切分，比较多数类基线和逻辑回归，找出最容易混淆的两类。 | 一份短 Notebook，记录切分方式、两种模型的指标和一张混淆矩阵。 |
| 第 5 周 | 独热编码、`ColumnTransformer`、`Pipeline`、随机森林和时间顺序交叉验证。 | 回到单车项目。在前 80% 数据上，用相同的 `TimeSeriesSplit(n_splits=3)` 折比较中位数基线、线性回归和随机森林。用 `OneHotEncoder(handle_unknown="ignore")` 对月份、星期做独热编码，让预处理在每折训练段内拟合，并固定随机森林的随机种子。 | 一张逐折验证 MAE 对照表，说明所选模型和选择依据。 |
| 第 6 周 | 最终测试、误差分析、复现。 | 固定模型和特征后，用前 80% 重新训练，在最后 20% 同时评估选定模型和基线。画实际值与预测值的对照图，分别查看工作日和非工作日的误差。 | 一份可从头运行的项目 Notebook 和简短 README，记录数据来源、环境、运行步骤、结果和局限。 |

## 阶段项目 · 预测次日共享单车租借量

用 `day.csv` 的每日总租借量 `cnt` 做预测目标。假设在前一晚进行预测，第一版只用已经知道的日历特征，例如月份、星期、节假日和工作日属性。把 `dteday` 用于排序和划分。`casual` 和 `registered` 相加就是 `cnt`，当天的实测天气也要等当天才有，这些列都从特征中排除。[UCI 数据集与字段说明](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

- **评估方式**。记录开发集与最终测试集的日期边界。开发集用于探索、选特征和选模型。最后一段数据留到方案确定后评估。[时间顺序交叉验证](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- **完成标准**。从下载数据开始可以复现结果。报告基线和所选模型在同一测试集上的 MAE，给出一张预测对照图，并结合误差最大的几天写出一个局限。若模型输给基线，也把结果和可能原因记下来。
- **进阶衔接**。想覆盖仓库总路线里的聚类，可在项目完成后用鸢尾花特征做一轮 K-means 可视化，比较缩放前后的分组。能独立解释数据切分、基线和误差后，再进入 PyTorch，从张量、自动求导和一个小型神经网络开始。

## 今天就做第一步

从 UCI 下载并解压 `day.csv`，放到当前 Notebook 所在目录。在 Notebook 使用的环境里准备 Pandas 和 Matplotlib，然后运行下面的代码，查看前 80% 数据的工作日分组摘要和每日趋势图。

```python
import matplotlib.pyplot as plt
import pandas as pd

rides = pd.read_csv("day.csv", parse_dates=["dteday"]).sort_values("dteday")
split_at = int(len(rides) * 0.8)
development = rides.iloc[:split_at]
summary = development.groupby("workingday")["cnt"].agg(["size", "mean", "median"])
print(summary)

development.plot(x="dteday", y="cnt", xlabel="Date", ylabel="Daily rentals")
plt.tight_layout()
plt.show()
```

## 学习资料

- 第 1–2 周看数组索引、汇总统计、表格合并和日期处理。[NumPy 基础教程](https://numpy.org/doc/stable/user/absolute_beginners.html) · [Pandas 入门教程](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
- 第 3–4 周看线性回归、逻辑回归与分类。[Google 机器学习速成课程（简体中文）](https://developers.google.com/machine-learning/crash-course?hl=zh-cn)
- 第 5–6 周查训练与预测接口、流水线，以及怎样避免在划分前拟合预处理。[scikit-learn 入门](https://scikit-learn.org/stable/getting_started.html) · [常见陷阱与数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)

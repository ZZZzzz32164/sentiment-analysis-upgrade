# 电商评论情感分析（升级版）

> 这是原 `sentiment_llm` 的升级分支。原项目验证的是「大模型打标 → 小模型训练」这条**链路能不能跑通**（300 条训练 / 200 条评估）；这个版本把它升级成一次**正经的文本分类实验**（6 万条全量、多模型对比、难例路由）。

## 三个升级点

| | 原来 | 现在 |
|---|---|---|
| 数据 | 只用 300 条训练 | 全部 6.27 万条，真实标签直接用 |
| 模型 | 只有朴素贝叶斯 | NB / 逻辑回归 / LinearSVC 三选一，5 折交叉验证 |
| 大模型的角色 | 打标造训练集 | 变成**难例路由**：小模型置信度低时才调用，算成本-效果账 |

**为什么这么改：**数据集自带真实标签，拿 LLM 去"造"标签等于把已有的信息丢掉。LLM 的正确用法是在小模型不确定的地方补位——这个权衡才是能讲出东西的地方。

## 目录

```
sentiment_upgrade/
├─ src/
│  ├─ features.py    # 清洗 + 分词（全流程共用，改一处全生效）
│  ├─ train_full.py  # 全量训练 + 5 折 CV + 混淆矩阵 + 落盘
│  └─ route_llm.py   # 难例路由：低置信度样本交给 DeepSeek
├─ data/             # 放 csv（不进 git）
└─ out/              # metrics.json / report.txt / preds.csv / model.joblib
```

## 1. 先跑通环境（30 秒）

```bash
python -m src.train_full --demo
```

用内置 240 条样例造的数据，只是验证依赖装没装、代码跑不跑得通。**这里的数字没有意义**（模板生成的，三个模型都是 100%）。

## 2. 跑真实数据

数据下载（GitHub，约 30MB）：

```
https://github.com/SophonPlus/ChineseNlpCorpus
→ datasets/online_shopping_10_cats/online_shopping_10_cats.zip
```

解压后放到 `data/`，列名是 `cat,label,review`：

```bash
python -m src.train_full --data data/online_shopping_10_cats.csv
```

跑完看 `out/metrics.json`，长这样：

```json
{
  "n_total": 62724,
  "n_train": 50179,
  "n_test": 12545,
  "n_features": 50000,
  "best": "svc",
  "models": {
    "nb":  {"cv_mean": 0.8xxx, "cv_std": 0.00x, "test_acc": 0.8xxx, "test_f1": 0.8xxx},
    "lr":  {...},
    "svc": {...}
  }
}
```

**预期**（以你实跑为准，别背我给的数）：NB 大概 0.85–0.87，逻辑回归和 LinearSVC 通常再高 2–4 个点。如果三个都在 0.9 以上，正常；如果你跑出 0.7 以下，说明数据列读错了。

## 3. 阶段二：难例路由（可选，但这是加分项）

```bash
set DEEPSEEK_API_KEY=sk-xxx
python -m src.route_llm --preds out/preds.csv --limit 200
```

输出 `out/route.json`：

```json
{"n_all": 12545, "n_hard": 6xx, "n_sent": 200, "n_fixed": 1xx, "fix_rate": 0.6x}
```

这就是简历上「成本与效果权衡」的**真实来源**：把 200 条最难样本送给大模型，修好了 60% 左右，换算成"多花 1.6% 的调用量，换来 +1 个百分点"。

没有 key 时先用 `--mock` 跑通代码路径。

## 跑完之后简历怎么写

三个数字一定要用**自己跑出来的**，填进这两条：

> · 基于公开数据集 online_shopping_10_cats（6.27 万条）做中文情感二分类，TF-IDF（1–2 元组、5 万维）+ 朴素贝叶斯/逻辑回归/LinearSVC 三模型对比，5 折交叉验证 __，评估集准确率 __；
> · 设计难例路由：小模型置信度低于阈值时转大模型判定，抽 200 条难例修复 __%，给出「多花 __% 调用量换 __ 个点」的成本-效果结论。

## 已知局限（面试可以主动讲）

- `tokenize` 里丢弃单字，会丢掉「差」「贵」这类单字情感词 → 改进方向：情感单字白名单，或字符级 n-gram
- 没有做超参搜索（alpha / C 都是默认值附近）
- 评估集是随机切分，同一商品的评论可能跨集合 → 严格做法是按商品分组切分
- 难例路由用的是"预测错了"当作难例，真实系统应该用 `predict_proba` 的置信度

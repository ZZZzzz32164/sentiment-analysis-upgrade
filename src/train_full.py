# -*- coding: utf-8 -*-
"""情感分析 - 全量实验（升级版主线）。

和原来 300 条弱监督实验的区别：
  原来：验证「大模型打标 -> 小模型训练」这条链路能不能跑通；
  现在：用数据集自带的真实标签，在 6 万条全量上做一次正经的文本分类实验。

用法：
  # 1) 先跑内置样例确认环境没问题（秒级）
  python -m src.train_full --demo

  # 2) 跑真实数据（在线购物评论 10 类，列名 cat,label,review）
  python -m src.train_full --data data/online_shopping_10_cats.csv

  # 3) 大模型难例路由（阶段二，需要 key）
  python -m src.train_full --data ... --route

输出：out/metrics.json、out/report.txt、out/preds.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import clean, tokenize  # noqa: E402

_NOUNS = ["手机", "耳机", "外套", "鞋", "背包", "水杯", "台灯", "椅子", "锅", "毛巾"]
_POS = ["质量很好，下次还会买", "东西不错，物流也快", "很喜欢，物超所值", "包装精美，体验很好",
        "用了一周，很满意", "性价比很高", "客服态度好，推荐购买", "颜色和图片一致",
        "很结实，做工细致", "送人很合适，好评", "手感舒服，细节到位", "第二次回购了，放心"]
_NEG = ["质量太差，用两天就坏", "发货慢，客服不理人", "和描述不符，退货了", "味道难闻，不会再买",
        "做工粗糙，有划痕", "尺寸不对，白买了", "包装破损，东西摔坏了", "根本不值这个价",
        "用了过敏，差评", "客服推卸责任，很失望", "用一个月就坏了，售后很麻烦", "色差严重，很失望"]

# 用「商品名 + 模板」拼出 120 条不重复的样例，仅用于快速验证环境
DEMO_DATA = [(f"{n}{t}", lab) for lab, tpl in ((1, _POS), (0, _NEG))
             for n in _NOUNS for t in tpl]


def load_frame(path, sample=None, seed=42):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col = next((cols[k] for k in ("review", "text", "content", "comment") if k in cols), None)
    label_col = next((cols[k] for k in ("label", "sentiment", "target", "y") if k in cols), None)
    if text_col is None or label_col is None:
        raise SystemExit(f"没找到文本列或标签列，实际列名：{list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "review", label_col: "label"})
    df = df.dropna()
    df["label"] = df["label"].astype(int)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed)
    return df


def load_demo():
    return pd.DataFrame(DEMO_DATA, columns=["review", "label"])


def build_models():
    return {
        "nb": MultinomialNB(alpha=1.0),
        "lr": LogisticRegression(max_iter=1000, C=1.0, n_jobs=None),
        "svc": LinearSVC(C=1.0, dual=False),
    }


def run(args):
    os.makedirs(args.out, exist_ok=True)

    # 1) 读数据
    df = load_demo() if args.demo else load_frame(args.data, args.sample, args.seed)
    print(f"[1/5] 原始样本 {len(df)} 条，正负比 {df.label.mean():.3f}")

    # 2) 清洗 + 去重（去重放在清洗之后，否则「带空格/不带空格」的重复逃过去）
    df["review"] = df["review"].map(clean)
    df = df[df["review"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["review"], keep="first").reset_index(drop=True)
    print(f"[2/5] 去空后 {before} 条，去重后 {len(df)} 条，正负比 {df.label.mean():.3f}")

    # 3) 分层切分：先切 holdout，再在训练集上做 CV，避免评估集信息泄漏
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"].to_numpy(), df["label"].to_numpy(),
        test_size=args.test_size, random_state=args.seed, stratify=df["label"].to_numpy(),
    )
    print(f"[3/5] 训练 {len(X_train)} / 评估 {len(X_test)}")

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        token_pattern=None,
    )
    Xtr = vectorizer.fit_transform(X_train)   # 只在训练集上 fit
    Xte = vectorizer.transform(X_test)
    print(f"[4/5] 词表 {len(vectorizer.vocabulary_)} 维，训练矩阵 {Xtr.shape}")

    # 4) 训练 + 5 折交叉验证 + 评估集
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    result = {"n_total": int(len(df)), "n_train": len(X_train), "n_test": len(X_test),
              "n_features": int(Xtr.shape[1]), "pos_ratio": float(df.label.mean()),
              "models": {}}

    for name, model in build_models().items():
        cv_scores = []
        for tr_idx, va_idx in cv.split(Xtr, y_train):
            m = build_models()[name]
            m.fit(Xtr[tr_idx], y_train[tr_idx])
            cv_scores.append(accuracy_score(y_train[va_idx], m.predict(Xtr[va_idx])))
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        cm = confusion_matrix(y_test, pred)
        result["models"][name] = {
            "cv_mean": round(float(np.mean(cv_scores)), 4),
            "cv_std": round(float(np.std(cv_scores)), 4),
            "test_acc": round(float(accuracy_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
        }
        print(f"  {name:>4}: CV {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f} | "
              f"测试 {result['models'][name]['test_acc']:.4f} | "
              f"混淆矩阵 {cm.tolist()}")

    # 5) 落盘：选测试集最好的模型出报告
    best = max(result["models"], key=lambda k: result["models"][k]["test_acc"])
    result["best"] = best
    best_model = build_models()[best]
    best_model.fit(Xtr, y_train)
    pred = best_model.predict(Xte)

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存模型，供 route_llm.py 做难例路由
    import joblib
    joblib.dump({"model": best_model, "vectorizer": vectorizer},
                os.path.join(args.out, "model.joblib"))

    report = [f"样本 {result['n_total']}（训练 {result['n_train']} / 评估 {result['n_test']}）",
              f"词表 {result['n_features']} 维，最优模型 {best}",
              "", classification_report(y_test, pred, digits=4, zero_division=0)]
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    pd.DataFrame({"review": X_test, "label": y_test, "pred": pred}).to_csv(
        os.path.join(args.out, "preds.csv"), index=False, encoding="utf-8-sig")

    print(f"[5/5] 最优 {best}，结果写入 {args.out}/")
    print("\n".join(report))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV 路径（需含 review/label 列或会自动识别）")
    ap.add_argument("--demo", action="store_true", help="用内置样例跑通环境")
    ap.add_argument("--sample", type=int, default=None, help="只取 N 条（快速试跑）")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-features", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    if not args.demo and not args.data:
        raise SystemExit("要么 --data 指定 CSV，要么 --demo")
    run(args)


if __name__ == "__main__":
    main()


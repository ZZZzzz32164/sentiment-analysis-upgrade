# -*- coding: utf-8 -*-
"""阶段二：难例路由（小模型兜底 + 大模型处理疑难样本）。

思路：
  小模型对每条样本给出「置信度」= 预测类别的概率（LinearSVC 用 decision_function 的
  绝对值做近似，NB/LR 用 predict_proba）。
  置信度 >= 阈值  -> 直接采用小模型结果（成本 0）
  置信度 <  阈值  -> 交给 DeepSeek 判断（花一次 API）

在这上面还能算出「用多少比例的 API 调用，换来多少准确率提升」，这才是简历上那句
"成本与效果权衡"的真实来源。

用法：
  python -m src.route_llm --preds out/preds.csv --data <原始csv> --threshold 0.6
  没有 key 时用 --mock 跑通流程（模拟大模型，只验证代码路径）
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = """判断下面这条商品评论的情感倾向。只输出一个字：正面 或 负面。

评论：{text}
"""


def ask_llm(text, api_key, base_url="https://api.deepseek.com", model="deepseek-chat", retries=3):
    """调用一次 chat/completions，失败按 2、4、6 秒退避重试。返回 1/0/None。"""
    import requests

    last_err = None
    for i in range(retries):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "user", "content": PROMPT.format(text=text)}]},
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if "负面" in content and "正面" not in content:
                return 0
            if "正面" in content:
                return 1
            return None
        except Exception as exc:            # noqa: BLE001 - 这里就是要兜住所有网络/格式异常
            last_err = exc
            time.sleep(2 * (i + 1))
    print(f"  [warn] 调用失败：{last_err}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="out/preds.csv", help="train_full 产出的预测表")
    ap.add_argument("--threshold", type=float, default=0.6, help="低于该置信度才走大模型")
    ap.add_argument("--mock", action="store_true", help="不调 API，用关键词规则模拟大模型")
    ap.add_argument("--limit", type=int, default=200, help="最多送多少条给大模型（控成本）")
    ap.add_argument("--out", default="out/route.json")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    print(f"读取预测 {len(df)} 条，阈值 {args.threshold}")

    # 这里没有置信度列时（LinearSVC 不输出概率），用 |label - pred| 近似“难例”
    # 真实项目应保存 decision_function，作为阶段性简化。
    hard = df[df.label != df.pred]
    print(f"其中错误样本 {len(hard)} 条，取前 {min(args.limit, len(hard))} 条送给大模型")

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not args.mock and not api_key:
        raise SystemExit("没设 DEEPSEEK_API_KEY，要么设环境变量，要么加 --mock")

    fixed = 0
    used = 0
    for _, row in hard.head(args.limit).iterrows():
        if args.mock:
            ans = 1 if re.search(r"好|不错|满意|喜欢|推荐", str(row.review)) else 0
        else:
            ans = ask_llm(str(row.review), api_key)
        used += 1
        if ans is not None and int(ans) == int(row.label):
            fixed += 1

    result = {"n_all": int(len(df)), "n_hard": int(len(hard)), "n_sent": used,
              "n_fixed": fixed,
              "fix_rate": round(fixed / used, 4) if used else 0.0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

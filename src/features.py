# -*- coding: utf-8 -*-
"""文本清洗与分词：全量训练和弱监督实验共用这一份，保证口径一致。

设计原则：
1. 每个函数都是纯函数（输入 -> 输出），方便单测和复用；
2. 清洗规则集中在这里，改一处全流程生效；
3. 不在这里写任何 IO 或模型逻辑。
"""
import re

import jieba

# 内置精简停用词（不依赖外部文件，保证 clone 下来就能跑）
STOPWORDS = set(
    """
的 了 和 是 就 都 而 及 与 着 或 一个 没有 我们 你们 他们 它们 这个 那个 这些 那些
自己 什么 怎么 因为 所以 但是 而且 如果 虽然 然后 还是 已经 可以 这样 那样 时候
非常 特别 真的 感觉 觉得 知道 现在 一下 一些 有点 一点 还是 就是 不是 不能 不会
""".split()
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")


def clean(text):
    """规范化：去 URL、去数字、去 HTML 标签、压缩空白。

    去数字是因为电商评论里的数字（12345、2024）几乎不含情感信号，
    还会把词表撑大；如果做价格类任务就不该去。
    """
    s = str(text)
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = NUM_RE.sub(" ", s)
    s = s.replace("\u3000", " ")
    s = SPACE_RE.sub(" ", s)
    return s.strip()


def tokenize(text):
    """jieba 分词 + 去停用词 + 去单字。

    注意：丢弃 len(w) == 1 会丢掉「差」「贵」「脏」这类单字情感词，
    两字词（差评、垃圾、不值）能救回一部分。改进方向见 README。
    """
    return [w for w in jieba.cut(text) if w.strip() and w not in STOPWORDS and len(w) > 1]

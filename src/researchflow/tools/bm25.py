"""自实现 BM25（k1=1.5, b=0.75）：词法检索路，索引在内存里由块列表构建（T-030）。

分词：英文小写词（去掉长度 1 的词）+ 中文字二元组（单字成段时保留该字），
与 `core/embedding.py` 的 HashEmbedder 保持一致，两路才能公平对比。

IDF 用 BM25+ 风格的 `log(1 + (N - df + 0.5) / (df + 0.5))`，避免负权重。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """英文小写词（长度 ≥ 2）+ 中文字二元组（单字成段时保留该字）。"""
    tokens: list[str] = []
    for segment in _TOKEN_RE.findall(text.lower()):
        if "\u4e00" <= segment[0] <= "\u9fff":
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        elif len(segment) >= 2:
            tokens.append(segment)
    return tokens


@dataclass
class BM25Index:
    """内存 BM25 索引：`docs` 是与文档一一对应的分词结果与长度缓存。"""

    k1: float = K1
    b: float = B
    _doc_tokens: list[list[str]] = field(default_factory=list)
    _doc_lens: list[int] = field(default_factory=list)
    _df: dict[str, int] = field(default_factory=dict)
    _avg_len: float = 0.0

    @classmethod
    def build(cls, docs: list[str], *, k1: float = K1, b: float = B) -> BM25Index:
        """由文档列表（块文本）构建索引。"""
        index = cls(k1=k1, b=b)
        index._doc_tokens = [tokenize(doc) for doc in docs]
        index._doc_lens = [len(tokens) for tokens in index._doc_tokens]
        total = sum(index._doc_lens)
        index._avg_len = total / len(docs) if docs else 0.0
        df: dict[str, int] = {}
        for tokens in index._doc_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        index._df = df
        return index

    def __len__(self) -> int:
        return len(self._doc_tokens)

    def score(self, query: str, doc_index: int) -> float:
        """BM25 打分公式：Σ idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d|/avg))。"""
        tokens = self._doc_tokens[doc_index]
        doc_len = self._doc_lens[doc_index]
        norm = self.k1 * (1 - self.b + self.b * doc_len / self._avg_len) if self._avg_len else 0.0
        n_docs = len(self._doc_tokens)
        total = 0.0
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token in set(tokenize(query)):
            df = self._df.get(token)
            if df is None:
                continue
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            total += idf * tf * (self.k1 + 1) / (tf + norm)
        return total

    def rank(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        """全部文档按得分降序排；同分按文档序号稳定排序，返回前 k（None = 全部）。"""
        scored = [(i, self.score(query, i)) for i in range(len(self._doc_tokens))]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored if k is None else scored[: max(k, 0)]

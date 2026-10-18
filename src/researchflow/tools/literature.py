"""① 检索数据源：Gate 1 使用本地语料，避免外部网络导致 e2e 不稳定。

两个入口：

- `LocalCorpusRetriever`：`core/retrieval.py::Retriever` 的本地语料实现，
  ① 与 `search_literature` 工具都通过它检索；
- `search_local()`：旧签名，Gate 2 删除。

`ref_id` 用内容哈希而不是随机 ID：同一篇文献每次运行必须得到同一个 ID，
否则产物 checksum 不可复现，以后检索评测（Recall@k）的人工标注也无处对照。
"""

from __future__ import annotations

import hashlib
import re

from ..core.config import Settings
from ..core.errors import ConfigError
from ..core.retrieval import Retriever
from ..core.schemas import LiteratureRef

# 标题规范化用：合并连续空白（含换行），否则同一篇文献换个写法就换 ID
_WHITESPACE_RE = re.compile(r"\s+")


LOCAL_CORPUS: list[dict] = [
    {
        "title": "数据质量对实证研究结论的影响",
        "source": "本地语料",
        "year": 2021,
        "url": "",
        "summary": "讨论缺失值与异常值如何在回归分析中放大结论偏差。",
        "keywords": ["数据质量", "缺失值", "实证", "回归"],
    },
    {
        "title": "可复现研究工作流：从原始数据到报告",
        "source": "本地语料",
        "year": 2022,
        "url": "",
        "summary": "提出固定随机种子、锁定依赖与产物清单的可复现实践。",
        "keywords": ["可复现", "工作流", "科研", "产物"],
    },
    {
        "title": "线性回归模型的稳健性检验",
        "source": "本地语料",
        "year": 2020,
        "url": "",
        "summary": "给出小样本下线性模型拟合指标的解释边界。",
        "keywords": ["回归", "线性", "模型", "指标"],
    },
    {
        "title": "缺失值处理方法综述",
        "source": "本地语料",
        "year": 2019,
        "url": "",
        "summary": "比较删除、均值插补与模型插补在不同缺失率下的表现。",
        "keywords": ["缺失值", "清洗", "插补", "数据质量"],
    },
    {
        "title": "科研数据清洗的自动化实践",
        "source": "本地语料",
        "year": 2023,
        "url": "",
        "summary": "总结去重、类型归一与阈值校验的流水线化做法。",
        "keywords": ["清洗", "数据质量", "自动化"],
    },
    {
        "title": "特征选择与模型解释",
        "source": "本地语料",
        "year": 2021,
        "url": "",
        "summary": "说明在有限特征下如何解释系数与拟合优度。",
        "keywords": ["特征", "模型", "解释"],
    },
    {
        "title": "小样本研究的统计功效",
        "source": "本地语料",
        "year": 2018,
        "url": "",
        "summary": "提示小样本结论需要标注局限性与外推风险。",
        "keywords": ["样本", "统计", "局限"],
    },
    {
        "title": "研究流程编排与人工审核",
        "source": "本地语料",
        "year": 2024,
        "url": "",
        "summary": "讨论在自动流程中固定插入人工审核节点的必要性。",
        "keywords": ["审核", "流程", "人机协同", "科研"],
    },
]


def make_ref_id(source: str, title: str, year: int | None) -> str:
    """按"来源 + 规范化标题 + 年份"生成稳定 ID。

    规范化：去首尾空白、合并连续空白、转小写；年份缺失时该段留空。
    形状 `{source}:{sha256 前 12 位}`，例如 `local:3f9a0c1b2d4e`。
    """
    normalized = _WHITESPACE_RE.sub(" ", title.strip()).lower()
    payload = f"{normalized}|{'' if year is None else year}"
    return f"{source}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _score_corpus(query: str) -> list[tuple[int, dict]]:
    """按关键词命中数给语料打分并排序；只保留命中的条目。

    排序键 `(-score, -year, title)`，保证同一查询的输出顺序稳定。
    """
    scored = [
        (sum(1 for keyword in entry["keywords"] if keyword in query), entry)
        for entry in LOCAL_CORPUS
    ]
    scored.sort(key=lambda item: (-item[0], -int(item[1]["year"]), item[1]["title"]))
    return [(score, entry) for score, entry in scored if score > 0]


class LocalCorpusRetriever:
    """`Retriever` 的本地语料实现：确定性打分，只返回命中的文献。"""

    @property
    def name(self) -> str:
        return "local"

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        """按关键词命中数检索本地语料，最多 `k` 条；没命中返回空列表。"""
        return [
            LiteratureRef(
                ref_id=make_ref_id("local", entry["title"], entry["year"]),
                title=entry["title"],
                source=entry["source"],
                year=entry["year"],
                url=entry["url"],
                summary=entry["summary"],
                score=float(score),
                retriever=self.name,
            )
            for score, entry in _score_corpus(query)[:k]
        ]


def build_retriever(settings: Settings) -> Retriever:
    """按配置造检索器：`retrieval.backend` 为 `local`（默认）或 `openalex`。

    后端值由 `load_settings` 从 `settings.yaml` 的 `retrieval.backend` 读入、可被
    环境变量 `RF_RETRIEVER` 覆盖；非法值抛 `ConfigError`。默认仍是 `local`，现有
    测试、Eval、可复现测试都不受影响。
    """
    backend = settings.retrieval.backend
    if backend == "local":
        return LocalCorpusRetriever()
    if backend == "openalex":
        from .openalex import OpenAlexRetriever

        return OpenAlexRetriever(settings)
    raise ConfigError(f"retrieval.backend 取值非法: {backend!r}（只能是 local 或 openalex）")


def search_local(topic: str, *, limit: int = 6) -> list[LiteratureRef]:
    """按关键词命中数排序检索本地语料，保证确定性输出。

    已由 `LocalCorpusRetriever` 取代，Gate 2 删除。原先"0 命中返回全部语料"的
    兜底一并去掉：调用方必须能区分"没命中"和"命中了不相关的"。
    """
    return LocalCorpusRetriever().search(topic, k=limit)

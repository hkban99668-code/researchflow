"""检索器契约：Gate 2 换真实学术 API、Gate 3 换向量库时，① 与工具都不用改。

契约只管"找"这一件事：给查询与上限，返回文献列表。**给模型看多少**（摘要截断）
是上下文管理，留在节点里（`nodes/retrieval.py` 的 `max_ref_chars`）。

分层与 LLM 一致：`Retriever` 定义在 `core`，实现放在上层（`tools/literature.py`
的 `LocalCorpusRetriever`），通过 `RunContext.retriever` 注入，`core` 不反向依赖上层。
"""

from __future__ import annotations

from typing import Protocol

from .schemas import LiteratureRef


class Retriever(Protocol):
    """文献检索器契约。

    实现方必须遵守的约定：

    - 结果按相关度降序；
    - **只返回真正命中的结果**，没命中就返回空列表。不能用"全部语料"凑数，
      否则调用方分不清"没找到"与"找到了但不相关"，检索 Agent 也不会去改写查询；
    - `k` 是上限不是下限，返回条数可以少于 `k`；
    - 同一输入多次调用结果相同：Gate 1 的实现必须确定性，否则产物 checksum 不可复现。
    """

    @property
    def name(self) -> str:
        """检索器名，写进 `LiteratureRef.retriever`，例如 `local` / `openalex`。"""

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        """按查询检索文献，最多返回 `k` 条；没命中返回空列表。"""
        ...

"""注入检索器（A1）：包装任意 `Retriever`，确定性地在指定位置插入攻击文本。

确定性：同输入同输出——每次 `search()` 都对内部结果做深拷贝再改写，不污染内部
检索器的状态，重复调用不会把攻击文本越叠越多；只改指定位置那一条的 `summary`
（攻击文本前置于原摘要），其余条目与顺序原样保留。

被注入的 ref_id 每次调用都记进 `last_injected_id`（该次结果里被改写的文献）；
结果不足 `position` 条时那次调用不注入、记 None（攻击落空，由评测按"未注入"
口径统计）。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from researchflow.core.schemas import LiteratureRef


class InjectingRetriever:
    """在 `position`（1 起）那条结果的摘要前插入攻击文本的检索器包装。"""

    def __init__(self, inner: Any, attack_text: str, *, position: int) -> None:
        if position < 1:
            raise ValueError(f"position 从 1 起，收到 {position}")
        self.inner = inner
        self.attack_text = attack_text
        self.position = position
        # 每次 search() 后可读：该次结果里被注入的 ref_id；未注入（结果不足）为 None。
        self.last_injected_id: str | None = None

    @property
    def name(self) -> str:
        return f"injecting({self.inner.name})"

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        results = [deepcopy(ref) for ref in self.inner.search(query, k=k)]
        if len(results) < self.position:
            self.last_injected_id = None
            return results
        target = results[self.position - 1]
        target.summary = self.attack_text + target.summary
        self.last_injected_id = target.ref_id
        return results

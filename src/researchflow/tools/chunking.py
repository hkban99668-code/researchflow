"""切块：把文献（标题 + 摘要）按句子窗口切成确定性的块（T-030）。

- `chunk_id = "{ref_id}#c{序号}"`，序号从 0 起，同一文献每次切分结果相同；
- 窗口约 400 字符（含标点），相邻窗口重叠 1 句；
- 单句超过窗口长度时按字符硬切；
- 摘要为空时只有标题一块；
- 所有块文本都是原文的连续子串，拼回后覆盖原文（A2）。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..core.schemas import LiteratureRef

# 窗口目标长度（字符）：标题 + 摘要级别的块粒度
CHUNK_CHARS = 400

# 句子分隔：中英文句末标点（标点保留在前一句末尾）
_SENTENCE_RE = re.compile(r"[^。！？!?\n]*[。！？!?\n]|[^。！？!?\n]+$")


class Chunk(BaseModel):
    """一块可检索的文本：属于某篇文献，`chunk_id` 全局稳定。"""

    chunk_id: str = Field(description='形如 "{ref_id}#c{序号}"')
    ref_id: str
    text: str


def _sentences(doc: str) -> list[str]:
    """把整篇文档切成带标点的句子；硬切超长句，保证每段 ≤ CHUNK_CHARS。"""
    pieces: list[str] = []
    for raw in _SENTENCE_RE.findall(doc):
        sentence = raw
        if not sentence:
            continue
        # 单句超过窗口长度时按字符硬切（不丢字）
        while len(sentence) > CHUNK_CHARS:
            pieces.append(sentence[:CHUNK_CHARS])
            sentence = sentence[CHUNK_CHARS:]
        if sentence:
            pieces.append(sentence)
    return pieces


def _windows(sentences: list[str]) -> list[str]:
    """贪心句窗口：装到约 CHUNK_CHARS 为止，下一窗口以上一句开头（重叠 1 句）。"""
    windows: list[str] = []
    start = 0
    while start < len(sentences):
        text = ""
        end = start
        while end < len(sentences) and (not text or len(text) + len(sentences[end]) <= CHUNK_CHARS):
            text += sentences[end]
            end += 1
        if not text:  # 单句就超长的极端情形已被 _sentences 硬切，这里兜底防死循环
            text = sentences[start]
            end = start + 1
        windows.append(text)
        if end >= len(sentences):
            break
        # 重叠 1 句；窗口只装下 1 句时 start 不回退，保证必然前进
        start = end - 1 if end - 1 > start else end
    return windows


def chunk_ref(ref: LiteratureRef) -> list[Chunk]:
    """把一篇文献切成块：标题在前，摘要在后；摘要为空时只有标题一块。"""
    doc = ref.title if not ref.summary else f"{ref.title}\n{ref.summary}"
    chunks = [
        Chunk(chunk_id=f"{ref.ref_id}#c{i}", ref_id=ref.ref_id, text=text)
        for i, text in enumerate(_windows(_sentences(doc)))
    ]
    return chunks


def chunk_refs(refs: list[LiteratureRef]) -> list[Chunk]:
    """按输入顺序切全部文献，拼成一份块列表。"""
    return [chunk for ref in refs for chunk in chunk_ref(ref)]

"""T-030 切块：确定性、chunk_id 稳定、空摘要、长句硬切、拼回覆盖原文（A2）。"""

from __future__ import annotations

import re

from researchflow.core.schemas import LiteratureRef
from researchflow.tools.chunking import CHUNK_CHARS, chunk_ref, chunk_refs


def _ref(
    summary: str = "", title: str = "缺失值插补方法综述", ref_id: str = "stub:r1"
) -> LiteratureRef:
    return LiteratureRef(ref_id=ref_id, title=title, summary=summary)


def test_chunk_deterministic_and_ids_stable():
    ref = _ref(summary="第一句。第二句。第三句。")
    first = chunk_ref(ref)
    second = chunk_ref(ref)
    assert first == second, "同一切块函数对同一文献结果必须完全一致"
    assert [chunk.chunk_id for chunk in first] == [f"stub:r1#c{i}" for i in range(len(first))]


def test_chunk_id_varies_by_ref():
    a = chunk_ref(_ref(ref_id="stub:ra"))
    b = chunk_ref(_ref(ref_id="stub:rb"))
    assert a[0].chunk_id == "stub:ra#c0"
    assert b[0].chunk_id == "stub:rb#c0"


def test_empty_summary_only_title_chunk():
    ref = _ref(summary="")
    chunks = chunk_ref(ref)
    assert len(chunks) == 1
    assert chunks[0].text == ref.title


def test_long_sentence_hard_cut():
    long_text = "长" * (CHUNK_CHARS * 2 + 100)
    ref = _ref(summary=long_text)
    chunks = chunk_ref(ref)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.text) <= CHUNK_CHARS, f"每块不超过窗口长度: {len(chunk.text)}"
    # 拼回覆盖原文：硬切场景每块都是原文连续子串，且首尾衔接无缝
    doc = f"{ref.title}\n{long_text}"
    pos = 0
    for chunk in chunks:
        assert doc.startswith(chunk.text, pos) or chunk.text in doc
        pos = doc.find(chunk.text, pos) + len(chunk.text)
    assert pos == len(doc), "所有块拼回后覆盖原文到最后一个字符"


def test_windows_cover_original_text():
    summary = "。".join(f"这是第{i}句话关于缺失值插补" for i in range(40)) + "。"
    ref = _ref(summary=summary)
    chunks = chunk_ref(ref)
    assert len(chunks) > 1
    # 每块都是原文的连续子串
    doc = f"{ref.title}\n{summary}"
    for chunk in chunks:
        assert chunk.text in doc
    # 相邻窗口共享开头（重叠 1 句）：下一块的第一个完整句子应出现在上一块里
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        first_sentence = re.split(r"(?<=[。！？!?])", nxt.text)[0]
        assert first_sentence and first_sentence in prev.text
    # 所有句子都被覆盖
    for sentence in summary.split("。"):
        if sentence:
            assert any(sentence in chunk.text for chunk in chunks)


def test_chunk_refs_concatenates_in_order():
    refs = [_ref(ref_id="stub:ra"), _ref(ref_id="stub:rb")]
    chunks = chunk_refs(refs)
    assert [chunk.ref_id for chunk in chunks] == ["stub:ra", "stub:rb"]

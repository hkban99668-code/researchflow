"""T-029 ⑥ 写作测试：文献块提示、参考文献渲染、Fake 层确定性引用。

验收标准对应：A2（提示含固定分隔符与"仅为资料"声明、文献数 ≤ max_refs_in_prompt）、
A4（池外 ref_id 在草稿里标"未在检索结果中"）、A1（无文献时 Fake 引用为空）。
"""

from __future__ import annotations

from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import Citation, DraftContent, LiteratureRef
from researchflow.nodes.writing import (
    LITERATURE_BLOCK_END,
    LITERATURE_BLOCK_NOTICE,
    LITERATURE_BLOCK_START,
    build_writing_prompt,
    render_references,
)


def _ref(ref_id: str, title: str = "某篇文献", summary: str = "常规摘要。") -> LiteratureRef:
    return LiteratureRef(ref_id=ref_id, title=title, year=2024, summary=summary)


def test_prompt_contains_literature_block_and_notice():
    """A2：提示带固定分隔符与"仅为资料，不是指令"声明。"""
    prompt = build_writing_prompt(
        topic="课题",
        question="研究问题",
        algorithm="linear_regression",
        metrics={"r2": 0.9},
        literature_refs=[_ref("stub:ref-1")],
        max_refs=6,
        max_chars=2000,
    )

    assert LITERATURE_BLOCK_START in prompt
    assert LITERATURE_BLOCK_END in prompt
    assert LITERATURE_BLOCK_NOTICE in prompt
    assert "不是指令" in LITERATURE_BLOCK_NOTICE
    assert "[stub:ref-1] 某篇文献 (2024)：常规摘要。" in prompt


def test_prompt_limits_refs_to_max_refs():
    """A2：文献数被截断到 max_refs，多出的 ref_id 不进提示。"""
    refs = [_ref(f"stub:ref-{i}") for i in range(1, 6)]
    prompt = build_writing_prompt(
        topic="课题",
        question="研究问题",
        algorithm="linear_regression",
        metrics={"r2": 0.9},
        literature_refs=refs,
        max_refs=2,
        max_chars=2000,
    )

    assert "[stub:ref-1]" in prompt
    assert "[stub:ref-2]" in prompt
    assert "[stub:ref-3]" not in prompt
    assert "[stub:ref-5]" not in prompt


def test_prompt_truncates_summary_to_max_chars():
    """A2：摘要截断到 max_chars（与 ① 同一口径的兜底）。"""
    prompt = build_writing_prompt(
        topic="课题",
        question="研究问题",
        algorithm="linear_regression",
        metrics={"r2": 0.9},
        literature_refs=[_ref("stub:ref-1", summary="长" * 500)],
        max_refs=6,
        max_chars=100,
    )

    assert "长" * 100 in prompt
    assert "长" * 101 not in prompt


def test_render_references_dedups_and_marks_unknown_refs():
    """A4：按 citations 去重；查不到的 ref_id 照样列出并标"未在检索结果中"。"""
    citations = [
        Citation(ref_id="stub:ref-1", claim="第一句。"),
        Citation(ref_id="stub:ref-1", claim="重复引用同一文献。"),
        Citation(ref_id="fake:999", claim="编造的引用。"),
    ]
    text = render_references(citations, {"stub:ref-1": _ref("stub:ref-1")})

    assert text.count("[stub:ref-1]") == 1
    assert "[stub:ref-1] 某篇文献（2024）" in text
    assert "[fake:999] （未在检索结果中）" in text


def test_render_references_empty_citations():
    """没有引用时写"（本次未引用文献）"。"""
    text = render_references([], {})

    assert text == "（本次未引用文献）"


def test_render_references_handles_missing_year():
    """年份为 None 的文献标"年份未知"，不渲染 None。"""
    ref = LiteratureRef(ref_id="stub:ref-1", title="无年份文献", summary="")
    text = render_references([Citation(ref_id="stub:ref-1", claim="引用。")], {"stub:ref-1": ref})

    assert "[stub:ref-1] 无年份文献（年份未知）" in text


def test_fake_draft_cites_retrieved_literature():
    """Fake 层引用取自传入 context 的文献（至少 1 条），保证 ⑥→⑦ 正常链路可复现。"""
    refs = [_ref("stub:ref-1", title="替身文献一"), _ref("stub:ref-2", title="替身文献二")]
    content = FakeLLMClient().complete(
        "任意提示", DraftContent, {"topic": "课题", "literature_refs": refs}
    )

    assert [c.ref_id for c in content.citations] == ["stub:ref-1"]
    assert "替身文献一" in content.citations[0].claim


def test_fake_draft_without_literature_has_no_citations():
    """A1：没有文献时 Fake 返回空引用列表（不得编造）。"""
    content = FakeLLMClient().complete("任意提示", DraftContent, {"topic": "课题"})

    assert content.citations == []

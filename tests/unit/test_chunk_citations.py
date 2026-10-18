"""T-030 块级引用校验：合法 / 不属于该文献 / 不存在三种情形，旧数据行为不变（A8）。"""

from __future__ import annotations

from researchflow.tools.rules_engine import SubmissionRules, evaluate_submission

_RULES = SubmissionRules(
    required_sections=["背景"],
    required_artifacts=["draft"],
    owner_steps={
        "section": "writing",
        "draft_min_chars": "writing",
        "forbidden_tokens": "writing",
        "artifact:draft": "writing",
        "variables_in_dataset": "extraction",
        "citations_subset": "writing",
        "citation_has_claim": "writing",
    },
)


def _evaluate(citations: list[dict], valid_chunk_ids: list[str] | set[str] | None):
    return evaluate_submission(
        _RULES,
        metrics={},
        draft_text="背景\n" + "x" * 200,
        artifacts={"draft": True},
        citations=citations,
        valid_ref_ids=["stub:r1", "stub:r2"],
        valid_chunk_ids=valid_chunk_ids,
    )


def test_valid_chunk_citation_passes():
    citations = [{"ref_id": "stub:r1", "claim": "有一句话", "chunk_id": "stub:r1#c0"}]
    report = _evaluate(citations, valid_chunk_ids={"stub:r1#c0", "stub:r2#c1"})
    check = next(c for c in report.checks if c.name == "citations_subset")
    assert check.passed


def test_chunk_not_owned_by_ref_fails():
    # chunk_id 属于 r2，却挂在 r1 的引用上
    citations = [{"ref_id": "stub:r1", "claim": "有一句话", "chunk_id": "stub:r2#c0"}]
    report = _evaluate(citations, valid_chunk_ids={"stub:r2#c0"})
    check = next(c for c in report.checks if c.name == "citations_subset")
    assert not check.passed
    assert "不属于所引文献" in check.detail


def test_chunk_not_in_index_fails():
    citations = [{"ref_id": "stub:r1", "claim": "有一句话", "chunk_id": "stub:r1#c9"}]
    report = _evaluate(citations, valid_chunk_ids={"stub:r1#c0"})
    check = next(c for c in report.checks if c.name == "citations_subset")
    assert not check.passed
    assert "不在本次索引块集合中" in check.detail


def test_chunk_id_without_index_set_fails_closed():
    # 没传 valid_chunk_ids（None）：带 chunk_id 的引用无法通过校验（宁可拦下）
    citations = [{"ref_id": "stub:r1", "claim": "有一句话", "chunk_id": "stub:r1#c0"}]
    report = _evaluate(citations, valid_chunk_ids=None)
    check = next(c for c in report.checks if c.name == "citations_subset")
    assert not check.passed


def test_old_data_without_chunk_id_unchanged():
    # 旧数据（chunk_id 为 None / 缺字段）行为不变：只查 ref_id 在池内
    citations = [
        {"ref_id": "stub:r1", "claim": "有一句话"},
        {"ref_id": "stub:r2", "claim": "另一句话", "chunk_id": None},
    ]
    report = _evaluate(citations, valid_chunk_ids=set())
    check = next(c for c in report.checks if c.name == "citations_subset")
    assert check.passed
    assert report.citation_stats.invalid == 0


def test_citation_object_form_also_checked():
    from researchflow.core.schemas import Citation

    report = _evaluate(
        [Citation(ref_id="stub:r1", claim="有一句话", chunk_id="stub:r1#c0")],
        valid_chunk_ids={"stub:r1#c0"},
    )
    check = next(c for c in report.checks if c.name == "citations_subset")
    assert check.passed

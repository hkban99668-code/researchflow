"""T-029 引用硬规则测试：citations_subset / citation_has_claim / citation_stats。

验收标准对应：A3（两条规则通过/失败与逐条信息）、A1（旧 checkpoint 字典形态兼容）、
A6（total=0 时不除零）。
"""

from __future__ import annotations

from researchflow.core.schemas import Citation, CitationStats, DraftContent
from researchflow.tools.rules_engine import evaluate_submission

# 与既有 ⑦ 测试同款：草稿本身满足全部既有检查，只叠加引用维度。
GOOD_DRAFT = (
    "# Demo\n\n## 背景\n" + "研究背景与问题来源描述。" * 20 + "\n\n## 方法\n"
    "使用固定参数与固定随机种子的单算法回归。\n\n"
    "## 结果\nDIRECT 计算给出拟合指标与图表。\n\n## 局限\n样本量有限，结论待扩展。\n"
)
GOOD_ARTIFACTS = {"cleaned_data": True, "chart": True, "draft": True}


def _evaluate(submission_rules, *, citations=None, valid_ref_ids=None):
    """带引用参数跑一遍 ⑦，返回报告。"""
    return evaluate_submission(
        submission_rules,
        metrics={"r2": 0.95},
        draft_text=GOOD_DRAFT,
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
        citations=citations,
        valid_ref_ids=valid_ref_ids,
    )


def test_citations_subset_passes_when_all_refs_retrieved(submission_rules):
    """A3：全部 ref_id 都在检索结果里 → 通过，detail 说明全部合法。"""
    citations = [
        Citation(ref_id="stub:ref-1", claim="第一条合法引用。"),
        Citation(ref_id="stub:ref-2", claim="第二条合法引用。"),
    ]
    report = _evaluate(
        submission_rules,
        citations=citations,
        valid_ref_ids=["stub:ref-1", "stub:ref-2"],
    )

    check = next(c for c in report.checks if c.name == "citations_subset")
    assert check.passed is True
    assert "2 条引用全部在检索结果内" in check.detail
    assert report.passed is True


def test_citations_subset_fails_and_lists_invalid_ids(submission_rules):
    """A3：池外 ref_id → 失败，detail 逐条列出非法 ref_id 并写明约束。"""
    citations = [
        Citation(ref_id="stub:ref-1", claim="合法。"),
        Citation(ref_id="fake:999", claim="编造的第一条。"),
        Citation(ref_id="arxiv:2401.1", claim="编造的第二条。"),
    ]
    report = _evaluate(
        submission_rules,
        citations=citations,
        valid_ref_ids=["stub:ref-1"],
    )

    check = next(c for c in report.checks if c.name == "citations_subset")
    assert check.passed is False
    assert "只能引用 retrieval 返回过的文献" in check.detail
    assert "fake:999" in check.detail
    assert "arxiv:2401.1" in check.detail
    assert "stub:ref-1" not in check.detail.split("：", 1)[1]
    assert report.passed is False


def test_citation_has_claim_fails_on_empty_and_whitespace(submission_rules):
    """A3：claim 为空或纯空白 → citation_has_claim 失败（ref 合法也不放行）。"""
    citations = [
        Citation(ref_id="stub:ref-1", claim=""),
        Citation(ref_id="stub:ref-2", claim="   "),
    ]
    report = _evaluate(
        submission_rules,
        citations=citations,
        valid_ref_ids=["stub:ref-1", "stub:ref-2"],
    )

    check = next(c for c in report.checks if c.name == "citation_has_claim")
    assert check.passed is False
    assert check.owner_step == "writing"
    subset = next(c for c in report.checks if c.name == "citations_subset")
    assert subset.passed is True


def test_citation_stats_written_into_report(submission_rules):
    """A6：total / invalid / hallucination_rate 随报告带出。"""
    citations = [
        Citation(ref_id="stub:ref-1", claim="合法。"),
        Citation(ref_id="fake:1", claim="非法。"),
    ]
    report = _evaluate(
        submission_rules,
        citations=citations,
        valid_ref_ids=["stub:ref-1"],
    )

    assert isinstance(report.citation_stats, CitationStats)
    assert report.citation_stats.total == 2
    assert report.citation_stats.invalid == 1
    assert report.citation_stats.hallucination_rate == 0.5


def test_citation_stats_zero_total_rate_is_zero(submission_rules):
    """A6：total=0 时 hallucination_rate 为 0.0，不除零。"""
    report = _evaluate(submission_rules, citations=[], valid_ref_ids=["stub:ref-1"])

    stats = report.citation_stats
    assert stats.total == 0
    assert stats.invalid == 0
    assert stats.hallucination_rate == 0.0
    assert report.passed is True


def test_citations_none_defaults_to_empty(submission_rules):
    """旧调用方（不传 citations）不报错，统计为空且两条检查通过。"""
    report = _evaluate(submission_rules)

    assert report.passed is True
    assert report.citation_stats.total == 0
    assert report.citation_stats.hallucination_rate == 0.0


def test_dict_citations_from_old_checkpoint(submission_rules):
    """A1：checkpoint 反序列化后的字典形态引用同样被校验。"""
    citations = [
        {"ref_id": "stub:ref-1", "claim": "字典形态的合法引用。"},
        {"ref_id": "fake:7", "claim": "字典形态的非法引用。"},
    ]
    report = _evaluate(
        submission_rules,
        citations=citations,
        valid_ref_ids=["stub:ref-1"],
    )

    subset = next(c for c in report.checks if c.name == "citations_subset")
    assert subset.passed is False
    assert "fake:7" in subset.detail
    assert report.citation_stats.invalid == 1


def test_old_draft_content_without_citations_deserializes():
    """A1：旧 DraftContent（无 citations 字段）反序列化后落回空列表。"""
    content = DraftContent.model_validate(
        {
            "background": "背景",
            "findings": "发现",
            "limitation": "局限",
        }
    )

    assert content.citations == []


def test_real_rules_yaml_admits_new_checks(project_root):
    """真实规则文件能为两条新检查配齐 owner（loading 报错即失败）。"""
    from researchflow.tools.rules_engine import SubmissionRules, load_rules

    rules = load_rules(
        project_root / "configs" / "rules" / "submission_checks.yaml",
        model=SubmissionRules,
    )

    assert rules.owner_steps["citations_subset"] == "writing"
    assert rules.owner_steps["citation_has_claim"] == "writing"

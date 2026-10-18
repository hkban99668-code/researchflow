"""T-010 A3：⑦ 每条检查结果都有 owner_step，且是图中真实存在的节点名。"""

from __future__ import annotations

from researchflow.graphs.orchestrator import build_graph
from researchflow.tools.rules_engine import SubmissionRules, evaluate_submission, load_rules

GOOD_DRAFT = (
    "# Demo\n\n## 背景\n" + "研究背景与问题来源描述。" * 20 + "\n\n## 方法\n"
    "使用固定参数与固定随机种子的单算法回归。\n\n"
    "## 结果\nDIRECT 计算给出拟合指标与图表。\n\n## 局限\n样本量有限，结论待扩展。\n"
)
GOOD_ARTIFACTS = {"cleaned_data": True, "chart": True, "draft": True}


def test_every_check_result_has_owner_in_graph_nodes(project_root):
    """A3：evaluate_submission 的每条 CheckResult 都有 owner_step 且属于图的节点。"""
    graph_nodes = set(build_graph().get_graph().nodes)
    rules = load_rules(
        project_root / "configs" / "rules" / "submission_checks.yaml",
        model=SubmissionRules,
    )

    report = evaluate_submission(
        rules,
        metrics={"r2": 0.95},
        draft_text=GOOD_DRAFT,
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
    )

    assert len(report.checks) > 0, "应产生非空检查列表"
    for check in report.checks:
        assert check.owner_step, f"{check.name} 缺少 owner_step"
        assert check.owner_step in graph_nodes, f"{check.name} 的 owner_step 不是图节点"


def test_owner_lookup_prefers_full_name_over_prefix(project_root):
    """artifact:* 有完整名（如 artifact:chart），命中完整名而非 artifact 前缀。"""
    rules = load_rules(
        project_root / "configs" / "rules" / "submission_checks.yaml",
        model=SubmissionRules,
    )
    report = evaluate_submission(
        rules,
        metrics={"r2": 0.95},
        draft_text=GOOD_DRAFT,
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
    )
    owners = {check.name: check.owner_step for check in report.checks}

    assert owners["artifact:chart"] == "chart"
    assert owners["artifact:cleaned_data"] == "data_review"
    assert owners["artifact:draft"] == "writing"
    assert owners["metric:r2"] == "compute"
    assert owners["draft_min_chars"] == "writing"
    assert owners["min_rows"] == "data_prepare"

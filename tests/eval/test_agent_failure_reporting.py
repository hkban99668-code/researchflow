"""T-027 归因评测单列 Agent 失败数：AttributionRecord.agent_failed 进记录、
报告单列并把失败用例排除出归因准确率与恢复率的分母。

全程 Fake：用一个 `decide()` 直接抛 LLMError 的测试替身归因器制造"Agent 调用
失败降级"，不发真实网络请求、不调真实模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下（与 test_retrieval_eval 同法）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import run as eval_run  # noqa: E402

from researchflow.core.errors import LLMError  # noqa: E402


class FailingAttributor:
    """每次决策都抛 LLMError 的归因器替身：模拟 Agent 调用失败。"""

    def __init__(self, ctx: object = None) -> None:
        self.ctx = ctx

    def decide(self, state: dict) -> object:
        raise LLMError("模拟 Agent 崩溃（测试替身）")


def _record(case_id: str, **overrides: object) -> dict:
    base: dict = {
        "id": case_id,
        "group": "cross_step",
        "fault": "y_outliers",
        "expected": {
            "termination": "budget_exhausted",
            "error_code": None,
            "target_step": "data_prepare",
        },
        "actual_termination": "budget_exhausted",
        "termination_ok": True,
        "attribution_ok": True,
        "recovered": True,
        "agent_failed": False,
        "backtracks": 0,
        "tokens": 0,
        "latency_ms": 0,
        "decisions": [],
        "agent_target": None,
    }
    base.update(overrides)
    return base


# --- 汇总口径：失败用例单列并排除出分母 -----------------------------------------


def test_summarize_counts_agent_failed_and_excludes_from_denominators() -> None:
    """Agent 调用失败的用例不计入归因准确率与恢复率的分母，且单独列出编号。"""
    records = [
        _record("A"),
        _record("B"),
        _record("F", agent_failed=True, attribution_ok=False, recovered=False),
    ]
    summary = eval_run._summarize(records)

    assert summary["agent_failed"]["n"] == 1
    assert summary["agent_failed"]["ids"] == ["F"]
    # 归因准确率的分母只有 A、B（F 被排除），两者都判对 → 1.0（否则会是 2/3）
    assert summary["attribution_ok"]["overall"] == 1.0
    # 恢复率的分母同样排除 F
    assert summary["recovered"]["overall"] == 1.0


def test_summarize_without_failures_reports_zero() -> None:
    summary = eval_run._summarize([_record("A"), _record("B")])
    assert summary["agent_failed"] == {"n": 0, "ids": []}


def test_render_markdown_lists_failed_case_ids() -> None:
    """Markdown 报告单列"Agent 调用失败 N 条"并列出用例编号。"""
    records = [_record("A"), _record("F", agent_failed=True, attribution_ok=False)]
    report = {
        "generated_at": "2026-10-14T00:00:00+00:00",
        "git_commit": None,
        "cases_sha256": "0" * 64,
        "cases": records,
        "summary": eval_run._summarize(records),
        "smart": {"cases": [], "summary": eval_run._summarize([])},
        "search": {"cases": [], "summary": eval_run._summarize([])},
    }
    text = eval_run._render_markdown(report)
    assert "Agent 调用失败：1 条（F）" in text
    assert "已从归因准确率与恢复率的分母中排除" in text


# --- 端到端：失败归因器跑一条用例，记录带 agent_failed ---------------------------


def test_run_one_records_agent_failed_for_failing_attributor() -> None:
    """decide() 抛 LLMError 的归因器：attribute 节点降级转人工，记录带 agent_failed，
    评测记录把该用例标为失败（从而被排除出分母）。"""
    cases = [case for case in eval_run.load_cases() if case["id"] == "F-OUT-01"]
    assert cases, "用例集里应有 F-OUT-01"
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="rf-agent-failed-") as tmp:
        record = eval_run._run_one(cases[0], Path(tmp), FailingAttributor)

    assert record["agent_failed"] is True
    # 降级成转人工：决策是 escalate，用例恢复失败
    assert record["agent_target"] is None or record["attribution_ok"] is not True
    assert record["recovered"] is False

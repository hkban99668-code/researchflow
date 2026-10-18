"""T-019 A6：聪明规则基线单测——按 iqr → log 的顺序盲试，其余情况转人工。"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.smart_rule_attributor import SmartRuleAttributor  # noqa: E402

from researchflow.core.schemas import (  # noqa: E402
    AttributionDecision,
    AttributionRecord,
    CheckResult,
    ValidationReport,
)


def _state(failed_names: list[str], prior_backtracks: int = 0) -> dict:
    """构造 decide() 入参：校验报告 + 已发生的归因记录。"""
    checks = [CheckResult(name=n, passed=False, owner_step="compute") for n in failed_names]
    validation = ValidationReport(passed=False, checks=checks)
    attributions = [
        AttributionRecord(
            attempt=1,
            trigger="validation_failed",
            decision=AttributionDecision(
                action="backtrack", target_step="data_prepare", reason="上次回溯", source="agent"
            ),
        )
        for _ in range(prior_backtracks)
    ]
    return {"validation_result": validation, "attributions": attributions}


def test_first_entry_with_r2_backtracks_iqr():
    """有 metric:r2 且第一次进入 → 回溯到 data_prepare，patch iqr k=1.5。"""
    decision = SmartRuleAttributor().decide(_state(["metric:r2"]))
    assert decision.action == "backtrack"
    assert decision.target_step == "data_prepare"
    assert decision.patch.y_outlier_strategy == "iqr"
    assert decision.patch.y_outlier_k == 1.5


def test_second_entry_with_r2_backtracks_log():
    """有 metric:r2 且已回溯过一次 → 回溯到 compute，patch target_transform=log。"""
    decision = SmartRuleAttributor().decide(_state(["metric:r2"], prior_backtracks=1))
    assert decision.action == "backtrack"
    assert decision.target_step == "compute"
    assert decision.patch.target_transform == "log"


def test_third_entry_with_r2_escalates():
    """有 metric:r2 但已回溯过两次 → 转人工（图的预算闸通常会更早拦住）。"""
    decision = SmartRuleAttributor().decide(_state(["metric:r2"], prior_backtracks=2))
    assert decision.action == "escalate"


def test_no_r2_escalates():
    """失败检查里没有 metric:r2 → 转人工。"""
    decision = SmartRuleAttributor().decide(_state(["section:背景"]))
    assert decision.action == "escalate"

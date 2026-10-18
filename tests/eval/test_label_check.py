"""T-019 A5：标签核验——cross_step 每条可修且 expected_fix 可行，human 每条不可修。"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import label_check  # noqa: E402
from evals.run import load_cases  # noqa: E402

THRESHOLD = 0.8


def _agent_cases():
    return [c for c in load_cases() if c["group"] in ("cross_step", "human")]


def test_cross_step_cases_are_fixable():
    """A5：cross_step 组每条 fixable 必须为 True。"""
    for case in _agent_cases():
        if case["group"] != "cross_step":
            continue
        result = label_check.check_fixable(case)
        assert result["fixable"] is True, (
            f"{case['id']} 应可修但核验为不可修：r2={result['r2']} rows={result['rows']}"
        )


def test_human_cases_are_not_fixable():
    """A5：human 组每条 fixable 必须为 False。"""
    for case in _agent_cases():
        if case["group"] != "human":
            continue
        result = label_check.check_fixable(case)
        assert result["fixable"] is False, (
            f"{case['id']} 应不可修但核验为可修：shortest={result['shortest']}"
        )


def test_expected_fix_is_in_feasible_combos():
    """A5：带 expected_fix 的 cross_step 用例，其 expected_fix 必须出现在可行组合里。"""
    for case in _agent_cases():
        if case["group"] != "cross_step":
            continue
        if "expected_fix" not in case:
            continue  # T-019 之前的用例没有 expected_fix 字段
        op = label_check.expected_fix_op(case.get("expected_fix"))
        assert op is not None, f"{case['id']} 的 expected_fix 无法映射到某个修法"
        result = label_check.check_fixable(case)
        feasible_ops = {combo_op for combo in result["feasible"] for combo_op in combo}
        assert op in feasible_ops, (
            f"{case['id']} 的 expected_fix {op!r} 不在可行组合 {result['feasible']} 里"
        )


def test_none_setting_r2_below_threshold():
    """A5：每条 Agent 用例在不处理（none）时 r2 必须低于阈值，保证故障真的让 ⑦ 失败。"""
    for case in _agent_cases():
        result = label_check.check_fixable(case)
        none_r2 = result["r2"]["none"]
        assert none_r2 is not None, f"{case['id']} 的 none 设置计算失败"
        assert none_r2 < THRESHOLD, (
            f"{case['id']} 不处理时 r2={none_r2:.4f} 不低于阈值 {THRESHOLD}，故障不会让 ⑦ 失败"
        )


def test_k_choice_trap_k15_drops_below_min_rows():
    """A5：k_choice_trap 用 k=1.5 时行数低于 min_rows（陷阱）。"""
    min_rows = label_check._constants()[0]
    for case in _agent_cases():
        if case["fault"] != "k_choice_trap":
            continue
        result = label_check.check_fixable(case)
        assert result["rows"]["iqr_k1_5"] < min_rows, (
            f"{case['id']} 的 k=1.5 行数 {result['rows']['iqr_k1_5']} 未低于 min_rows {min_rows}"
        )
        assert result["rows"]["iqr_k3"] >= min_rows, (
            f"{case['id']} 的 k=3 行数 {result['rows']['iqr_k3']} 低于 min_rows {min_rows}"
        )

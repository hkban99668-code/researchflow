"""T-011 A1：用例清单的数量与覆盖约束。"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.run import load_cases  # noqa: E402

REQUIRED_FAULTS = [
    "clean",
    "duplicate_rows",
    "high_missing",
    "drop_column",
    "rename_column",
    "x_out_of_range",
    "too_few_rows",
    "type_pollution",
    "unknown_algorithm",
    "empty_retrieval",
    "y_outliers",
    "y_noise",
    "y_shuffled",
    # T-018 新增
    "y_outliers_with_noise",
    "y_outliers_one_feature_missing",
    "y_outliers_heavy_noise",
    "nonlinear_target",
    "y_outliers_partial",
    # T-019 新增
    "exp_target",
    "exp_target_with_outliers",
    "k_choice_trap",
]

# T-018 每类新故障至少 3 条；T-019 每类新故障至少 4 条
NEW_FAULTS = [
    "y_outliers_with_noise",
    "y_outliers_one_feature_missing",
    "y_outliers_heavy_noise",
    "nonlinear_target",
    "y_outliers_partial",
    "exp_target",
    "exp_target_with_outliers",
    "k_choice_trap",
]

VALID_GROUPS = {"clean", "single_step", "cross_step", "human"}
VALID_TERMINATIONS = {
    "ok",
    "error",
    "escalated",
    "rejected",
    "empty",
    "validation_failed",
    "budget_exhausted",
}


def test_cases_count_and_fault_coverage():
    """A1：至少 36 条；每类故障至少 2 条；cross_step + human 至少 36 条；新故障各至少 3 条。"""
    cases = load_cases()
    assert len(cases) >= 36

    by_fault = Counter(c["fault"] for c in cases)
    for fault in REQUIRED_FAULTS:
        assert by_fault[fault] >= 2, f"故障 {fault} 不足 2 条"
    for fault in NEW_FAULTS:
        assert by_fault[fault] >= 3, f"新故障 {fault} 不足 3 条"

    cross_human = sum(1 for c in cases if c["group"] in ("cross_step", "human"))
    assert cross_human >= 36


def test_cases_ids_unique_and_fields_valid():
    """每条用例 id 唯一、group 合法、expected 字段齐全且取值合法。"""
    cases = load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "用例 id 必须唯一"

    for case in cases:
        assert case["group"] in VALID_GROUPS, case["id"]
        expected = case["expected"]
        assert expected["termination"] in VALID_TERMINATIONS, case["id"]
        assert "target_step" in expected, case["id"]
        # clean 组 target_step 必须是 null
        if case["group"] == "clean":
            assert expected["target_step"] is None, case["id"]
        if expected["termination"] == "error":
            assert expected.get("error_code"), f"{case['id']} 期望 error 却没有 error_code"


def test_cases_split_assignment():
    """A5：每条都有 split；cross_step/human 按 id 奇偶分 dev/test 且各占一半；其余组 test。"""
    cases = load_cases()
    for case in cases:
        assert case["split"] in ("dev", "test"), case["id"]

    for group in ("cross_step", "human"):
        group_cases = [c for c in cases if c["group"] == group]
        ordered_ids = sorted(c["id"] for c in group_cases)
        # 顶部注释的规则：按 id 字典序排序后，奇数位（1-indexed）进 dev、偶数位进 test。
        expected_dev = {ordered_ids[i] for i in range(len(ordered_ids)) if i % 2 == 0}
        expected_test = {ordered_ids[i] for i in range(len(ordered_ids)) if i % 2 == 1}
        dev = {c["id"] for c in group_cases if c["split"] == "dev"}
        test = {c["id"] for c in group_cases if c["split"] == "test"}
        assert dev == expected_dev, f"{group} 的 dev 分配不符合奇偶规则"
        assert test == expected_test, f"{group} 的 test 分配不符合奇偶规则"
        # dev 与 test 各占一半（组内数量为偶数）
        assert len(dev) == len(test), f"{group} 的 dev/test 数量不相等"

    # 其他组一律 test
    for case in cases:
        if case["group"] in ("clean", "single_step"):
            assert case["split"] == "test", case["id"]

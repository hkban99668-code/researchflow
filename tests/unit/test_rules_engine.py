"""⑦ 硬规则校验测试（先红后绿）。"""

from __future__ import annotations

import pytest

from researchflow.core.errors import ConfigError
from researchflow.core.schemas import ValidationReport
from researchflow.tools.rules_engine import (
    SubmissionRules,
    evaluate_submission,
    load_rules,
)

GOOD_DRAFT = (
    "# Demo\n\n## 背景\n" + "研究背景与问题来源描述。" * 20 + "\n\n## 方法\n"
    "使用固定参数与固定随机种子的单算法回归。\n\n"
    "## 结果\nDIRECT 计算给出拟合指标与图表。\n\n## 局限\n样本量有限，结论待扩展。\n"
)
GOOD_ARTIFACTS = {"cleaned_data": True, "chart": True, "draft": True}


def test_all_checks_pass(submission_rules):
    report = evaluate_submission(
        submission_rules,
        metrics={"r2": 0.95, "rmse": 0.3},
        draft_text=GOOD_DRAFT,
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
    )

    assert isinstance(report, ValidationReport)
    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_metric_below_threshold_fails(submission_rules):
    report = evaluate_submission(
        submission_rules,
        metrics={"r2": 0.1},
        draft_text=GOOD_DRAFT,
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
    )

    assert report.passed is False
    assert any(check.name == "metric:r2" and not check.passed for check in report.checks)


def test_forbidden_token_fails(submission_rules):
    report = evaluate_submission(
        submission_rules,
        metrics={"r2": 0.95},
        draft_text=GOOD_DRAFT + "\nTODO 待补充",
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
    )

    assert report.passed is False
    assert any(check.name == "forbidden_tokens" and not check.passed for check in report.checks)


def test_missing_required_section_fails(submission_rules):
    report = evaluate_submission(
        submission_rules,
        metrics={"r2": 0.95},
        draft_text="# Demo\n\n## 背景\n只有背景。\n",
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
    )

    assert report.passed is False
    assert any(check.name.startswith("section:") and not check.passed for check in report.checks)


def test_missing_artifact_fails(submission_rules):
    report = evaluate_submission(
        submission_rules,
        metrics={"r2": 0.95},
        draft_text=GOOD_DRAFT,
        artifacts={"cleaned_data": True, "chart": False, "draft": True},
        row_count=60,
    )

    assert report.passed is False
    assert any(check.name == "artifact:chart" and not check.passed for check in report.checks)


def test_variables_in_dataset_fails_when_variable_not_in_columns(submission_rules):
    """A5：variables 含不存在的列名时 variables_in_dataset 不通过，owner_step 为 extraction。"""
    report = evaluate_submission(
        submission_rules,
        metrics={"r2": 0.95},
        draft_text=GOOD_DRAFT,
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
        variables=["x1", "missing_rate"],
        dataset_columns=["x1", "x2", "x3", "y"],
    )

    assert report.passed is False
    check = next(c for c in report.checks if c.name == "variables_in_dataset")
    assert check.passed is False
    assert "missing_rate" in check.detail
    assert check.owner_step == "extraction"


def test_variables_in_dataset_passes_when_all_in_columns(submission_rules):
    """A5：variables 全部在列中时通过。"""
    report = evaluate_submission(
        submission_rules,
        metrics={"r2": 0.95},
        draft_text=GOOD_DRAFT,
        artifacts=GOOD_ARTIFACTS,
        row_count=60,
        variables=["x1", "y"],
        dataset_columns=["x1", "x2", "x3", "y"],
    )

    assert report.passed is True
    check = next(c for c in report.checks if c.name == "variables_in_dataset")
    assert check.passed is True


def test_missing_rules_file_and_empty_rules_must_fail(tmp_path):
    """规则文件缺失必须抛错；空规则不得被判为通过。"""
    with pytest.raises(ConfigError):
        load_rules(tmp_path / "not-exist.yaml")

    with pytest.raises(ConfigError):
        evaluate_submission(
            {},
            metrics={"r2": 0.0},
            draft_text="很短的草稿",
            artifacts={},
            row_count=1,
        )


def test_rules_missing_required_section_must_fail(tmp_path):
    """规则文件缺必填段时 load_rules 必须抛错，而不是用默认值静默通过。"""
    partial = tmp_path / "partial_submission.yaml"
    partial.write_text("min_draft_chars: 200\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_rules(partial, model=SubmissionRules)

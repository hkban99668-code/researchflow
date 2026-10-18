"""③ 数据质量预检测试（先红后绿）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchflow.core.config import sha256_file
from researchflow.core.errors import DataValidationError
from researchflow.core.schemas import QualitySummary
from researchflow.tools.clean_strategies import clean_frame
from researchflow.tools.quality import precheck


def test_precheck_returns_summary_for_valid_frame(signal_frame, data_rules):
    summary = precheck(signal_frame, data_rules)

    assert isinstance(summary, QualitySummary)
    assert summary.rows == len(signal_frame)
    assert summary.columns == 4
    assert summary.missing_rate == 0.0
    assert summary.duplicates_removed == 0


def test_precheck_rejects_missing_required_column(signal_frame, data_rules):
    frame = signal_frame.drop(columns=["y"])

    with pytest.raises(DataValidationError) as exc:
        precheck(frame, data_rules)

    assert exc.value.code == "RF-003"
    assert "y" in exc.value.message


def test_precheck_rejects_missing_rate_above_threshold(signal_frame, data_rules):
    frame = signal_frame.copy()
    frame.loc[frame.index[:60], "x2"] = pd.NA

    with pytest.raises(DataValidationError):
        precheck(frame, data_rules)


def test_precheck_rejects_value_out_of_range(signal_frame, data_rules):
    frame = signal_frame.copy()
    frame.loc[0, "x1"] = 999.0

    with pytest.raises(DataValidationError):
        precheck(frame, data_rules)


def test_precheck_rejects_too_few_rows(signal_frame, data_rules):
    frame = signal_frame.head(5)

    with pytest.raises(DataValidationError):
        precheck(frame, data_rules)


def test_sha256_file_is_stable_and_content_sensitive(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("researchflow", encoding="utf-8")
    first = sha256_file(path)

    other = tmp_path / "b.txt"
    other.write_text("researchflow", encoding="utf-8")
    assert sha256_file(other) == first

    other.write_text("changed", encoding="utf-8")
    assert sha256_file(other) != first


def test_missing_rate_is_measured_before_dropna(data_rules):
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(0, 10, 100),
            "x2": rng.uniform(-5, 5, 100),
            "x3": rng.uniform(1, 3, 100),
            "y": rng.normal(0, 1, 100),
        }
    )
    frame.loc[frame.index[:40], "x1"] = np.nan

    cleaned, stats = clean_frame(frame, columns=["x1", "x2", "x3", "y"])

    assert stats["raw_missing_rate"] == pytest.approx(0.4), "原始缺失率必须在 dropna 之前统计"
    assert stats["dropped_rows"] == 40

    with pytest.raises(DataValidationError) as exc:
        precheck(
            cleaned,
            data_rules,
            duplicates_removed=stats["duplicates_removed"],
            raw_missing_rate=stats["raw_missing_rate"],
            dropped_rows=stats["dropped_rows"],
            rows_before=stats["rows_before"],
        )

    assert "0.4" in exc.value.message, "应按原始缺失率 0.4 判定"


def test_duplicates_are_detected_after_numeric_coercion():
    frame = pd.DataFrame({"x1": ["1", "1.0"], "x2": ["2", "2"], "x3": ["3", "3"], "y": ["4", "4"]})

    _cleaned, stats = clean_frame(frame, columns=["x1", "x2", "x3", "y"])

    assert stats["duplicates_removed"] == 1


def test_clean_frame_default_does_not_remove_y_outliers():
    """A1：默认参数（y_outlier_strategy 缺省为 none）下不改行为，离群值原样保留。"""
    frame = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
        }
    )

    cleaned, stats = clean_frame(frame, columns=["x1", "y"], target="y")

    assert stats["y_outliers_removed"] == 0
    assert len(cleaned) == len(frame)


def test_clean_frame_iqr_removes_y_outliers():
    """A1：`y_outlier_strategy="iqr"` 在 dropna 之后去掉 target 列 IQR 外的行。"""
    frame = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
        }
    )

    cleaned, stats = clean_frame(
        frame, columns=["x1", "y"], target="y", y_outlier_strategy="iqr", y_outlier_k=3.0
    )

    assert stats["y_outliers_removed"] == 1
    assert len(cleaned) == 9
    assert cleaned["y"].max() == 9.0
    assert any("离群值" in note for note in stats["notes"])


def test_clean_frame_iqr_requires_target():
    """`y_outlier_strategy="iqr"` 但没有 target 时抛错，而不是静默跳过。"""
    frame = pd.DataFrame({"x1": [1, 2, 3], "y": [1.0, 2.0, 100.0]})

    with pytest.raises(ValueError):
        clean_frame(frame, columns=["x1", "y"], y_outlier_strategy="iqr")

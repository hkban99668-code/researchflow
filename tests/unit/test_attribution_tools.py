"""A11：归因只读工具 `inspect_column` 只返回统计量、不返回原始行，列不存在时报错。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from researchflow.core.config import RunContext, load_settings, sha256_file
from researchflow.core.schemas import CleanedDataRef, ParamSet
from researchflow.core.tools import ToolContext
from researchflow.tools.attribution_tools import build_attribution_registry

STAT_KEYS = {
    "column",
    "count",
    "mean",
    "std",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "outliers_k15",
    "outliers_k30",
}


def _registry_with_cleaned_data(tmp_path, project_root):
    settings = load_settings(project_root=project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    csv_path = run_dir / "cleaned_candidate.csv"
    frame.to_csv(csv_path, index=False)
    ref = CleanedDataRef(
        path="cleaned_candidate.csv",
        checksum=sha256_file(csv_path),
        rows=6,
        columns=["x", "y"],
    )
    state = {"cleaned_data": ref}
    run_ctx = RunContext(settings=settings, run_dir=run_dir)
    return build_attribution_registry(state, run_ctx), ToolContext(
        workspace_root=settings.project_root, settings=settings
    )


def test_inspect_column_returns_stats_not_raw_rows(tmp_path, project_root):
    """A11：`inspect_column` 只返回汇总数字，不返回原始行。"""
    registry, tool_ctx = _registry_with_cleaned_data(tmp_path, project_root)

    result = registry.invoke("inspect_column", {"column": "y"}, tool_ctx)

    assert result.ok
    assert set(result.output) == STAT_KEYS, "只返回统计量字段，不含原始行"
    assert result.output["count"] == 6
    assert result.output["mean"] == 3.5


def test_inspect_column_reports_outlier_counts(tmp_path, project_root):
    """`inspect_column` 对含离群值的列给出两种口径的 IQR 外点数。"""
    registry, tool_ctx = _registry_with_cleaned_data(tmp_path, project_root)

    result = registry.invoke("inspect_column", {"column": "x"}, tool_ctx)

    assert result.ok
    assert result.output["outliers_k15"] == 1
    assert result.output["outliers_k30"] == 1


def test_inspect_column_missing_column_returns_invalid_args(tmp_path, project_root):
    """A11：列不存在时返回 `invalid_args` 并列出可用列。"""
    registry, tool_ctx = _registry_with_cleaned_data(tmp_path, project_root)

    result = registry.invoke("inspect_column", {"column": "nonexistent"}, tool_ctx)

    assert not result.ok
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert "x" in result.error.message and "y" in result.error.message


def _registry_with_relation_data(tmp_path, project_root, frame, features=None):
    """构造带 param_set 的注册表，供 `feature_target_relation` 测试使用。"""
    settings = load_settings(project_root=project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "cleaned_candidate.csv"
    frame.to_csv(csv_path, index=False)
    ref = CleanedDataRef(
        path="cleaned_candidate.csv",
        checksum=sha256_file(csv_path),
        rows=len(frame),
        columns=list(frame.columns),
    )
    state = {
        "cleaned_data": ref,
        "param_set": ParamSet(
            algorithm="linear_regression",
            params={"target": "y", "features": features or ["x"]},
            seed=42,
        ),
    }
    run_ctx = RunContext(settings=settings, run_dir=run_dir)
    return build_attribution_registry(state, run_ctx), ToolContext(
        workspace_root=settings.project_root, settings=settings
    )


def test_feature_target_relation_strong_linear(tmp_path, project_root):
    """A1：y 与 x 强线性相关时，相关系数与全量 R² 都接近 1，且不返回原始行。"""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 100)
    y = 3.0 * x + rng.normal(0, 0.05, 100)
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path, project_root, pd.DataFrame({"x": x, "y": y})
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    # A3：输出仍然只有统计量，不含原始行。
    assert set(out) == {
        "correlations",
        "r2",
        "n_rows",
        "min_rows",
        "note",
        "robust_by_k",
        "log_target",
        "r2_without_feature",
    }
    assert len(out["correlations"]) == 1
    assert out["correlations"][0]["feature"] == "x"
    assert out["correlations"][0]["pearson"] > 0.9
    assert out["r2"] > 0.9
    assert out["n_rows"] == 100
    assert out["note"] == ""
    # 没有离群值：两个口径的稳健统计都与原值一致，去掉行数为 0、剩余行数等于全量。
    assert set(out["robust_by_k"]) == {"1.5", "3"}
    for stats in out["robust_by_k"].values():
        assert stats["rows_removed"] == 0
        assert stats["rows_after"] == out["n_rows"]
        assert stats["r2"] == out["r2"]
        assert stats["correlations"][0]["pearson"] == out["correlations"][0]["pearson"]


def test_feature_target_relation_outliers_robust_r2_higher(tmp_path, project_root):
    """A1：注入极端值后 r2 明显低于稳健 r2，rows_removed 等于注入个数。"""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(-5, 5, n)
    y = 3.0 * x + rng.normal(0, 0.05, n)
    n_outliers = 8
    y = y.copy()
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    y[outlier_idx] += 200.0
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path, project_root, pd.DataFrame({"x": x, "y": y})
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    assert out["r2"] < out["robust_by_k"]["3"]["r2"]
    assert out["robust_by_k"]["3"]["rows_removed"] == n_outliers
    assert out["robust_by_k"]["3"]["r2"] > 0.9


def test_feature_target_relation_unrelated_robust_equals_full(tmp_path, project_root):
    """A2：y 与 x 无关、没有极端值时 robust r2 == r2，rows_removed == 0。"""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 100)
    y = rng.normal(0, 1, 100)
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path, project_root, pd.DataFrame({"x": x, "y": y})
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    for stats in out["robust_by_k"].values():
        assert stats["rows_removed"] == 0
        assert stats["r2"] == out["r2"]


def test_feature_target_relation_robust_by_k_15_tighter(tmp_path, project_root):
    """必改项2：少量 y 同方向平移时，k=1.5 的 r2 明显高于 k=3 的 r2。"""
    rng = np.random.default_rng(2)
    x_body = rng.uniform(-5, 5, 94)
    y_body = 3.0 * x_body + rng.normal(0, 0.05, 94)
    # 6 个同方向平移点：x 固定在 5（y≈15），整体 +30，落在 k=1.5 边界外、k=3 边界内。
    x_shift = np.full(6, 5.0)
    y_shift = 15.0 + 30.0 + rng.normal(0, 0.05, 6)
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path,
        project_root,
        pd.DataFrame(
            {"x": np.concatenate([x_body, x_shift]), "y": np.concatenate([y_body, y_shift])}
        ),
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    r2_15 = out["robust_by_k"]["1.5"]["r2"]
    r2_3 = out["robust_by_k"]["3"]["r2"]
    # k 越小去得越多：1.5 去掉全部 6 个平移点、恢复高 r2；3 识别不到、r2 仍被拉低。
    assert out["robust_by_k"]["1.5"]["rows_removed"] == 6
    assert out["robust_by_k"]["3"]["rows_removed"] == 0
    assert r2_15 - r2_3 > 0.1
    assert r2_15 > 0.95


def test_feature_target_relation_unrelated(tmp_path, project_root):
    """A1：y 与 x 无关时，相关系数与全量 R² 都接近 0。"""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 100)
    y = rng.normal(0, 1, 100)
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path, project_root, pd.DataFrame({"x": x, "y": y})
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    assert abs(out["correlations"][0]["pearson"]) < 0.3
    assert out["r2"] < 0.1


def test_feature_target_relation_zero_variance_feature(tmp_path, project_root):
    """T-015 裁定：零方差特征无法算相关系数，pearson 记为 null，输出带 note。"""
    rng = np.random.default_rng(0)
    x2 = rng.uniform(-5, 5, 100)
    y = 3.0 * x2 + rng.normal(0, 0.05, 100)
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path,
        project_root,
        pd.DataFrame({"x1": [5.0] * 100, "x2": x2, "y": y}),
        features=["x1", "x2"],
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    corr = {c["feature"]: c["pearson"] for c in out["correlations"]}
    assert corr["x1"] is None
    assert corr["x2"] > 0.9
    assert "x1" in out["note"]
    assert out["r2"] > 0.9


def test_feature_target_relation_reports_min_rows_and_r2_without_feature(tmp_path, project_root):
    """A4：min_rows 来自 data_acceptance.yaml；r2_without_feature 键与当前特征一致。"""
    rng = np.random.default_rng(3)
    n = 120
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    x3 = rng.uniform(1, 3, n)
    y = 2.0 * x1 - 1.5 * x2 + 0.5 * x3 + rng.normal(0, 0.05, n)
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path,
        project_root,
        pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y}),
        features=["x1", "x2", "x3"],
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    assert out["min_rows"] == 30
    assert set(out["r2_without_feature"]) == {"x1", "x2", "x3"}
    assert all(isinstance(value, float) for value in out["r2_without_feature"].values())


def test_feature_target_relation_log_target_applicable(tmp_path, project_root):
    """A1：y 全部 > 0 时 log_target.applicable=true，log 后 R² 明显高于全量 R²。"""
    rng = np.random.default_rng(4)
    n = 120
    x = rng.uniform(0, 5, n)
    y = np.exp(0.5 * x + 1.0) * (1.0 + rng.normal(0, 0.05, n))
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path, project_root, pd.DataFrame({"x": x, "y": y})
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    assert out["log_target"]["applicable"] is True
    assert out["log_target"]["r2"] > 0.9
    assert out["log_target"]["r2"] > out["r2"]


def test_feature_target_relation_log_target_nonpositive(tmp_path, project_root):
    """A2：y 含 ≤ 0 值时 log_target.applicable=false，note 写明个数。"""
    rng = np.random.default_rng(5)
    n = 100
    x = rng.uniform(0, 5, n)
    y = np.exp(0.5 * x + 1.0)
    y = y.copy()
    y[:5] = -1.0
    registry, tool_ctx = _registry_with_relation_data(
        tmp_path, project_root, pd.DataFrame({"x": x, "y": y})
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    assert out["log_target"]["applicable"] is False
    assert out["log_target"]["r2"] is None
    assert "5" in out["log_target"]["note"]

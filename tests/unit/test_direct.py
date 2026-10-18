"""④ DIRECT 单算法测试（先红后绿）。"""

from __future__ import annotations

import pytest

from researchflow.compute.direct import run_direct
from researchflow.core.errors import ComputeError
from researchflow.core.schemas import ComputeResult


def test_run_direct_is_deterministic(signal_frame):
    kwargs = {
        "target": "y",
        "features": ["x1", "x2", "x3"],
        "params": {},
        "seed": 42,
        "algorithm": "linear_regression",
    }

    first = run_direct(signal_frame, **kwargs)
    second = run_direct(signal_frame, **kwargs)

    assert isinstance(first, ComputeResult)
    assert first.metrics == second.metrics


def test_run_direct_reports_expected_metrics(signal_frame):
    result = run_direct(
        signal_frame,
        target="y",
        features=["x1", "x2", "x3"],
        params={},
        seed=42,
        algorithm="linear_regression",
    )

    assert set(result.metrics) == {"r2", "rmse", "mae", "n_train", "n_test"}
    assert result.metrics["r2"] > 0.8
    assert result.parameters["seed"] == 42
    assert result.algorithm == "linear_regression"


def test_run_direct_rejects_missing_target_column(signal_frame):
    with pytest.raises(ComputeError) as exc:
        run_direct(
            signal_frame.drop(columns=["y"]),
            target="y",
            features=["x1", "x2", "x3"],
            params={},
            seed=42,
            algorithm="linear_regression",
        )

    assert exc.value.code == "RF-004"


def test_run_direct_rejects_unregistered_algorithm(signal_frame):
    """A8：未登记的算法抛 RF-004，消息列出已登记算法，不再静默跑线性回归。"""
    with pytest.raises(ComputeError) as exc:
        run_direct(
            signal_frame,
            target="y",
            features=["x1", "x2", "x3"],
            params={},
            seed=42,
            algorithm="ridge",
        )

    assert exc.value.code == "RF-004"
    assert "ridge" in exc.value.message
    assert "linear_regression" in exc.value.message


def test_run_direct_log_transform_recovers_exponential_target():
    """A2：target_transform=log 对指数目标在原始尺度上计算 r2 且 ≥ 0.8。"""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 10, 200)
    x2 = rng.uniform(-10, 10, 200)
    x3 = rng.uniform(0, 5, 200)
    y = np.exp(0.5 * x1 - 0.4 * x2 + 0.3 * x3 + 1.0)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

    raw = run_direct(
        frame,
        target="y",
        features=["x1", "x2", "x3"],
        params={},
        seed=42,
        algorithm="linear_regression",
    )
    logged = run_direct(
        frame,
        target="y",
        features=["x1", "x2", "x3"],
        params={},
        seed=42,
        algorithm="linear_regression",
        target_transform="log",
    )

    assert raw.metrics["r2"] < 0.8
    assert logged.metrics["r2"] >= 0.8
    # 指标在原始尺度上：n_train / n_test 与默认口径一致
    assert logged.metrics["n_train"] == raw.metrics["n_train"]
    assert logged.metrics["n_test"] == raw.metrics["n_test"]


def test_run_direct_log_transform_rejects_nonpositive_y(signal_frame):
    """A2：y 含非正值时 target_transform=log 抛 RF-004，消息说明原因。"""
    frame = signal_frame.copy()
    frame.loc[frame.index[0], "y"] = -1.0

    with pytest.raises(ComputeError) as exc:
        run_direct(
            frame,
            target="y",
            features=["x1", "x2", "x3"],
            params={},
            seed=42,
            algorithm="linear_regression",
            target_transform="log",
        )

    assert exc.value.code == "RF-004"
    assert "log" in exc.value.message


def test_run_direct_exclude_features_uses_remaining_features(signal_frame):
    """A3：排除后建模只用剩下的特征。"""
    result = run_direct(
        signal_frame,
        target="y",
        features=["x1", "x2", "x3"],
        params={},
        seed=42,
        algorithm="linear_regression",
        exclude_features=["x3"],
    )

    assert result.parameters["features"] == ["x1", "x2"]
    assert result.parameters["exclude_features"] == ["x3"]
    assert len(result.parameters["coefficients"]) == 2

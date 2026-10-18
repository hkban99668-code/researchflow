"""④ DIRECT 计算：Gate 1 只实现单一算法 + 固定随机种子。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..core.errors import ComputeError
from ..core.schemas import ComputeResult

ALGORITHM = "linear_regression"
# Gate 1 只登记线性回归；未登记的值抛 ComputeError，不静默跑线性回归（P1-4 前半）。
REGISTERED_ALGORITHMS = {ALGORITHM}


def run_direct(
    frame: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    params: dict | None = None,
    seed: int = 42,
    algorithm: str,
    target_transform: str = "none",
    exclude_features: list[str] | None = None,
) -> ComputeResult:
    """在同一进程内直接执行单算法计算（Gate 1 不做沙箱与进程隔离）。

    `target_transform="log"` 时用 `log(y)` 拟合，预测值取 `exp` 还原后在**原始尺度**上
    计算 r2/rmse/mae（指标含义不变）；y 中有 ≤ 0 的值时抛 `ComputeError`（RF-004）。
    `exclude_features` 里的特征不参与建模，建模只用剩下的特征。默认值下行为与原来
    完全一致。
    """
    if algorithm not in REGISTERED_ALGORITHMS:
        raise ComputeError(
            f"未登记的算法: {algorithm!r}（已登记: {', '.join(sorted(REGISTERED_ALGORITHMS))}）"
        )
    params = dict(params or {})
    if target_transform not in ("none", "log"):
        raise ComputeError(f"target_transform 取值非法: {target_transform!r}（只能是 none 或 log）")
    used_features = [name for name in features if name not in (exclude_features or [])]
    needed = [*used_features, target]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ComputeError(f"缺少计算所需列: {', '.join(missing)}")

    data = frame[needed].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ComputeError("清洗后可用样本为空")

    x = data[used_features].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)

    if target_transform == "log":
        if (y <= 0).any():
            n_bad = int((y <= 0).sum())
            raise ComputeError(
                f"target_transform=log 要求目标列 {target!r} 全部为正，"
                f"但存在 {n_bad} 个 ≤ 0 的值，无法取对数"
            )

    test_size = float(params.get("test_size", 0.25))
    # 始终在原始 y 上划分，保证 n_train/n_test 与默认口径一致；log 只影响拟合目标。
    x_train, x_test, y_train_orig, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )
    y_train = np.log(y_train_orig) if target_transform == "log" else y_train_orig

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred_fit = model.predict(x_test)
    y_pred = np.exp(y_pred_fit) if target_transform == "log" else y_pred_fit

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_train": float(len(x_train)),
        "n_test": float(len(x_test)),
    }
    parameters = {
        "seed": seed,
        "test_size": test_size,
        "features": list(used_features),
        "target": target,
        "target_transform": target_transform,
        "exclude_features": list(exclude_features or []),
        "coefficients": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
    }

    return ComputeResult(
        algorithm=algorithm,
        metrics=metrics,
        parameters=parameters,
        summary=(
            f"{algorithm} 在 {len(data)} 行样本上完成拟合，"
            f"测试集 R²={metrics['r2']:.4f}，RMSE={metrics['rmse']:.4f}"
        ),
    )

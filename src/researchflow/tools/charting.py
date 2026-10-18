"""⑤ 固定模板图表：Agg 后端 + 固定尺寸，避免环境差异导致 e2e 不稳定。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (必须在 use("Agg") 之后导入)

FIGSIZE = (6.4, 4.8)
DPI = 120


def render_actual_vs_predicted(
    path: str | Path,
    *,
    y_true: list[float],
    y_pred: list[float],
    title: str = "",
) -> Path:
    """绘制真实值 vs 预测值散点图，返回产物路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axes.scatter(y_true, y_pred, s=18, color="#2563eb", alpha=0.8, label="samples")

    low = float(min([*y_true, *y_pred]))
    high = float(max([*y_true, *y_pred]))
    axes.plot(
        [low, high], [low, high], color="#dc2626", linewidth=1.2, linestyle="--", label="y = x"
    )

    axes.set_xlabel("actual")
    axes.set_ylabel("predicted")
    axes.set_title(title or "Actual vs Predicted")
    axes.legend(loc="best")
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)

    return target

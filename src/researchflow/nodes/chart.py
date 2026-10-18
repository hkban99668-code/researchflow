"""⑤ 固定模板图表：用 DIRECT 系数回算全量预测值并出图。"""

from __future__ import annotations

import pandas as pd
from langgraph.runtime import Runtime

from ..core.config import RunContext, sha256_file
from ..core.schemas import ArtifactRef
from ..core.step import current_attempt, step
from ..tools.charting import render_actual_vs_predicted


@step("chart")
def chart_node(state: dict, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    parameters = state["compute_result"].parameters
    features = list(parameters["features"])
    target = parameters["target"]
    coefficients = list(parameters["coefficients"])
    intercept = float(parameters["intercept"])

    # P2-2：读取前校验 checksum，读到的必须是审核过的那份数据
    frame = pd.read_csv(ctx.read_verified(state["cleaned_data"]))
    y_true = [float(value) for value in frame[target].tolist()]
    y_pred = [
        sum(coef * float(value) for coef, value in zip(coefficients, row, strict=True)) + intercept
        for row in frame[features].to_numpy(dtype=float).tolist()
    ]

    path = ctx.artifact_path("chart.png")
    render_actual_vs_predicted(
        path, y_true=y_true, y_pred=y_pred, title=f"Actual vs Predicted · {target}"
    )
    checksum = sha256_file(path)
    rel = ctx.relative_to_run(path)

    return {
        "chart_path": rel,
        "artifacts": {
            "chart": ArtifactRef(
                kind="chart",
                path=rel,
                checksum=checksum,
                produced_by="chart",
                attempt=current_attempt(state),
            )
        },
        "step_status": {"chart": "ok"},
    }

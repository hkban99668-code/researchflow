"""④ DIRECT 计算：读取清洗后数据，执行单算法并落盘结果。"""

from __future__ import annotations

import json

import pandas as pd
from langgraph.runtime import Runtime

from ..compute.direct import run_direct
from ..core.config import RunContext, sha256_file
from ..core.schemas import ArtifactRef
from ..core.step import current_attempt, step


@step("compute")
def compute_node(state: dict, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    cleaned = state["cleaned_data"]
    param_set = state["param_set"]
    params = param_set.params

    # P2-2：读取前校验 checksum，读到的必须是审核过的那份数据
    frame = pd.read_csv(ctx.read_verified(cleaned))
    result = run_direct(
        frame,
        target=params["target"],
        features=list(params.get("features") or []),
        params=params,
        seed=param_set.seed,
        algorithm=param_set.algorithm,
        target_transform=params.get("target_transform", "none"),
        exclude_features=list(params.get("exclude_features") or []),
    )

    target = ctx.artifact_path("compute_result.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksum = sha256_file(target)

    return {
        "compute_result": result,
        "artifacts": {
            "compute_result": ArtifactRef(
                kind="compute_result",
                path=ctx.relative_to_run(target),
                checksum=checksum,
                produced_by="compute",
                attempt=current_attempt(state),
            )
        },
        "step_status": {"compute": "ok"},
    }

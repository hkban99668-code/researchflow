"""请求模型：`topic` 与 `topic.yaml` 同结构，`decision` 取值以审核节点为准。

`decision` 用 `Literal["approve", "reject"]`：与 `core.schemas.ReviewDecision` 一致，
非法取值由 FastAPI 直接 422，不会流到审核节点才报 ConfigError。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetSpec(BaseModel):
    """数据集三要素：路径（相对项目根或绝对）、目标列、特征列。"""

    path: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    features: list[str] = Field(..., min_length=1)


class ParamSetSpec(BaseModel):
    """参数集：算法、随机种子与算法参数（与 topic.yaml 的 `param_set` 对齐）。"""

    algorithm: str = "linear_regression"
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class TopicSpec(BaseModel):
    """课题对象，与 `topic.yaml` 同结构。"""

    topic: str = Field(..., min_length=1)
    dataset: DatasetSpec
    param_set: ParamSetSpec | None = None

    def to_spec(self) -> dict[str, Any]:
        """转成 `main.initial_state` 认识的 spec（`seed=None` 时不写，落回默认种子）。"""
        spec: dict[str, Any] = {"topic": self.topic, "dataset": self.dataset.model_dump()}
        if self.param_set is not None:
            spec["param_set"] = self.param_set.model_dump(exclude_none=True)
        return spec


class StartRunRequest(BaseModel):
    """`POST /runs` 请求体。

    `mode` 默认 `fake`：省掉这个字段时绝不能悄悄去花真实模型的钱（`real` 还要过
    `RF_API_ALLOW_REAL`）。
    """

    topic: TopicSpec
    mode: Literal["fake", "real"] = "fake"


class ResumeRequest(BaseModel):
    """`POST /runs/{id}/resume` 请求体：审核决定与原因。"""

    decision: Literal["approve", "reject"]
    reason: str = ""

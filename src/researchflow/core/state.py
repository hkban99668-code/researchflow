"""GraphState 与追加字段的 reducer。

三类字段的 reducer 对应三种语义（P1-3）：
- `literature_refs`：检索结果是"一次完整结果集"，重跑就整体替换，用普通字段（默认覆盖）。
- `artifacts`：每种产物只有一份当前有效，按 `kind` 覆盖，用 `merge_dict`。
- `review_log` / `step_records`：审计日志只增不减，用 `append_list`。
`step_status` 是最新结论，用 `merge_dict`。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, TypeVar

from .schemas import (
    ArtifactRef,
    AttributionRecord,
    CleanedDataRef,
    ComputeResult,
    DraftContent,
    ErrorRecord,
    LiteratureRef,
    LLMCallRecord,
    ParamSet,
    ResearchQuestion,
    RetrievalTrace,
    ReviewRecord,
    StepRecord,
    Termination,
    ValidationReport,
)

T = TypeVar("T")


def append_list(left: list[T] | None, right: list[T] | None) -> list[T]:
    """列表追加 reducer。"""
    return [*(left or []), *(right or [])]


def merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """字典浅合并 reducer，右侧覆盖左侧。"""
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    """Gate 1 链路状态。

    字段说明见设计方案 v2.0 §5.1。
    """

    run_id: str
    topic: str
    # 替换语义：重跑 ① 会整体替换上一次的检索结果（P1-3）
    literature_refs: list[LiteratureRef]
    research_question: ResearchQuestion | None
    # ③ 清洗后的**候选**数据（data_prepare 产出，审核之前就落盘）
    cleaned_candidate: CleanedDataRef | None
    # 审核通过的数据（data_review 产出，指向候选文件）；`cleaned_data` 语义变为
    # "审核通过的数据"，因此打回时为 None，④⑤ 也不会读到未过审的数据。
    cleaned_data: CleanedDataRef | None
    param_set: ParamSet
    review_log: Annotated[list[ReviewRecord], append_list]
    compute_result: ComputeResult | None
    chart_path: str
    draft_content: DraftContent | None
    draft_path: str
    # 按 kind 覆盖：每种产物只有一份当前有效，重跑后旧引用失效（P1-3）
    artifacts: Annotated[dict[str, ArtifactRef], merge_dict]
    validation_result: ValidationReport | None
    llm_calls: Annotated[list[LLMCallRecord], append_list]
    step_status: Annotated[dict[str, str], merge_dict]
    # 每次节点执行一条审计记录（列表追加）；step_status 是"最新结论"（字典覆盖）
    step_records: Annotated[list[StepRecord], append_list]
    termination: Termination | None
    error: ErrorRecord | None
    # 每次进入 attribute 追加一条归因审计记录（P1-3 追加语义）
    attributions: Annotated[list[AttributionRecord], append_list]
    # 检索 Agent 的运行轨迹（仅 agent 开启且真实模式时写入；重跑整体替换）
    retrieval_trace: RetrievalTrace | None


def mark_step(state: GraphState, step: str, status: str) -> dict[str, Any]:
    """生成 step_status 的增量更新。"""
    return {"step_status": {step: status}}

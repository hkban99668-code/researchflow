"""A11：含 `step_records` 的 state 经 checkpointer 序列化往返后不变。

`StepRecord` 是 T-004 新进入 checkpoint 的类型。T-001 时漏登记 msgpack 白名单出过
问题，这里锁住：未登记时 jsonplus 会拒绝重建类、退化成字典（字段还会丢），
往返结果与原值不相等。

T-008 追加 A1：白名单改为自动收集后，一个**所有字段都填满**的 GraphState 经
`make_serializer()` 往返与原值 `==`，且收集函数对 schemas 模块的结果覆盖该模块
定义的每一个 BaseModel 子类（新增一个子类无需改 orchestrator）。
"""

from __future__ import annotations

import inspect

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from researchflow.core import schemas
from researchflow.core.schemas import (
    ArtifactRef,
    AttributionDecision,
    AttributionRecord,
    BaseModel,
    CheckResult,
    CleanedDataRef,
    ComputeResult,
    DraftContent,
    ErrorRecord,
    LiteratureRef,
    LLMCallRecord,
    LLMTurnTrace,
    LLMUsage,
    ParamSet,
    QualitySummary,
    ResearchQuestion,
    ReviewRecord,
    StepRecord,
    ToolCallTrace,
    ValidationReport,
)
from researchflow.core.state import GraphState
from researchflow.graphs.orchestrator import (
    ALLOWED_MSGPACK_MODULES,
    collect_msgpack_modules,
    make_serializer,
)


def _state_with_records() -> dict:
    return {
        "step_records": [
            StepRecord(
                step="extraction",
                status="ok",
                attempt=1,
                started_at="2026-10-09T00:00:00+00:00",
                duration_ms=12,
                llm_usage=LLMUsage(prompt_tokens=3, completion_tokens=4),
            ),
            StepRecord(
                step="data_prepare",
                status="error",
                attempt=2,
                started_at="2026-10-09T00:00:01+00:00",
                duration_ms=5,
                error=ErrorRecord(code="RF-003", step="data_prepare", attempt=2),
            ),
        ]
    }


def test_step_record_is_registered_in_msgpack_allowlist():
    assert ("researchflow.core.schemas", "StepRecord") in ALLOWED_MSGPACK_MODULES


def test_state_with_step_records_survives_checkpoint_roundtrip():
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    state = _state_with_records()

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert restored == state
    assert all(isinstance(record, StepRecord) for record in restored["step_records"])
    assert restored["step_records"][1].error == state["step_records"][1].error
    assert restored["step_records"][0].llm_usage == state["step_records"][0].llm_usage


# --- T-008 A1：自动收集白名单 + 全字段往返 ------------------------------------


def _full_state() -> dict:
    """GraphState 的每一个字段都填上值（含嵌套的 QualitySummary / CheckResult）。"""
    quality = QualitySummary(
        rows=58,
        columns=4,
        missing_rate=0.03,
        duplicates_removed=1,
        raw_missing_rate=0.12,
        dropped_rows=3,
        notes=["预检通过"],
    )
    cleaned = CleanedDataRef(
        path="cleaned_candidate.csv",
        checksum="a" * 64,
        rows=58,
        columns=["x1", "x2", "x3", "y"],
        quality=quality,
        raw_checksum="b" * 64,
    )
    state: GraphState = {
        "run_id": "serde-run",
        "topic": "科研数据质量与线性回归可复现性评估",
        "literature_refs": [
            LiteratureRef(
                ref_id="local:abc123def456",
                title="一篇文献",
                source="local",
                year=2020,
                url="https://example.org",
                summary="摘要",
                score=2.0,
                retriever="local",
                chunk_id=None,
            )
        ],
        "research_question": ResearchQuestion(
            question="变量如何影响目标变量？",
            variables=["x1", "x2", "x3", "y"],
            hypothesis="存在可量化的线性关系。",
            method="单算法回归。",
            confidence=0.6,
        ),
        "cleaned_candidate": cleaned,
        "cleaned_data": cleaned,
        "param_set": ParamSet(
            algorithm="linear_regression",
            params={
                "dataset_path": "examples/sample_dataset.csv",
                "target": "y",
                "features": ["x1", "x2", "x3"],
            },
            seed=42,
        ),
        "review_log": [
            ReviewRecord(
                stage="data_review",
                decision="approve",
                reason="通过",
                reviewer="cli",
                subject_path="cleaned_candidate.csv",
                subject_checksum="a" * 64,
                raw_checksum="b" * 64,
                timestamp="2026-10-09T00:00:00+00:00",
            )
        ],
        "compute_result": ComputeResult(
            algorithm="linear_regression",
            metrics={"r2": 0.95},
            parameters={"target": "y", "features": ["x1", "x2", "x3"]},
            summary="拟合良好",
        ),
        "chart_path": "chart.png",
        "draft_content": DraftContent(background="背景", findings="发现", limitation="局限"),
        "draft_path": "draft.md",
        "artifacts": {
            "cleaned_data": ArtifactRef(
                kind="cleaned_data",
                path="cleaned_candidate.csv",
                checksum="a" * 64,
                produced_by="data_review",
            )
        },
        "validation_result": ValidationReport(
            passed=True,
            checks=[
                CheckResult(name="r2", passed=True, detail="0.95 >= 0.8", owner_step="compute")
            ],
            summary="全部通过",
        ),
        "llm_calls": [
            LLMCallRecord(
                step="extraction",
                response_model="ResearchQuestion",
                usage=LLMUsage(prompt_tokens=100, completion_tokens=50, attempts=1),
                structured_output="json_schema",
            )
        ],
        "step_status": {"retrieval": "ok", "extraction": "ok"},
        "step_records": [
            StepRecord(
                step="retrieval",
                status="ok",
                attempt=1,
                started_at="2026-10-09T00:00:00+00:00",
                duration_ms=3,
                error=None,
                llm_usage=None,
            )
        ],
        "termination": None,
        "error": None,
    }
    return dict(state)


def test_allowlist_is_auto_collected_from_schemas():
    """白名单由代码自动收集：对 schemas 模块的结果覆盖该模块每个 BaseModel 子类。

    新增一个 BaseModel 子类不需要改 orchestrator——收集函数会在下次 import 时自动
    把它纳入。这里用 inspect **独立**枚举一遍做对照，防止收集逻辑退化回手工列表。
    """
    collected = {name for _, name in collect_msgpack_modules(schemas)}
    defined = {
        name
        for name, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    }

    assert collected == defined
    assert {"QualitySummary", "CheckResult", "StepRecord", "ReviewRecord"} <= collected
    # 只收本模块定义的类：import 进来的 pydantic 基类不能混进白名单
    assert "BaseModel" not in collected


def test_full_graph_state_survives_serializer_roundtrip():
    """A1：所有字段都填满的 GraphState 经 make_serializer() 往返与原值相等。"""
    serde = make_serializer()
    state = _full_state()

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert restored == state
    # 嵌套类型也必须还原成原类，而不是退化成字典（T-001 的教训）
    assert isinstance(restored["cleaned_candidate"].quality, QualitySummary)
    assert isinstance(restored["validation_result"].checks[0], CheckResult)
    assert isinstance(restored["review_log"][0], ReviewRecord)
    assert isinstance(restored["step_records"][0], StepRecord)


def test_attribution_with_llm_turns_survives_roundtrip_and_legacy_defaults():
    """T-025 第 2 轮：`llm_turns` / `ToolCallTrace.turn` 进 checkpoint 往返不丢；
    旧 checkpoint 的 `LLMCallRecord` 没有 duration_ms / attempt 时两者为 None。"""
    record = AttributionRecord(
        attempt=1,
        trigger="validation_failed",
        decision=AttributionDecision(action="escalate", reason="r", source="agent"),
        tool_calls=[ToolCallTrace(name="t", ok=True, duration_ms=1, turn=2)],
        llm_turns=[
            LLMTurnTrace(
                turn=2, usage=LLMUsage(prompt_tokens=1, completion_tokens=2), duration_ms=5
            )
        ],
    )
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    restored = serde.loads_typed(serde.dumps_typed({"attributions": [record]}))
    assert restored["attributions"] == [record]
    assert isinstance(restored["attributions"][0].llm_turns[0], LLMTurnTrace)

    # 旧记录：缺字段 → None（trace 生成对这两者都有 None 分支）
    legacy = LLMCallRecord.model_validate(
        {"step": "extraction", "response_model": "ResearchQuestion", "usage": {"prompt_tokens": 1}}
    )
    assert legacy.duration_ms is None
    assert legacy.attempt is None

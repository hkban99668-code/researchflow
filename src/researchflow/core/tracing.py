"""Tracing：运行结束后从 state 里已有的记录生成 OpenTelemetry 结构的 span 树。

设计判断（T-025）：
1. 本地优先、导出可选。默认只写 `runs/<id>/trace.json`，只有设置 `RF_OTLP_ENDPOINT`
   才通过 OTLP/HTTP 导出；导出直接用 httpx 拼 OTLP JSON，不依赖任何 SDK。
2. span 从 `step_records` / `llm_calls` / `attributions` / `retrieval_trace` 组装，
   不在节点里埋点，时间戳全部取自记录里已有的 `started_at` 与 `duration_ms`，
   生成时绝不读当前时间，保证同一份 state 两次生成的 trace 逐字节相同（确定性）。
3. 成本用 `configs/pricing.yaml` 里的价格表算，币种按模型（`models` 里的 `currency`
   覆盖顶部的默认 `currency`，T-027），查不到的模型成本记 null，不猜；不同币种
   分开汇总，绝不相加。

层级：`run` → `step`（每个 `StepRecord` 一个）→ `llm`（该 step 的 `LLMCallRecord`，
以及归因 / 检索 Agent 回路的每轮 `chat()`）和 `tool`（Agent 的工具调用）。Agent 的
llm 与 tool 按发生顺序交错。span_id 由 `(run_id, 层级路径, 序号)` 哈希得到，
`trace_id` 由 run_id 补零到 32 位十六进制，都确定性可复现。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, Field

from .schemas import (
    AttributionRecord,
    LLMCallRecord,
    LLMTurnTrace,
    LLMUsage,
    RetrievalTrace,
    StepRecord,
    ToolCallTrace,
)

TRACE_ID_LEN = 32
SPAN_ID_LEN = 16

SpanStatus = Literal["ok", "error", "rejected", "escalated"]

# OTLP span kind / status 枚举值（见 OTLP JSON 编码规范）。
_OTLP_KIND_INTERNAL = 1
_OTLP_KIND_CLIENT = 3
_OTLP_STATUS_OK = 1
_OTLP_STATUS_ERROR = 2
_OTLP_TIMEOUT_SECONDS = 5.0

# 币种符号；表里没登记的币种退化为 "0.0021 XXX"。
_CURRENCY_SYMBOLS = {"USD": "$", "CNY": "¥", "RMB": "¥"}


class ModelPricing(BaseModel):
    """单个模型的价格：每百万输入/输出 token 单价与可选的模型级币种。

    `currency` 为 None 时回落到价格表顶部的默认币种（`Pricing.currency`）——
    旧价格文件只有顶部币种，加载后每个模型的 `currency` 都是 None，行为不变。
    """

    input_per_m: float | None = None
    output_per_m: float | None = None
    currency: str | None = None


class Pricing(BaseModel):
    """模型价格表：默认币种 + 每个模型的每百万输入/输出 token 单价。

    `currency` 是没写模型级币种时的默认值；span 与 manifest 按各自模型的币种记录，
    不同币种分开汇总，绝不相加。
    """

    currency: str = "USD"
    models: dict[str, ModelPricing] = Field(default_factory=dict)

    def currency_for(self, model: str | None) -> str:
        """某次调用的计价币种：模型级覆盖优先，否则顶部默认。"""
        if model:
            entry = self.models.get(model)
            if entry is not None and entry.currency:
                return entry.currency
        return self.currency


class Span(BaseModel):
    """一个 span：进入 `trace.json` 的 `spans` 列表。

    `duration_ms` 对 step/tool/run 恒有值；llm span 有真实调用耗时时用它，
    没有（旧 checkpoint）时记 `None` 并在 attributes 里标 `timing: "inherited"`。
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: Literal["run", "step", "llm", "tool", "rerank"]
    start_ts: str
    duration_ms: int | None
    status: SpanStatus
    attributes: dict[str, Any] = Field(default_factory=dict)


def load_pricing(path: Path) -> Pricing:
    """读价格表；文件缺失或格式不对时返回空的 `Pricing()`（币种取默认值）。

    每个模型条目可以是带 `currency` 的字典（模型级币种），也可以是 `null`（无公开
    价格，成本记 None）；旧文件（只有顶部币种、条目无 `currency`）行为不变。
    """
    if not path.exists():
        return Pricing()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models_raw = raw.get("models") or {}
    currency = raw.get("currency")
    models: dict[str, ModelPricing] = {}
    if isinstance(models_raw, dict):
        for name, entry in models_raw.items():
            if isinstance(entry, dict):
                entry_currency = entry.get("currency")
                models[str(name)] = ModelPricing(
                    input_per_m=entry.get("input_per_m"),
                    output_per_m=entry.get("output_per_m"),
                    currency=str(entry_currency) if entry_currency else None,
                )
            else:
                # null 或旧的非字典条目：整条价格未配置（不猜），币种回落默认。
                models[str(name)] = ModelPricing()
    return Pricing(
        currency=str(currency) if currency else "USD",
        models=models,
    )


def compute_cost_with_currency(
    model: str | None, usage: LLMUsage, pricing: Pricing
) -> tuple[float | None, str]:
    """按 token 数算一次调用的成本，返回 `(成本, 币种)`。

    `fake` 成本恒为 0（不算"未配置"）；真实模型查不到价格、或价格为 null 时成本为
    None（未配置，不猜），币种仍返回该模型的计价币种。
    """
    currency = pricing.currency_for(model)
    if model == "fake":
        return 0.0, currency
    if not model:
        return None, currency
    price = pricing.models.get(model)
    if not price:
        return None, currency
    if price.input_per_m is None or price.output_per_m is None:
        return None, currency
    cost = (
        usage.prompt_tokens / 1_000_000 * price.input_per_m
        + usage.completion_tokens / 1_000_000 * price.output_per_m
    )
    return cost, currency


def compute_cost(model: str | None, usage: LLMUsage, pricing: Pricing) -> float | None:
    """按 token 数算一次调用的成本（该模型的计价币种，见 `compute_cost_with_currency`）。"""
    cost, _currency = compute_cost_with_currency(model, usage, pricing)
    return cost


def _money(cost: float, currency: str) -> str:
    """成本按币种渲染，如 `$0.0021`；未知币种退化为 `0.0021 XXX`。"""
    symbol = _CURRENCY_SYMBOLS.get(currency.upper())
    if symbol:
        return f"{symbol}{cost:.4f}"
    return f"{cost:.4f} {currency}"


def _span_id(run_id: str, *parts: object) -> str:
    """由 `(run_id, 层级路径, 序号)` 哈希出 16 位十六进制 span_id（确定性）。"""
    key = ":".join([str(run_id), *(str(part) for part in parts)])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:SPAN_ID_LEN]


# step 状态 → span 状态；未列出的（如 "empty"）视为正常完成。
_STEP_STATUS_MAP: dict[str, SpanStatus] = {
    "ok": "ok",
    "error": "error",
    "rejected": "rejected",
    "escalated": "escalated",
}


def _span_status(status: str) -> SpanStatus:
    return _STEP_STATUS_MAP.get(status, "ok")


def _run_status(state: dict) -> SpanStatus:
    """run span 的状态：按终止语义映射，无终止则看 error。"""
    termination = state.get("termination")
    kind: str | None = None
    if isinstance(termination, dict):
        kind = termination.get("kind")
    elif termination is not None:
        kind = getattr(termination, "kind", None)
    if kind == "error":
        return "error"
    if kind == "rejected":
        return "rejected"
    if kind == "escalated":
        return "escalated"
    return "error" if state.get("error") else "ok"


def _as_step_records(raw: Any) -> list[StepRecord]:
    return [
        record if isinstance(record, StepRecord) else StepRecord.model_validate(record)
        for record in (raw or [])
    ]


def _as_llm_calls(raw: Any) -> list[LLMCallRecord]:
    return [
        call if isinstance(call, LLMCallRecord) else LLMCallRecord.model_validate(call)
        for call in (raw or [])
    ]


def _as_attributions(raw: Any) -> list[AttributionRecord]:
    return [
        record
        if isinstance(record, AttributionRecord)
        else AttributionRecord.model_validate(record)
        for record in (raw or [])
    ]


def _as_retrieval_trace(raw: Any) -> RetrievalTrace | None:
    if raw is None:
        return None
    return raw if isinstance(raw, RetrievalTrace) else RetrievalTrace.model_validate(raw)


def _llm_span(
    run_id: str,
    trace_id: str,
    parent_span_id: str,
    step_name: str,
    attempt: int,
    index: int,
    call: LLMCallRecord,
    step_start_ts: str,
    model: str | None,
    pricing: Pricing,
) -> Span:
    """一条结构化调用的 llm span：有真实耗时就用它，没有则继承所属 step。"""
    cost, currency = compute_cost_with_currency(model, call.usage, pricing)
    # 失败那次调用由 @step 记成 response_model=""，据此标记为 error。
    status: SpanStatus = "error" if call.response_model == "" else "ok"
    attributes: dict[str, Any] = {
        "model": model,
        "structured_output": call.structured_output,
        "prompt_tokens": call.usage.prompt_tokens,
        "completion_tokens": call.usage.completion_tokens,
        "total_tokens": call.usage.total_tokens,
        "attempts": call.usage.attempts,
        "cost": cost,
        "currency": currency,
    }
    if call.duration_ms is None:
        attributes["timing"] = "inherited"
    return Span(
        trace_id=trace_id,
        span_id=_span_id(run_id, "step", step_name, attempt, "llm", index),
        parent_span_id=parent_span_id,
        name=f"llm:{call.response_model or 'unknown'}",
        kind="llm",
        start_ts=step_start_ts,
        duration_ms=call.duration_ms,
        status=status,
        attributes=attributes,
    )


def _agent_llm_span(
    run_id: str,
    trace_id: str,
    parent_span_id: str,
    step_name: str,
    attempt: int,
    start_ts: str,
    turn: LLMTurnTrace,
    model: str | None,
    pricing: Pricing,
) -> Span:
    """Agent 回路里一轮 `chat()` 的 llm span，挂在所属 step 下。

    `start_ts` 由 `_interleave_agent_spans` 按交错顺序累加得到，不是 step 的开始时间。
    """
    cost, currency = compute_cost_with_currency(model, turn.usage, pricing)
    attributes: dict[str, Any] = {
        "model": model,
        "turn": turn.turn,
        "prompt_tokens": turn.usage.prompt_tokens,
        "completion_tokens": turn.usage.completion_tokens,
        "total_tokens": turn.usage.total_tokens,
        "attempts": turn.usage.attempts,
        "cost": cost,
        "currency": currency,
    }
    if turn.duration_ms is None:
        attributes["timing"] = "inherited"
    return Span(
        trace_id=trace_id,
        span_id=_span_id(run_id, "step", step_name, attempt, "llm", f"turn{turn.turn}"),
        parent_span_id=parent_span_id,
        name="llm:tool_loop",
        kind="llm",
        start_ts=start_ts,
        duration_ms=turn.duration_ms,
        status="ok",
        attributes=attributes,
    )


def _tool_span(
    run_id: str,
    trace_id: str,
    parent_span_id: str,
    step_name: str,
    attempt: int,
    index: int,
    start_ts: str,
    trace: ToolCallTrace,
) -> Span:
    """一条 tool span：耗时取自 ToolResult，开始时间由调用方按交错顺序累加。"""
    status: SpanStatus = "ok" if trace.ok else "error"
    return Span(
        trace_id=trace_id,
        span_id=_span_id(run_id, "step", step_name, attempt, "tool", index),
        parent_span_id=parent_span_id,
        name=f"tool:{trace.name}",
        kind="tool",
        start_ts=start_ts,
        duration_ms=trace.duration_ms,
        status=status,
        attributes={
            "arguments_digest": trace.arguments_digest,
            "ok": trace.ok,
            "error_code": trace.error_code,
        },
    )


def _offset_ts(start_ts: str, offset_ms: int) -> str:
    """在毫秒级 ISO 时间戳上加 `offset_ms`；解析失败或空值原样返回。

    Agent span 的顺序靠 trace.json 的列表顺序，但 Jaeger / Langfuse 按开始时间排
    瀑布图：全部等于 step 开始时间会让它们叠在一起，看不出"第几轮慢在哪"。
    """
    if not start_ts or offset_ms == 0:
        return start_ts
    try:
        base = datetime.fromisoformat(start_ts)
    except ValueError:
        return start_ts
    return (base + timedelta(milliseconds=offset_ms)).isoformat(timespec="milliseconds")


def _interleave_agent_spans(
    run_id: str,
    trace_id: str,
    parent_span_id: str,
    step_name: str,
    attempt: int,
    step_start_ts: str,
    turns: list[LLMTurnTrace],
    tools: list[ToolCallTrace],
    model: str | None,
    pricing: Pricing,
) -> list[Span]:
    """把 Agent 回路的 llm 轮次与它触发的 tool 按发生顺序交错。

    顺序：第 1 轮 llm → 该轮触发的 tool → 第 2 轮 llm → …。每个 span 的开始时间 =
    step 开始时间 + 同一 step 内排在它前面的 agent span 的耗时之和（耗时为 None 按 0）：
    这样瀑布图能看出"第 1 轮 → 工具 → 第 2 轮"的先后。`llm_turns` 为空的旧记录退回
    "只挂 tool"的旧行为；`turn` 对不上任何轮次的 tool 补在该 step 末尾，不丢。
    """
    spans: list[Span] = []
    elapsed = 0
    tools_by_turn: dict[int, list[tuple[int, ToolCallTrace]]] = {}
    for index, tool in enumerate(tools):
        tools_by_turn.setdefault(tool.turn, []).append((index, tool))

    accounted: set[int] = set()
    for turn in turns:
        accounted.add(turn.turn)
        spans.append(
            _agent_llm_span(
                run_id,
                trace_id,
                parent_span_id,
                step_name,
                attempt,
                _offset_ts(step_start_ts, elapsed),
                turn,
                model,
                pricing,
            )
        )
        elapsed += turn.duration_ms or 0
        for index, tool in tools_by_turn.get(turn.turn, []):
            spans.append(
                _tool_span(
                    run_id,
                    trace_id,
                    parent_span_id,
                    step_name,
                    attempt,
                    index,
                    _offset_ts(step_start_ts, elapsed),
                    tool,
                )
            )
            elapsed += tool.duration_ms or 0

    for index, tool in enumerate(tools):
        if tool.turn not in accounted:
            spans.append(
                _tool_span(
                    run_id,
                    trace_id,
                    parent_span_id,
                    step_name,
                    attempt,
                    index,
                    _offset_ts(step_start_ts, elapsed),
                    tool,
                )
            )
            elapsed += tool.duration_ms or 0
    return spans


def build_trace(
    state: dict,
    run_id: str,
    *,
    model: str | None = None,
    pricing: Pricing | None = None,
) -> dict:
    """由 state 组装 span 树，返回 `{"trace_id", "currency", "spans"}`。

    `pricing` 传入价格表（`load_pricing` 的返回值）；不传则币种取默认、成本全为 null。
    llm 调用优先按 `(step, attempt)` 挂接；`attempt` 为 None 的旧记录退回按出现顺序。
    Agent 回路的 `tool_loop` 汇总记录会被跳过（已由逐轮 span 表示，见 D-015）。

    `model` 对所有 llm span 用同一个模型名（当前归因与检索用同一模型）；以后两者改用
    不同模型时，需要改成按 span 的来源取模型，否则成本会算错。
    """
    pricing = pricing or Pricing()
    step_records = _as_step_records(state.get("step_records"))
    llm_calls = _as_llm_calls(state.get("llm_calls"))
    attributions = _as_attributions(state.get("attributions"))
    retrieval_trace = _as_retrieval_trace(state.get("retrieval_trace"))
    rerank_trace = retrieval_trace.rerank if retrieval_trace is not None else None

    trace_id = str(run_id).zfill(TRACE_ID_LEN)
    run_span_id = _span_id(run_id, "run")
    run_start = step_records[0].started_at if step_records else ""
    run_duration = sum(record.duration_ms for record in step_records)

    run_span = Span(
        trace_id=trace_id,
        span_id=run_span_id,
        parent_span_id=None,
        name="run",
        kind="run",
        start_ts=run_start,
        duration_ms=run_duration,
        status=_run_status(state),
        attributes={},
    )

    spans: list[Span] = [run_span]

    # llm 调用按 (step, attempt) 分组；attempt 为 None 的旧记录退回按出现顺序。
    llm_by_step_attempt: dict[tuple[str, int], list[LLMCallRecord]] = {}
    legacy_by_step: dict[str, list[LLMCallRecord]] = {}
    for call in llm_calls:
        if call.attempt is None:
            legacy_by_step.setdefault(call.step, []).append(call)
        else:
            llm_by_step_attempt.setdefault((call.step, call.attempt), []).append(call)

    # 归因 Agent 的 tool / llm 轮次按 attempt 分组，挂到对应 attempt 的 attribute step 下。
    attr_tools_by_attempt: dict[int, list[ToolCallTrace]] = {}
    attr_turns_by_attempt: dict[int, list[LLMTurnTrace]] = {}
    for attr in attributions:
        attr_tools_by_attempt.setdefault(attr.attempt, []).extend(attr.tool_calls)
        attr_turns_by_attempt.setdefault(attr.attempt, []).extend(attr.llm_turns)

    retrieval_tools = retrieval_trace.tool_calls if retrieval_trace is not None else []
    retrieval_turns = retrieval_trace.llm_turns if retrieval_trace is not None else []
    last_retrieval_pos: int | None = None
    last_retrieval_span_id: str | None = None
    last_retrieval_start_ts = ""
    # 检索 Agent 的 llm / tool 挂到**最后一个** retrieval step 下（RetrievalTrace 是
    # 覆盖语义，只反映最后一次执行）；tool_loop 汇总记录的抑制也只看那一次。
    retrieval_attempts = [r.attempt for r in step_records if r.step == "retrieval"]
    last_retrieval_attempt = retrieval_attempts[-1] if retrieval_attempts else 1

    occ: dict[str, int] = {}
    for record in step_records:
        name = record.step
        k = occ.get(name, 0)
        occ[name] = k + 1
        step_span_id = _span_id(run_id, "step", name, k + 1)
        attrs: dict[str, Any] = {"attempt": record.attempt}
        if record.error is not None:
            attrs["error_code"] = record.error.code
        spans.append(
            Span(
                trace_id=trace_id,
                span_id=step_span_id,
                parent_span_id=run_span_id,
                name=name,
                kind="step",
                start_ts=record.started_at,
                duration_ms=record.duration_ms,
                status=_span_status(record.status),
                attributes=attrs,
            )
        )
        if name == "retrieval":
            last_retrieval_pos = len(spans) - 1
            last_retrieval_span_id = step_span_id
            last_retrieval_start_ts = record.started_at

        # Agent 回路的用量由逐轮 `llm:tool_loop` span 表示，这里跳过 `tool_loop`
        # 汇总记录，避免同一份用量出现两个 span（D-015）。旧 checkpoint 没有逐轮
        # 记录（`llm_turns` 为空）时保留汇总记录，trace 不丢用量。
        has_turn_spans = bool(
            (name == "attribute" and attr_turns_by_attempt.get(record.attempt))
            or (
                name == "retrieval" and record.attempt == last_retrieval_attempt and retrieval_turns
            )
        )
        keyed = llm_by_step_attempt.get((name, record.attempt))
        if keyed is None:
            calls: list[LLMCallRecord] = []
            legacy = legacy_by_step.get(name, [])
            if k < len(legacy):
                calls = [legacy[k]]
        else:
            calls = [
                call
                for call in keyed
                if not (has_turn_spans and call.response_model == "tool_loop")
            ]
        for index, call in enumerate(calls):
            # 嵌入记录必须按**嵌入模型**计价（T-034）：`model` 是对话模型，用它算
            # 嵌入 token 的单位会错。模型名由 ① 节点写进 RetrievalTrace.rerank。
            call_model = model
            if (
                call.response_model == "embedding"
                and retrieval_trace is not None
                and retrieval_trace.rerank is not None
                and retrieval_trace.rerank.embedding_model
            ):
                call_model = retrieval_trace.rerank.embedding_model
            spans.append(
                _llm_span(
                    run_id,
                    trace_id,
                    step_span_id,
                    name,
                    k + 1,
                    index,
                    call,
                    record.started_at,
                    call_model,
                    pricing,
                )
            )

        if name == "attribute":
            spans.extend(
                _interleave_agent_spans(
                    run_id,
                    trace_id,
                    step_span_id,
                    name,
                    k + 1,
                    record.started_at,
                    attr_turns_by_attempt.get(record.attempt, []),
                    attr_tools_by_attempt.get(record.attempt, []),
                    model,
                    pricing,
                )
            )

    # 检索 Agent 的 llm / tool 挂到最后一个 retrieval step 下（RetrievalTrace 是覆盖
    # 语义，反映最后一次检索；回溯到 retrieval 时也只应挂在最后一次执行上）。
    if last_retrieval_pos is not None and last_retrieval_span_id is not None:
        retrieval_spans = _interleave_agent_spans(
            run_id,
            trace_id,
            last_retrieval_span_id,
            "retrieval",
            last_retrieval_attempt,
            last_retrieval_start_ts,
            retrieval_turns,
            retrieval_tools,
            model,
            pricing,
        )
        if retrieval_spans:
            insert_at = last_retrieval_pos + 1
            spans[insert_at:insert_at] = retrieval_spans

        # 在线重排（T-034）：重排在 Agent 回路之后，作为 retrieval step 下的
        # 一个独立 span；只在"配置开启且实际执行"时产出（跳过时 reason 非空）。
        if rerank_trace is not None and rerank_trace.enabled and rerank_trace.reason is None:
            elapsed = sum(turn.duration_ms or 0 for turn in retrieval_turns) + sum(
                tool.duration_ms for tool in retrieval_tools
            )
            spans.insert(
                last_retrieval_pos + 1 + len(retrieval_spans),
                Span(
                    trace_id=trace_id,
                    span_id=_span_id(run_id, "step", "retrieval", last_retrieval_attempt, "rerank"),
                    parent_span_id=last_retrieval_span_id,
                    name="rerank",
                    kind="rerank",
                    start_ts=_offset_ts(last_retrieval_start_ts, elapsed),
                    duration_ms=rerank_trace.duration_ms,
                    status="ok",
                    attributes={
                        "candidates": rerank_trace.candidates,
                        "queries": rerank_trace.queries,
                        "prompt_tokens": rerank_trace.prompt_tokens,
                        "embedding_model": rerank_trace.embedding_model,
                        "agent_pick_kept": rerank_trace.agent_pick_kept,
                        "replaced_ref_id": rerank_trace.replaced_ref_id,
                    },
                ),
            )

    return {
        "trace_id": trace_id,
        "currency": pricing.currency,
        "spans": [span.model_dump(mode="json") for span in spans],
    }


def trace_summary(trace: dict) -> dict:
    """从 trace 提取 manifest 的 `trace` 节：总耗时、总 token、按币种分组的总成本、
    未配置价格的调用数与最慢 3 span。

    `total_cost` 是 `{币种: 金额}` 字典：不同币种分开汇总，绝不相加；价格未配置
    的调用不计入任何币种，用 `unpriced_calls` 显式计数，不悄悄忽略。旧 trace
    （span 无 `currency` 属性）回落到顶部的 `currency`。
    """
    default_currency = trace.get("currency", "USD")
    spans = trace["spans"]
    llm_spans = [span for span in spans if span["kind"] == "llm"]
    total_tokens = sum(span["attributes"].get("total_tokens", 0) for span in llm_spans)
    total_cost: dict[str, float] = {}
    unpriced_calls = 0
    for span in llm_spans:
        cost = span["attributes"].get("cost")
        if cost is None:
            unpriced_calls += 1
            continue
        currency = span["attributes"].get("currency") or default_currency
        total_cost[currency] = total_cost.get(currency, 0.0) + cost
    total_cost = {currency: round(value, 6) for currency, value in sorted(total_cost.items())}
    run_span = next((span for span in spans if span["kind"] == "run"), None)
    slowest = sorted(
        (span for span in spans if span["kind"] != "run"),
        key=lambda span: span.get("duration_ms") or 0,
        reverse=True,
    )[:3]
    return {
        "trace_id": trace["trace_id"],
        "duration_ms": run_span["duration_ms"] if run_span else 0,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "unpriced_calls": unpriced_calls,
        "slowest_spans": [
            {"name": span["name"], "duration_ms": span.get("duration_ms")} for span in slowest
        ],
    }


def _descendant_llm(span: dict, children: dict) -> list[dict]:
    """收集某 span 子树里的全部 llm span。"""
    result: list[dict] = []
    for child in children.get(span["span_id"], []):
        if child["kind"] == "llm":
            result.append(child)
        result.extend(_descendant_llm(child, children))
    return result


def render_trace_tree(trace: dict) -> str:
    """把 span 树渲染成缩进文本，每行显示名称、耗时、token、成本（按币种）。

    成本按币种分别汇总展示（如 `$0.0021` / `¥0.0123`），不同币种不相加；价格未
    配置的调用显示"成本未配置"。旧 trace（span 无 `currency` 属性）回落到顶部的
    `currency`，旧 manifest 的数字型 `total_cost` / 字符串 `currency` 不经过本函数，
    读取方原样展示。
    """
    default_currency = trace.get("currency", "USD")
    spans = trace["spans"]
    children: dict[str | None, list[dict]] = {}
    for span in spans:
        children.setdefault(span["parent_span_id"], []).append(span)

    lines: list[str] = []

    def walk(span: dict, depth: int) -> None:
        llm_spans = _descendant_llm(span, children)
        tokens = sum(item["attributes"].get("total_tokens", 0) for item in llm_spans)
        costs_by_currency: dict[str, float] = {}
        unpriced = 0
        for item in llm_spans:
            cost = item["attributes"].get("cost")
            if cost is None:
                unpriced += 1
                continue
            currency = item["attributes"].get("currency") or default_currency
            costs_by_currency[currency] = costs_by_currency.get(currency, 0.0) + cost
        parts = [span["name"]]
        if span.get("duration_ms") is not None:
            parts.append(f"{span['duration_ms'] / 1000:.1f}s")
        if tokens:
            parts.append(f"tokens={tokens}")
        for currency in sorted(costs_by_currency):
            parts.append(_money(costs_by_currency[currency], currency))
        if unpriced and llm_spans:
            parts.append("成本未配置")
        error_code = (span.get("attributes") or {}).get("error_code")
        if error_code:
            parts.append(f"error={error_code}")
        lines.append("  " * depth + "  ".join(parts))
        for child in children.get(span["span_id"], []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return "\n".join(lines)


def _iso_to_ns(ts: str) -> int:
    """ISO 8601 时间戳 → epoch 纳秒；解析失败返回 0（仅用于 OTLP 导出）。"""
    if not ts:
        return 0
    try:
        return int(datetime.fromisoformat(ts).timestamp() * 1_000_000_000)
    except ValueError:
        return 0


def _otlp_value(value: Any) -> dict | None:
    """标量属性 → OTLP AnyValue；None / dict / list 不进属性（与旧行为一致）。"""
    # bool 必须排在 int 前面：Python 里 bool 是 int 的子类。
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return None


def _otlp_attributes(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    """span attributes → OTLP `KeyValue` 列表。"""
    encoded: list[dict[str, Any]] = []
    for key, value in attributes.items():
        any_value = _otlp_value(value)
        if any_value is not None:
            encoded.append({"key": key, "value": any_value})
    return encoded


def _otlp_span(span: dict) -> dict[str, Any]:
    """一个本地 span → OTLP JSON span（id 原样用 trace.json 的十六进制值）。"""
    start_ns = _iso_to_ns(span["start_ts"])
    duration_ns = (span.get("duration_ms") or 0) * 1_000_000
    item: dict[str, Any] = {
        "traceId": span["trace_id"],
        "spanId": span["span_id"],
        "name": span["name"],
        # llm / tool 是外部调用，映射为 CLIENT，Langfuse 之类会据此分类；其余 INTERNAL。
        "kind": _OTLP_KIND_CLIENT if span["kind"] in ("llm", "tool") else _OTLP_KIND_INTERNAL,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(start_ns + duration_ns),
        "attributes": _otlp_attributes(span.get("attributes") or {}),
        "status": {"code": _OTLP_STATUS_ERROR if span["status"] == "error" else _OTLP_STATUS_OK},
    }
    if span.get("parent_span_id"):
        item["parentSpanId"] = span["parent_span_id"]
    return item


def _otlp_body(trace: dict) -> dict[str, Any]:
    """把本地 trace 编成 OTLP/HTTP JSON：`resourceSpans[].scopeSpans[].spans[]`。"""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "researchflow"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "researchflow"},
                        "spans": [_otlp_span(span) for span in trace["spans"]],
                    }
                ],
            }
        ]
    }


def export_trace_otlp(trace: dict, *, endpoint: str, headers: dict[str, str] | None = None) -> None:
    """把 span 列表通过 OTLP/HTTP JSON 发到 `{endpoint}/v1/traces`（可选，不依赖 SDK）。

    直接用 httpx（主依赖）：超时 5 秒、不重试；非 2xx 或网络错误抛 `RuntimeError`，
    由调用方兜住并打印一行警告，不影响运行结果与退出码。导出的 traceId / spanId
    与 trace.json 完全一致，Langfuse / Jaeger 里看到的 span 能和本地文件对上。
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/v1/traces"):
        url = f"{url}/v1/traces"
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    try:
        response = httpx.post(
            url,
            json=_otlp_body(trace),
            headers=request_headers,
            timeout=_OTLP_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"OTLP 导出失败: {exc}") from exc
    if response.status_code >= 300:
        raise RuntimeError(f"OTLP 导出失败: HTTP {response.status_code}")

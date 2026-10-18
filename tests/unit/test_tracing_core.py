"""T-025 tracing 的纯单元测试：build_trace 确定性、成本（带币种）、价格表、挂接、
耗时继承、树渲染与摘要。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from researchflow.core.schemas import (
    AttributionDecision,
    AttributionRecord,
    LLMCallRecord,
    LLMTurnTrace,
    LLMUsage,
    StepRecord,
    ToolCallTrace,
)
from researchflow.core.tracing import (
    ModelPricing,
    Pricing,
    build_trace,
    compute_cost,
    compute_cost_with_currency,
    load_pricing,
    render_trace_tree,
    trace_summary,
)

RUN_ID = "a89cf62eee61"


def _usage(prompt: int, completion: int, attempts: int = 1) -> LLMUsage:
    return LLMUsage(prompt_tokens=prompt, completion_tokens=completion, attempts=attempts)


def _step(
    name: str,
    attempt: int,
    started_at: str,
    duration_ms: int,
    status: str = "ok",
    error=None,
) -> StepRecord:
    return StepRecord(
        step=name,
        status=status,
        attempt=attempt,
        started_at=started_at,
        duration_ms=duration_ms,
        error=error,
    )


def _sample_state(*, llm_duration_ms: int | None = 1000, llm_attempt: int | None = 1) -> dict:
    """一段手工构造的最小 state：3 个 step、1 次 LLM 调用。"""
    return {
        "run_id": RUN_ID,
        "step_records": [
            _step("retrieval", 1, "2026-10-13T10:00:00.000+08:00", 5),
            _step("extraction", 1, "2026-10-13T10:00:01.000+08:00", 10),
            _step("validation", 1, "2026-10-13T10:00:02.000+08:00", 3),
        ],
        "llm_calls": [
            LLMCallRecord(
                step="extraction",
                response_model="ResearchQuestion",
                usage=_usage(100, 50),
                attempt=llm_attempt,
                duration_ms=llm_duration_ms,
            ),
        ],
    }


def test_trace_id_is_run_id_zero_padded_to_32():
    trace = build_trace(_sample_state(), RUN_ID)
    assert trace["trace_id"] == "00000000000000000000" + RUN_ID
    assert len(trace["trace_id"]) == 32


def test_build_trace_deterministic_and_byte_identical():
    """A2：同一份 state 两次生成的结果逐字节相同。"""
    state = _sample_state()
    a = build_trace(state, RUN_ID)
    b = build_trace(state, RUN_ID)
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_build_trace_hierarchy_and_llm_attachment():
    trace = build_trace(_sample_state(), RUN_ID, model="fake", pricing=Pricing(currency="USD"))
    spans = trace["spans"]
    kinds = [span["kind"] for span in spans]
    assert kinds == ["run", "step", "step", "llm", "step"]

    run = spans[0]
    extraction = spans[2]
    llm = spans[3]
    assert run["name"] == "run" and run["parent_span_id"] is None
    assert extraction["name"] == "extraction" and extraction["parent_span_id"] == run["span_id"]
    assert llm["name"] == "llm:ResearchQuestion" and llm["parent_span_id"] == extraction["span_id"]
    # llm span 有真实耗时：直接用记录里的值，不再标 inherited
    assert llm["duration_ms"] == 1000
    assert "timing" not in llm["attributes"]
    assert llm["attributes"]["total_tokens"] == 150
    assert llm["attributes"]["cost"] == 0.0  # fake 成本恒 0
    assert trace["currency"] == "USD"


def test_llm_span_without_duration_inherits_step_timing():
    """旧 checkpoint 没有 duration_ms：llm span 记 null 并标 timing=inherited。"""
    trace = build_trace(_sample_state(llm_duration_ms=None), RUN_ID, model="fake")
    llm = next(span for span in trace["spans"] if span["kind"] == "llm")
    assert llm["duration_ms"] is None
    assert llm["attributes"]["timing"] == "inherited"


def test_llm_span_attaches_by_attempt_when_first_attempt_fails_before_call():
    """第 1 次执行在 complete() 前就失败、回溯后第 2 次成功：唯一一条调用挂到第 2 次。"""
    state = {
        "run_id": RUN_ID,
        "step_records": [
            _step("extraction", 1, "2026-10-13T10:00:00.000+08:00", 4, status="error"),
            _step("extraction", 2, "2026-10-13T10:00:05.000+08:00", 10),
        ],
        "llm_calls": [
            LLMCallRecord(
                step="extraction",
                response_model="ResearchQuestion",
                usage=_usage(10, 5),
                attempt=2,
                duration_ms=8,
            )
        ],
    }
    trace = build_trace(state, RUN_ID, model="fake")
    steps = [span for span in trace["spans"] if span["kind"] == "step"]
    llms = [span for span in trace["spans"] if span["kind"] == "llm"]
    assert len(llms) == 1
    assert llms[0]["parent_span_id"] == steps[1]["span_id"], "应挂到第 2 次（成功的）step 下"
    assert llms[0]["parent_span_id"] != steps[0]["span_id"]


def test_llm_calls_without_attempt_fall_back_to_occurrence_order():
    """attempt 为 None 的旧记录：退回"第 k 次 step 拿第 k 条调用"的旧行为。"""
    state = {
        "run_id": RUN_ID,
        "step_records": [
            _step("extraction", 1, "2026-10-13T10:00:00.000+08:00", 4),
            _step("extraction", 2, "2026-10-13T10:00:05.000+08:00", 10),
        ],
        "llm_calls": [
            LLMCallRecord(step="extraction", response_model="A", usage=_usage(1, 1)),
            LLMCallRecord(step="extraction", response_model="B", usage=_usage(2, 2)),
        ],
    }
    trace = build_trace(state, RUN_ID, model="fake")
    steps = [span for span in trace["spans"] if span["kind"] == "step"]
    llms = [span for span in trace["spans"] if span["kind"] == "llm"]
    assert [span["name"] for span in llms] == ["llm:A", "llm:B"]
    assert [span["parent_span_id"] for span in llms] == [steps[0]["span_id"], steps[1]["span_id"]]


def test_agent_llm_turns_interleave_with_tool_spans():
    """Agent 回路的 llm 轮次与 tool 按发生顺序交错，token 合计等于各轮之和。"""
    state = {
        "run_id": RUN_ID,
        "step_records": [_step("attribute", 1, "2026-10-13T10:00:00.000+08:00", 20)],
        "attributions": [
            AttributionRecord(
                attempt=1,
                trigger="validation_failed",
                decision=AttributionDecision(action="escalate", reason="r", source="agent"),
                tool_calls=[
                    ToolCallTrace(name="t1", ok=True, duration_ms=1, turn=1),
                    ToolCallTrace(name="t2", ok=True, duration_ms=2, turn=2),
                ],
                llm_turns=[
                    LLMTurnTrace(turn=1, usage=_usage(10, 5), duration_ms=3),
                    LLMTurnTrace(turn=2, usage=_usage(20, 5), duration_ms=4),
                ],
            )
        ],
    }
    trace = build_trace(state, RUN_ID, model="fake")
    under_attribute = [
        span
        for span in trace["spans"]
        if span["parent_span_id"]
        == next(s for s in trace["spans"] if s["name"] == "attribute")["span_id"]
    ]
    assert [span["name"] for span in under_attribute] == [
        "llm:tool_loop",
        "tool:t1",
        "llm:tool_loop",
        "tool:t2",
    ]
    llm_spans = [span for span in trace["spans"] if span["kind"] == "llm"]
    assert (
        trace_summary(trace)["total_tokens"]
        == sum(span["attributes"]["total_tokens"] for span in llm_spans)
        == 40
    )


def test_agent_spans_start_times_accumulate_in_interleave_order():
    """必改项 2：Agent span 的开始时间按交错顺序累加，不再是 step 开始时间。

    给定 llm 100ms → tool 20ms → llm 50ms，三个 span 的开始时间依次相差 100ms、20ms。
    """
    state = {
        "run_id": RUN_ID,
        "step_records": [_step("attribute", 1, "2026-10-13T10:00:00.000+08:00", 200)],
        "attributions": [
            AttributionRecord(
                attempt=1,
                trigger="validation_failed",
                decision=AttributionDecision(action="escalate", reason="r", source="agent"),
                tool_calls=[ToolCallTrace(name="t1", ok=True, duration_ms=20, turn=1)],
                llm_turns=[
                    LLMTurnTrace(turn=1, usage=_usage(10, 5), duration_ms=100),
                    LLMTurnTrace(turn=2, usage=_usage(20, 5), duration_ms=50),
                ],
            )
        ],
    }
    trace = build_trace(state, RUN_ID, model="fake")
    attribute = next(span for span in trace["spans"] if span["name"] == "attribute")
    children = [span for span in trace["spans"] if span["parent_span_id"] == attribute["span_id"]]
    assert [span["name"] for span in children] == ["llm:tool_loop", "tool:t1", "llm:tool_loop"]

    starts = [datetime.fromisoformat(span["start_ts"]) for span in children]
    deltas_ms = [(starts[index + 1] - starts[index]).total_seconds() * 1000 for index in range(2)]
    assert deltas_ms == [100, 20]
    assert starts[0] == datetime.fromisoformat(attribute["start_ts"])
    # 毫秒级 ISO 格式保持不变
    assert all(span["start_ts"].endswith("+08:00") for span in children)


def test_tool_loop_summary_record_does_not_duplicate_turn_spans():
    """必改项 1：有逐轮 llm_turns 时，tool_loop 汇总记录不再单独成 span。"""
    state = {
        "run_id": RUN_ID,
        "step_records": [_step("attribute", 1, "2026-10-13T10:00:00.000+08:00", 20)],
        # 节点追加的汇总记录：与逐轮 span 表示同一份用量
        "llm_calls": [
            LLMCallRecord(
                step="attribute",
                response_model="tool_loop",
                usage=_usage(30, 10),
                attempt=1,
                duration_ms=7,
            )
        ],
        "attributions": [
            AttributionRecord(
                attempt=1,
                trigger="validation_failed",
                decision=AttributionDecision(action="escalate", reason="r", source="agent"),
                llm_turns=[
                    LLMTurnTrace(turn=1, usage=_usage(10, 5), duration_ms=3),
                    LLMTurnTrace(turn=2, usage=_usage(20, 5), duration_ms=4),
                ],
            )
        ],
    }
    trace = build_trace(state, RUN_ID, model="fake")
    llms = [span for span in trace["spans"] if span["kind"] == "llm"]
    # 只有两条逐轮 span，没有汇总记录额外生成的那一条
    assert [span["name"] for span in llms] == ["llm:tool_loop", "llm:tool_loop"]
    assert trace_summary(trace)["total_tokens"] == 40


def test_tool_loop_summary_record_kept_when_no_turns():
    """旧 checkpoint 没有 llm_turns：保留 tool_loop 汇总记录作为 span，不丢用量。"""
    state = {
        "run_id": RUN_ID,
        "step_records": [_step("attribute", 1, "2026-10-13T10:00:00.000+08:00", 20)],
        "llm_calls": [
            LLMCallRecord(
                step="attribute",
                response_model="tool_loop",
                usage=_usage(30, 10),
                attempt=1,
                duration_ms=7,
            )
        ],
        "attributions": [
            AttributionRecord(
                attempt=1,
                trigger="validation_failed",
                decision=AttributionDecision(action="escalate", reason="r", source="agent"),
            )
        ],
    }
    trace = build_trace(state, RUN_ID, model="fake")
    llms = [span for span in trace["spans"] if span["kind"] == "llm"]
    assert [span["name"] for span in llms] == ["llm:tool_loop"]
    assert llms[0]["attributes"]["total_tokens"] == 40
    assert llms[0]["duration_ms"] == 7


def test_compute_cost():
    """A7：价格为 null 时成本 null；设置价格后按 token 数算对；fake 为 0。"""
    pricing = Pricing(
        currency="USD",
        models={"qwen3.7-flash-2026-07-15": {"input_per_m": 1.0, "output_per_m": 2.0}},
    )
    usage = _usage(1_000_000, 500_000)
    assert compute_cost("fake", usage, pricing) == 0.0
    assert compute_cost("qwen3.7-flash-2026-07-15", usage, pricing) == 2.0
    null_pricing = Pricing(
        models={"qwen3.7-flash-2026-07-15": {"input_per_m": None, "output_per_m": None}}
    )
    assert compute_cost("qwen3.7-flash-2026-07-15", usage, null_pricing) is None
    assert compute_cost("unknown-model", usage, pricing) is None
    assert compute_cost(None, usage, pricing) is None


def test_load_pricing_reads_currency(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "currency: USD\nmodels:\n  a:\n    input_per_m: 1.0\n    output_per_m: null\n",
        encoding="utf-8",
    )
    pricing = load_pricing(path)
    assert pricing.currency == "USD"
    assert pricing.models["a"].input_per_m == 1.0
    assert pricing.models["a"].output_per_m is None
    assert pricing.models["a"].currency is None  # 无模型级币种 → 回落顶部默认
    assert pricing.currency_for("a") == "USD"
    missing = load_pricing(tmp_path / "missing.yaml")
    assert missing.models == {}
    assert missing.currency == "USD"


def test_legacy_pricing_file_cost_unchanged(tmp_path):
    """A1：旧价格文件（只有顶部币种、模型条目无 `currency`）加载后算成本与改动前一致。"""
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "currency: USD\nmodels:\n  m:\n    input_per_m: 3.0\n    output_per_m: 9.0\n",
        encoding="utf-8",
    )
    pricing = load_pricing(path)
    usage = _usage(1_000_000, 500_000)
    # 1M 输入 × 3.0 + 0.5M 输出 × 9.0 = 7.5，币种取顶部 USD（无模型级覆盖）
    assert compute_cost("m", usage, pricing) == 7.5
    assert pricing.currency_for("m") == "USD"


def test_load_pricing_reads_per_model_currency(tmp_path):
    """T-027：模型级 `currency` 覆盖顶部默认；null 条目表示无公开价格。"""
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "currency: USD\n"
        "models:\n"
        "  a:\n"
        "    currency: CNY\n"
        "    input_per_m: 2.0\n"
        "    output_per_m: 8.0\n"
        "  b: null\n",
        encoding="utf-8",
    )
    pricing = load_pricing(path)
    assert pricing.currency_for("a") == "CNY"
    assert pricing.currency_for("b") == "USD"  # null 条目无价格，币种回落默认
    assert pricing.models["b"].input_per_m is None
    assert pricing.models["b"].output_per_m is None


def test_trace_summary():
    trace = build_trace(_sample_state(), RUN_ID, model="fake", pricing=Pricing(currency="USD"))
    summary = trace_summary(trace)
    assert summary["trace_id"] == trace["trace_id"]
    assert summary["duration_ms"] == 18  # 5 + 10 + 3
    assert summary["total_tokens"] == 150
    assert summary["total_cost"] == {"USD": 0.0}  # T-027：按币种分组的字典
    assert summary["unpriced_calls"] == 0
    # llm span 现在有真实耗时（1000ms），是最慢的；其后是 extraction(10) / retrieval(5)
    assert [item["name"] for item in summary["slowest_spans"]] == [
        "llm:ResearchQuestion",
        "extraction",
        "retrieval",
    ]


def test_render_trace_tree():
    trace = build_trace(_sample_state(), RUN_ID, model="fake", pricing=Pricing(currency="USD"))
    text = render_trace_tree(trace)
    lines = text.splitlines()
    assert lines[0].startswith("run ")
    assert "extraction" in text
    assert "llm:ResearchQuestion" in text
    assert "tokens=150" in text
    assert "$0.0000" in text


# --- T-027：按模型计币种 -------------------------------------------------------


def test_llm_span_carries_model_currency():
    """模型级币种进 llm span 的 attributes，与 cost 并列。"""
    pricing = Pricing(
        currency="USD",
        models={"cny-model": ModelPricing(input_per_m=3.0, output_per_m=9.0, currency="CNY")},
    )
    trace = build_trace(_sample_state(), RUN_ID, model="cny-model", pricing=pricing)
    llm = next(span for span in trace["spans"] if span["kind"] == "llm")
    assert llm["attributes"]["currency"] == "CNY"
    # 1M 输入 × 3 + 1M 输出 × 9（usage 100/50 → 远小于 1，按比例）
    assert llm["attributes"]["cost"] == pytest.approx(100 / 1e6 * 3 + 50 / 1e6 * 9)


def test_compute_cost_with_currency():
    """compute_cost_with_currency 返回 (成本, 币种)；未配置时成本 None、币种仍返回。"""
    pricing = Pricing(
        currency="USD",
        models={
            "cny-model": ModelPricing(input_per_m=2.0, output_per_m=8.0, currency="CNY"),
            "null-model": ModelPricing(),
        },
    )
    usage = _usage(1_000_000, 500_000)
    assert compute_cost_with_currency("cny-model", usage, pricing) == (6.0, "CNY")
    assert compute_cost_with_currency("null-model", usage, pricing)[0] is None
    assert compute_cost_with_currency("null-model", usage, pricing)[1] == "USD"
    assert compute_cost_with_currency("fake", usage, pricing) == (0.0, "USD")


def _multi_currency_trace() -> dict:
    """手工构造一次运行混用两种币种 + 一个未配置调用的 trace（T-027 A2）。"""
    return {
        "trace_id": "0" * 16 + RUN_ID,
        "currency": "USD",
        "spans": [
            {
                "trace_id": "0" * 16 + RUN_ID,
                "span_id": "s1",
                "parent_span_id": None,
                "name": "run",
                "kind": "run",
                "start_ts": "",
                "duration_ms": 10,
                "status": "ok",
                "attributes": {},
            },
            {
                "trace_id": "0" * 16 + RUN_ID,
                "span_id": "s2",
                "parent_span_id": "s1",
                "name": "llm:cny",
                "kind": "llm",
                "start_ts": "",
                "duration_ms": 5,
                "status": "ok",
                "attributes": {"total_tokens": 100, "cost": 0.012, "currency": "CNY"},
            },
            {
                "trace_id": "0" * 16 + RUN_ID,
                "span_id": "s3",
                "parent_span_id": "s1",
                "name": "llm:usd",
                "kind": "llm",
                "start_ts": "",
                "duration_ms": 5,
                "status": "ok",
                "attributes": {"total_tokens": 50, "cost": 0.0021, "currency": "USD"},
            },
            {
                "trace_id": "0" * 16 + RUN_ID,
                "span_id": "s4",
                "parent_span_id": "s1",
                "name": "llm:unpriced",
                "kind": "llm",
                "start_ts": "",
                "duration_ms": 5,
                "status": "ok",
                "attributes": {"total_tokens": 10, "cost": None, "currency": "USD"},
            },
        ],
    }


def test_trace_summary_groups_costs_by_currency_never_sums_them():
    """A2：两种币种分别汇总为两项，绝不相加；未配置的调用计入 unpriced_calls。"""
    summary = trace_summary(_multi_currency_trace())
    assert summary["total_cost"] == {"CNY": pytest.approx(0.012), "USD": pytest.approx(0.0021)}
    assert summary["unpriced_calls"] == 1
    assert summary["total_tokens"] == 160


def test_render_trace_tree_shows_each_currency_and_unpriced():
    """树渲染按币种分别显示，未配置的调用显示"成本未配置"。"""
    text = render_trace_tree(_multi_currency_trace())
    assert "¥0.0120" in text
    assert "$0.0021" in text
    assert "成本未配置" in text


def test_render_trace_tree_legacy_format_without_span_currency():
    """旧 trace.json（span 无 currency 属性）回落到顶部 currency，仍能显示。"""
    legacy = _multi_currency_trace()
    for span in legacy["spans"]:
        if span["kind"] == "llm":
            span["attributes"].pop("currency", None)
    text = render_trace_tree(legacy)
    # 全部按顶部 USD 渲染（0.012 + 0.0021），不报错
    assert "$0.0141" in text

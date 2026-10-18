"""T-003：有界工具回路 `run_tool_loop`（A7–A12）。

全部用 Fake 脚本模式 + T-002 的真实注册表与仓库自带示例数据，不联网、不改工具逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchflow.core.agent_loop import LoopBudget, run_tool_loop
from researchflow.core.config import RunContext, load_settings
from researchflow.core.errors import LLMError
from researchflow.core.llm import FakeLLMClient, add_usage
from researchflow.core.schemas import AssistantTurn, LLMUsage, ToolCall
from researchflow.core.tools import ToolContext, ToolRegistry
from researchflow.tools.attribution_tools import build_attribution_registry
from researchflow.tools.literature import build_retriever
from researchflow.tools.registry_tools import build_default_registry

SAMPLE_CSV = "examples/sample_dataset.csv"
FIRST_USER = {"role": "user", "content": "分析这份数据，先找文献再确认数据能不能用。"}


# --- fixtures 与辅助 -----------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry()


@pytest.fixture
def tool_ctx(project_root: Path) -> ToolContext:
    settings = load_settings(project_root=project_root)
    return ToolContext(
        workspace_root=settings.project_root,
        settings=settings,
        retriever=build_retriever(settings),
    )


@pytest.fixture
def attribution_registry(tmp_path, project_root: Path) -> ToolRegistry:
    """归因工具注册表（含 propose_backtrack / escalate_to_human 两个终态工具）。"""
    settings = load_settings(project_root=project_root)
    run_ctx = RunContext(settings=settings, run_dir=tmp_path / "runs")
    return build_attribution_registry({}, run_ctx)


def _call(call_id: str, name: str, **kwargs: Any) -> ToolCall:
    """按模型给参数的样子构造一次调用意图（arguments 是 JSON 字符串）。"""
    return ToolCall(id=call_id, name=name, arguments=json.dumps(kwargs, ensure_ascii=False))


def _turn(*calls: ToolCall) -> AssistantTurn:
    """带工具调用的回复：content 为 null 才是 OpenAI 的真实形状。"""
    return AssistantTurn(content=None, tool_calls=list(calls))


def _final(text: str) -> AssistantTurn:
    return AssistantTurn(content=text)


def _tool_ids(messages: list[dict[str, Any]]) -> list[str]:
    return [m["tool_call_id"] for m in messages if m.get("role") == "tool"]


def _call_ids(messages: list[dict[str, Any]]) -> list[str]:
    return [
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]


def _tool_message(messages: list[dict[str, Any]], call_id: str) -> str:
    return [m["content"] for m in messages if m.get("tool_call_id") == call_id][0]


# --- A7 完整回路 ---------------------------------------------------------------


def test_loop_runs_two_tools_then_answers(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    script = [
        _turn(_call("c1", "search_literature", query="回归", limit=2)),
        _turn(_call("c2", "check_data_quality", csv_path=SAMPLE_CSV)),
        _final("文献与数据都齐了，结论如下。"),
    ]
    client = FakeLLMClient(script)
    result = run_tool_loop(client, registry, tool_ctx, [FIRST_USER], LoopBudget())

    assert result.stop_reason == "final"
    assert result.final_content == "文献与数据都齐了，结论如下。"
    assert result.steps == 3
    assert [item.call.name for item in result.invocations] == [
        "search_literature",
        "check_data_quality",
    ]
    assert all(item.result.ok for item in result.invocations)

    # 第 2 次 chat 看到 c1 的结果，第 3 次看到 c1 与 c2 两条
    assert _tool_ids(client.chat_requests[0]) == []
    assert _tool_ids(client.chat_requests[1]) == ["c1"]
    assert _tool_ids(client.chat_requests[2]) == ["c1", "c2"]
    assert "refs" in _tool_message(result.messages, "c1")
    assert '"rows"' in _tool_message(result.messages, "c2")


# --- A8 一轮两个并行调用 -------------------------------------------------------


def test_parallel_tool_calls_run_in_order(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    script = [
        _turn(
            _call("c1", "search_literature", query="回归"),
            _call("c2", "check_data_quality", csv_path=SAMPLE_CSV),
        ),
        _final("完成"),
    ]
    client = FakeLLMClient(script)
    result = run_tool_loop(client, registry, tool_ctx, [FIRST_USER], LoopBudget())

    assert result.stop_reason == "final"
    assert len(result.invocations) == 2
    assert [item.call.id for item in result.invocations] == ["c1", "c2"]
    assert _tool_ids(result.messages) == ["c1", "c2"], "tool 消息顺序与 call 顺序一致"
    assert _tool_ids(client.chat_requests[1]) == ["c1", "c2"]


# --- A9 工具失败只回填，不中断 -------------------------------------------------


def test_failed_tool_result_is_fed_back_without_breaking_the_loop(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    script = [
        _turn(_call("c1", "check_data_quality", csv_path="/etc/passwd")),
        _turn(_call("c2", "check_data_quality", csv_path=SAMPLE_CSV)),
        _final("第二次取到了数据。"),
    ]
    client = FakeLLMClient(script)
    result = run_tool_loop(client, registry, tool_ctx, [FIRST_USER], LoopBudget())

    assert result.stop_reason == "final"
    assert result.invocations[0].result.ok is False
    assert result.invocations[0].result.error is not None
    assert result.invocations[0].result.error.code == "permission_denied"
    assert "permission_denied" in _tool_message(result.messages, "c1")
    assert result.invocations[1].result.ok is True
    assert _tool_ids(client.chat_requests[2]) == ["c1", "c2"], "失败结果也照样回填给模型"


# --- A10 max_steps 预算 --------------------------------------------------------


def test_max_steps_stops_with_budget_exhausted(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    script = [_turn(_call(f"c{i}", "search_literature", query="回归")) for i in range(1, 4)]
    client = FakeLLMClient(script)
    result = run_tool_loop(
        client, registry, tool_ctx, [FIRST_USER], LoopBudget(max_steps=3, max_tool_calls=10)
    )

    assert result.stop_reason == "budget_exhausted"
    assert result.steps == 3
    assert result.final_content is None
    assert len(result.invocations) == 3
    assert len(client.chat_requests) == 3


# --- A11 max_tool_calls 预算 ---------------------------------------------------


def test_max_tool_calls_leaves_every_call_id_answered(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    script = [
        _turn(
            _call("c1", "search_literature", query="回归"),
            _call("c2", "search_literature", query="线性"),
            _call("c3", "search_literature", query="最小二乘"),
        ),
        _final("不该被用到"),
    ]
    client = FakeLLMClient(script)
    result = run_tool_loop(
        client, registry, tool_ctx, [FIRST_USER], LoopBudget(max_steps=6, max_tool_calls=2)
    )

    assert result.stop_reason == "budget_exhausted"
    assert len(result.invocations) == 2, "只执行前 2 个"
    assert result.steps == 1
    assert "budget_exhausted" in _tool_message(result.messages, "c3")
    # 每个 tool_call_id 都有一条 tool 消息，否则下一次请求会被 Provider 拒绝
    assert _call_ids(result.messages) == ["c1", "c2", "c3"]
    assert _tool_ids(result.messages) == ["c1", "c2", "c3"]


# --- A12 累计用量 --------------------------------------------------------------


def test_loop_usage_is_sum_of_chat_last_usage(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    script = [
        _turn(_call("c1", "search_literature", query="回归", limit=2)),
        _turn(_call("c2", "check_data_quality", csv_path=SAMPLE_CSV)),
        _final("完成"),
    ]
    client = FakeLLMClient(script)
    result = run_tool_loop(client, registry, tool_ctx, [FIRST_USER], LoopBudget())

    expected = LLMUsage(attempts=0)
    for request, turn in zip(client.chat_requests, script, strict=True):
        expected = add_usage(
            expected,
            LLMUsage(
                prompt_tokens=len(json.dumps(request, ensure_ascii=False)),
                completion_tokens=len(turn.model_dump_json()),
                attempts=1,
            ),
        )
    assert result.usage == expected
    assert result.usage.total_tokens == expected.prompt_tokens + expected.completion_tokens
    assert result.usage.attempts == 3


# --- 回路不吞 LLMError ---------------------------------------------------------


def test_loop_propagates_llm_error(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    client = FakeLLMClient()  # 没有脚本
    with pytest.raises(LLMError):
        run_tool_loop(client, registry, tool_ctx, [FIRST_USER], LoopBudget())


# --- A3 terminal_tools：决策工具触发立即结束 -----------------------------------


def test_terminal_tool_stops_loop_and_backfills_later_calls(
    attribution_registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    """A3：调用决策工具后立即结束，`stop_reason == "terminal"`，后续调用不执行但每个
    tool_call_id 都有 tool 消息。"""
    script = [
        _turn(
            _call("c1", "propose_backtrack", target_step="compute"),
            _call("c2", "inspect_column", column="y"),
        ),
    ]
    client = FakeLLMClient(script)
    result = run_tool_loop(
        client,
        attribution_registry,
        tool_ctx,
        [FIRST_USER],
        LoopBudget(),
        terminal_tools=frozenset({"propose_backtrack", "escalate_to_human"}),
    )

    assert result.stop_reason == "terminal"
    assert result.terminal_call is not None
    assert result.terminal_call.call.id == "c1"
    # 决策工具之后的 inspect_column 不执行
    assert [item.call.name for item in result.invocations] == ["propose_backtrack"]
    assert [item.result.ok for item in result.invocations] == [True]
    # 但每个 tool_call_id 都有一条 tool 消息（Provider 要求）
    assert _call_ids(result.messages) == ["c1", "c2"]
    assert _tool_ids(result.messages) == ["c1", "c2"]
    assert "未执行" in _tool_message(result.messages, "c2")

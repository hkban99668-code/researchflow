"""有预算上限的工具回路：模型提意图 → 宿主执行 → 结果回填 → 再问模型。

职责划分（T-003 的设计判断 1）：`LLMClient.chat()` 只把 `tool_calls` 交回来，
不执行任何工具。执行、参数校验、路径权限都在本模块的 `run_tool_loop` 里完成，
于是权限控制与审计只有唯一落点——模型给的路径永远是不可信输入。

回路不吞 `LLMError`：`chat()` 抛错就往上抛，由调用方的 `@step` 收敛成 error。
工具失败不算回路失败：错误以 tool message 回填，让模型自己改参数重试。

约定（D-006 / D-015）：回路本身不写 state。调用 `run_tool_loop` 的节点负责两件事：
1. 把 `LoopResult.llm_turns` / `LoopResult.tool_calls` 写进自己的记录（归因 / 检索
   节点分别写进 `AttributionRecord`、`RetrievalTrace`），trace 据此把每轮 `chat()`
   与它触发的工具按发生顺序交错；
2. 用 `tool_loop_call_record()` 把整段回路的用量记成一条
   `response_model="tool_loop"` 的 `LLMCallRecord` 追加进 `llm_calls`，这样
   `StepRecord.llm_usage`、manifest 的 `by_step` / `total` 都包含 Agent 的用量
   （T-025 第 2 轮：Agent 的模型调用是整条运行最贵的部分）。trace 组装时跳过这条
   汇总记录（已由逐轮 `llm:tool_loop` span 表示），避免重复计数。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from .errors import LLMError
from .llm import LLMClient, add_usage
from .schemas import (
    AssistantTurn,
    LLMCallRecord,
    LLMTurnTrace,
    LLMUsage,
    ToolCall,
    ToolCallTrace,
)
from .tools import ToolContext, ToolError, ToolRegistry, ToolResult, _redact_root

ZERO_USAGE = LLMUsage(attempts=0)

# 参数摘要的截断上限（字符），超出部分丢弃；路径在此之前已脱敏。
_ARGS_DIGEST_LEN = 200

# 失败原因消息的截断上限（字符）；压平成一句，路径在此之前已脱敏。
_ERROR_MESSAGE_LEN = 200


def _sanitize_error_message(text: str, ctx: ToolContext) -> str:
    """把失败原因压成一句脱敏消息：去工作区绝对路径前缀、压平换行、截断。

    只保留类型名与一句消息（见 `ToolLoopInterrupted`），不携带 Key：错误消息由
    LLM 适配层构造，本层只做路径脱敏与截断，绝不拼接环境变量或客户端属性。
    """
    cleaned = _redact_root(" ".join(str(text).split()), ctx.workspace_root)
    return cleaned[:_ERROR_MESSAGE_LEN]


class LoopBudget(BaseModel):
    """回路的预算上限：两道闸，任一用完都以 `budget_exhausted` 停下。"""

    max_steps: int = 6  # 最多调用 chat() 的次数
    max_tool_calls: int = 10  # 全部轮次累计执行的工具次数


class ToolInvocation(BaseModel):
    """一次"模型提出 + 宿主执行"的完整记录，供审计与 Eval 使用。

    `turn` 是提出这次调用的 `chat()` 轮次（1 起），trace 据此把 llm 与 tool 交错。
    """

    call: ToolCall
    result: ToolResult
    turn: int = 1


class LoopResult(BaseModel):
    """回路的产出：完整对话（含 tool 消息）、逐次调用记录与累计用量。"""

    stop_reason: Literal["final", "budget_exhausted", "terminal"]
    final_content: str | None
    messages: list[dict[str, Any]]  # 完整对话，含 tool 消息，便于审计
    invocations: list[ToolInvocation]
    usage: LLMUsage  # 全部 chat() 的累计
    steps: int
    # 模型调用了 `terminal_tools` 之一（且参数校验通过）时的那条调用；否则为 None
    terminal_call: ToolInvocation | None = None
    # 逐次工具调用的追溯记录（T-025），由 run_tool_loop 从 invocations 生成
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    # 每一轮 chat() 的用量与耗时（T-025 第 2 轮）：Agent 的模型调用不在 llm_calls，
    # 成本与 token 统计必须靠它。
    llm_turns: list[LLMTurnTrace] = Field(default_factory=list)


class ToolLoopInterrupted(LLMError):
    """`chat()` 抛错时携带"失败前已发生"的追溯数据（T-025 第 3 轮）。

    回路照旧不吞 `LLMError`（对外仍是 RF-002），只是包一层把已完成的轮次带出来：
    Agent 失败降级转人工时，这部分 token 与工具调用必须记进 `llm_calls`，否则
    StepRecord / manifest 会静默少报。`usage` 只累计失败这一轮**之前**已完成的
    轮次——失败请求本身可能已计费，但客户端不保证读得到（Fake 耗尽时就没有），
    为避免把上一次的 `last_usage` 重复计入，不采信它。

    `error_type` / `error_message`（T-027）是脱敏后的失败原因：原始错误类型名 +
    一句消息（不含 Key、不含绝对路径），进 `AttributionRecord` / `RetrievalTrace`
    的 `failure_reason`，让事后能看见"为什么 agent_error"。
    """

    def __init__(
        self,
        original: LLMError,
        *,
        usage: LLMUsage,
        tool_calls: list[ToolCallTrace],
        llm_turns: list[LLMTurnTrace],
        invocations: list[ToolInvocation],
        ctx: ToolContext,
    ) -> None:
        super().__init__(original.message, detail=original.detail)
        self.usage = usage
        self.tool_calls = tool_calls
        self.llm_turns = llm_turns
        self.invocations = invocations
        self.error_type = type(original).__name__
        self.error_message = _sanitize_error_message(original.message, ctx)


def tool_loop_call_record(
    step_name: str, usage: LLMUsage | None, turns: list[LLMTurnTrace]
) -> list[LLMCallRecord]:
    """把一次回路的用量汇总成一条 `response_model="tool_loop"` 的调用记录。

    调用回路的节点把它追加进 `llm_calls`，让 `StepRecord.llm_usage`、manifest 的
    `by_step` / `total` 都包含 Agent 的用量（D-006 / D-015）。`duration_ms` 是各轮
    耗时之和，任一轮缺耗时则为 None；`attempt` 留空，由 `@step` 与 `StepRecord`
    同源注入。0 次调用（Agent 未配置、首轮即失败）返回空列表。
    """
    if usage is None or usage.attempts <= 0:
        return []
    durations = [turn.duration_ms for turn in turns]
    duration_ms = (
        None
        if any(value is None for value in durations)
        else sum(value or 0 for value in durations)
    )
    return [
        LLMCallRecord(
            step=step_name,
            response_model="tool_loop",
            structured_output=None,
            usage=usage,
            duration_ms=duration_ms,
        )
    ]


def _tool_calls_trace(invocations: list[ToolInvocation], ctx: ToolContext) -> list[ToolCallTrace]:
    """把逐次调用记录转成追溯记录：参数摘要脱敏并截断，耗时取自 ToolResult。"""
    traces: list[ToolCallTrace] = []
    for invocation in invocations:
        call = invocation.call
        result = invocation.result
        digest = _redact_root(call.arguments, ctx.workspace_root)[:_ARGS_DIGEST_LEN]
        traces.append(
            ToolCallTrace(
                name=call.name,
                arguments_digest=digest,
                ok=result.ok,
                error_code=result.error.code if result.error is not None else None,
                duration_ms=result.duration_ms,
                turn=invocation.turn,
            )
        )
    return traces


def run_tool_loop(
    llm: LLMClient,
    registry: ToolRegistry,
    ctx: ToolContext,
    messages: list[dict[str, Any]],
    budget: LoopBudget,
    terminal_tools: frozenset[str] = frozenset(),
) -> LoopResult:
    """跑工具回路，直到模型不再调用工具、调用到终态工具或预算用完。

    `messages` 不会被改动：回路在自己的副本上追加 assistant / tool 消息。

    `terminal_tools` 里的工具被调用且参数校验通过时，回路立即以
    `stop_reason="terminal"` 结束：同一轮里它之后的调用不再执行，但仍按 T-003
    的规则回填 tool 消息（每个 tool_call_id 都要有回应）。
    """
    history: list[dict[str, Any]] = deepcopy(messages)
    invocations: list[ToolInvocation] = []
    usage = LLMUsage(attempts=0)
    llm_turns: list[LLMTurnTrace] = []
    tools = registry.as_openai_tools()
    steps = 0

    while steps < budget.max_steps:
        try:
            turn = llm.chat(history, tools=tools, tool_choice="auto")
        except LLMError as exc:
            # 已完成的轮次不能丢：带上已发生的用量与调用记录再抛（T-025 第 3 轮）。
            raise ToolLoopInterrupted(
                exc,
                usage=usage,
                tool_calls=_tool_calls_trace(invocations, ctx),
                llm_turns=list(llm_turns),
                invocations=list(invocations),
                ctx=ctx,
            ) from exc
        steps += 1
        turn_usage = llm.last_usage or ZERO_USAGE
        usage = add_usage(usage, turn_usage)
        # 每轮 chat() 单独记一条：trace 里按发生顺序与它触发的工具交错。
        llm_turns.append(
            LLMTurnTrace(
                turn=steps,
                usage=turn_usage,
                duration_ms=getattr(llm, "last_duration_ms", None),
            )
        )
        history.append(_assistant_message(turn))

        if not turn.tool_calls:
            return LoopResult(
                stop_reason="final",
                final_content=turn.content,
                messages=history,
                invocations=invocations,
                usage=usage,
                steps=steps,
                tool_calls=_tool_calls_trace(invocations, ctx),
                llm_turns=llm_turns,
            )

        out_of_budget = False
        terminal_call: ToolInvocation | None = None
        for call in turn.tool_calls:
            if terminal_call is not None:
                # 决策工具已经触发终止：本轮的后续调用不执行，但仍回填 tool 消息。
                history.append(_tool_message(call.id, _not_executed(call.name)))
                continue
            if len(invocations) >= budget.max_tool_calls:
                # 不执行，但仍要回填：Provider 要求每个 tool_call_id 都有回应。
                out_of_budget = True
                history.append(_tool_message(call.id, _budget_exhausted(call.name)))
                continue
            result = registry.invoke(call.name, call.arguments, ctx)
            invocations.append(ToolInvocation(call=call, result=result, turn=steps))
            history.append(_tool_message(call.id, result))
            if call.name in terminal_tools and result.ok:
                terminal_call = ToolInvocation(call=call, result=result, turn=steps)

        if terminal_call is not None:
            return LoopResult(
                stop_reason="terminal",
                final_content=None,
                messages=history,
                invocations=invocations,
                usage=usage,
                steps=steps,
                terminal_call=terminal_call,
                tool_calls=_tool_calls_trace(invocations, ctx),
                llm_turns=llm_turns,
            )

        if out_of_budget:
            break

    return LoopResult(
        stop_reason="budget_exhausted",
        final_content=None,
        messages=history,
        invocations=invocations,
        usage=usage,
        steps=steps,
        tool_calls=_tool_calls_trace(invocations, ctx),
        llm_turns=llm_turns,
    )


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    """把 AssistantTurn 还原成 Chat Completions 的 assistant 消息形状。"""
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in turn.tool_calls
        ]
    return message


def _tool_message(call_id: str, result: ToolResult) -> dict[str, Any]:
    """工具结果以 JSON 字符串回填；失败的结果也一样，模型据此改参数。"""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result.model_dump_json(),
    }


def _budget_exhausted(tool_name: str) -> ToolResult:
    """预算用尽时给未执行的调用造一条结果，保证每个 tool_call_id 都有回应。"""
    return ToolResult(
        tool=tool_name,
        ok=False,
        error=ToolError(
            code="budget_exhausted",
            message="工具调用预算已用尽，本次调用未执行",
            retryable=False,
        ),
    )


def _not_executed(tool_name: str) -> ToolResult:
    """决策工具已触发终止，本轮后续调用不再执行，但同样要回填一条 tool 消息。"""
    return ToolResult(
        tool=tool_name,
        ok=False,
        error=ToolError(
            code="budget_exhausted",
            message="决策工具已提交决策，本轮后续调用未执行",
            retryable=False,
        ),
    )

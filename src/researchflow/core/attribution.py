"""归因 Agent 的机制：`Attributor` 契约与 `AgentAttributor` 实现。

`AgentAttributor` 不碰真实模型提示词打磨（那是 T-015 的事），只做机制：
用 `system_prompt` 加一条由 state 生成的情况说明作为初始消息，跑一次有界工具
回路，把决策工具的参数转成 `AttributionDecision`。模型没调用决策工具就结束、或
预算耗尽时，一律转人工（escalate），不让"沉默"被当成"成功修复"。

与任务卡接口的差异（见 T-014 汇报"偏离任务卡"）：构造器不接 `registry`——归因
工具要按最新 state 重建（回溯后 cleaned_data / validation_result 都变了），注册表
在 `decide()` 内现建；`ctx` 用 `RunContext` 而非 `ToolContext`，因为 `inspect_column`
要用 `RunContext.read_verified()` 校验清洗数据 checksum。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..tools.attribution_tools import build_attribution_registry
from .agent_loop import LoopBudget, ToolLoopInterrupted, run_tool_loop
from .config import RunContext
from .llm import LLMClient
from .schemas import AttributionDecision, BacktrackPatch, LLMTurnTrace, LLMUsage, ToolCallTrace
from .tools import ToolContext

# 模型一旦调用这两个工具之一（且参数校验通过），回路立即结束。
DECISION_TOOLS = frozenset({"propose_backtrack", "escalate_to_human"})


class Attributor(Protocol):
    """归因决策契约：给定当前 state，返回一条决策。"""

    def decide(self, state: dict) -> AttributionDecision: ...


def _decision_from_terminal_call(call: Any) -> AttributionDecision:
    """把回路的终态调用转成 `AttributionDecision(source="agent")`。

    `call.call.arguments` 已经过工具入参模型的校验（`terminal_call` 只在 `result.ok`
    时设置），这里只需原样解析。`patch` 与 `target_step` 的匹配性校验不在本层做，
    由 `attribute` 节点调用 `patch.validate_for()` 完成（越权时转人工并写 reason）。
    """
    name = call.call.name
    try:
        args = json.loads(call.call.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    if name == "propose_backtrack":
        return AttributionDecision(
            action="backtrack",
            target_step=args.get("target_step"),
            patch=BacktrackPatch.model_validate(args.get("patch") or {}),
            reason=str(args.get("reason") or ""),
            source="agent",
        )
    return AttributionDecision(
        action="escalate",
        target_step=args.get("suspected_step"),
        reason=str(args.get("reason") or ""),
        source="agent",
    )


class AgentAttributor:
    """基于工具回路的归因 Agent 机制实现（不调真实模型，提示词打磨在 T-015）。"""

    def __init__(
        self,
        llm: LLMClient,
        ctx: RunContext,
        budget: LoopBudget,
        system_prompt: str,
    ) -> None:
        self.llm = llm
        self.ctx = ctx
        self.budget = budget
        self.system_prompt = system_prompt
        # decide() 之后可读：用量与逐次工具调用追溯，由 attribute 节点写进 AttributionRecord
        self.last_usage: LLMUsage | None = None
        self.last_tool_calls: list[ToolCallTrace] = []
        # decide() 之后可读：每轮 chat() 的用量与耗时（T-025 第 2 轮）
        self.last_llm_turns: list[LLMTurnTrace] = []
        # decide() 之后可读：本次回路的逐次工具调用记录（供 Eval 真实层审计每次调用）
        self.last_invocations: list[Any] = []
        # decide() 之后可读：脱敏后的失败原因（T-027），Agent 调用失败降级时非 None
        self.last_failure_reason: str | None = None

    def _situation(self, state: dict) -> str:
        """由 state 生成一条简短情况说明，作为工具回路的初始 user 消息。"""
        lines: list[str] = []
        params = getattr(state.get("param_set"), "params", None) or {}
        if params.get("target"):
            features = "、".join(params.get("features") or []) or "（未提供）"
            lines.append(f"目标列：{params['target']}；特征列：{features}。")
        validation = state.get("validation_result")
        if validation is not None:
            failed = [check for check in validation.checks if not check.passed]
            if failed:
                lines.append("⑦ 硬规则校验未通过，失败检查如下：")
                for check in failed:
                    lines.append(f"- {check.name}（owner_step={check.owner_step}）：{check.detail}")
        error = state.get("error")
        if error is not None:
            lines.append(f"当前错误：{error.code}（step={error.step}）：{error.message}")
        compute = state.get("compute_result")
        if compute is not None:
            lines.append(f"当前计算指标：{compute.metrics}")
        lines.append(
            "请先用只读工具查明原因，再调用决策工具（propose_backtrack 或 "
            "escalate_to_human）给出决定。"
        )
        return "\n".join(lines)

    def decide(self, state: dict) -> AttributionDecision:
        """跑一次工具回路，把终态调用转成决策；失败/超预算一律转人工。"""
        # 每次决策前清空：失败时下面的 except 会写入"失败前已发生"的部分数据，
        # 不能把上一次 decide() 留下的用量错记到这一次。
        self.last_usage = None
        self.last_tool_calls = []
        self.last_llm_turns = []
        self.last_invocations = []
        self.last_failure_reason = None

        registry = build_attribution_registry(state, self.ctx)
        tool_ctx = ToolContext(
            workspace_root=self.ctx.settings.project_root,
            settings=self.ctx.settings,
            retriever=self.ctx.retriever,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._situation(state)},
        ]
        try:
            result = run_tool_loop(
                self.llm,
                registry,
                tool_ctx,
                messages,
                self.budget,
                terminal_tools=DECISION_TOOLS,
            )
        except ToolLoopInterrupted as exc:
            # 回路中途失败：把已发生的用量与调用记录留给节点，再照旧往上抛（转人工）。
            self.last_usage = exc.usage
            self.last_tool_calls = exc.tool_calls
            self.last_llm_turns = exc.llm_turns
            self.last_invocations = exc.invocations
            self.last_failure_reason = f"{exc.error_type}: {exc.error_message}"
            raise
        self.last_usage = result.usage
        self.last_tool_calls = result.tool_calls
        self.last_llm_turns = result.llm_turns
        self.last_invocations = result.invocations

        if result.stop_reason == "terminal" and result.terminal_call is not None:
            return _decision_from_terminal_call(result.terminal_call)
        if result.stop_reason == "budget_exhausted":
            reason = "工具回路预算耗尽，未得到决策"
        else:
            reason = "模型未调用决策工具就结束"
        return AttributionDecision(action="escalate", reason=reason, source="agent")

"""检索 Agent 的机制：`RetrievalAgent` 改写查询、多轮检索、自己决定"够了"。

本卡只做机制：Agent 的决策在测试里由 Fake 脚本给出，不调真实模型。`decide()`
复用 `run_tool_loop`（`terminal_tools={"finish_retrieval"}`），模型调用
`finish_retrieval` 且参数校验通过时回路立即结束，选中的 ref_id 由程序校验过
"必须来自本次检索结果"，所以可以放心按 id 回填 `searched` 里的完整文献。

停止语义：
- `terminal`（finish_retrieval 成功）→ `stop_reason="agent"`，refs 为选中的文献；
- `budget_exhausted` → `stop_reason="budget_exhausted"`，refs 回退为已检索结果里
  按相关度排在最前的 `max_refs_in_prompt` 条；
- `final`（模型没调 finish 就结束）→ 模型**尝试过** finish 但从未成功时
  `stop_reason="invalid_selection"`（T-027：不再把"选了一堆池外 id 后空手而归"
  伪装成"Agent 判断不选"，这正是 h15 一次成功运行却没选出任何文献的嫌疑），
  从未尝试过 finish 时仍是 `stop_reason="agent"`，refs 为空；
- `LLMError` → `stop_reason="agent_error"`，refs 回退为一次课题原文检索（T-014
  第 2 轮教训：兜底环节自己失败必须降级，不能让整条运行崩掉）。

失败原因（T-027）：`agent_error` / `invalid_selection` 时 `last_failure_reason`
带一句脱敏原因（错误类型名 + 消息，不含 Key 与绝对路径），由 ① 节点写进
`RetrievalTrace.failure_reason`，事后能看见"为什么 agent_error"。
"""

from __future__ import annotations

import json
from typing import Any

from ..tools.retrieval_tools import build_retrieval_registry
from .agent_loop import LoopBudget, ToolLoopInterrupted, run_tool_loop
from .config import RunContext
from .errors import LLMError
from .llm import LLMClient
from .schemas import LiteratureRef, LLMTurnTrace, LLMUsage, RetrievalDecision, ToolCallTrace
from .tools import ToolContext

# 模型调用 finish_retrieval 且参数校验通过时，回路立即结束。
DECISION_TOOLS = frozenset({"finish_retrieval"})


def selection_limit_notice(max_refs: int) -> str:
    """追加到系统提示末尾的"最多选 N 条"说明（T-028）。

    N 来自 `settings.llm.max_refs_in_prompt`，是选择上限的唯一来源：构造 Agent 时
    拼进提示，而不是把数字写进 `configs/prompts/`——配置改了提示跟着变，也不会
    出现"提示说 6、工具实现按配置改成 3"这种两处不一致。
    """
    return (
        f"\n\n## 选择数量上限\n\n"
        f"`finish_retrieval` 的 `selected_ref_ids` 最多选 {max_refs} 条，超过会被工具拒绝。"
        f"请按相关度从高到低只选最相关的、最多 {max_refs} 条。"
    )


def _top_refs(searched: dict[str, LiteratureRef], k: int) -> list[LiteratureRef]:
    """已检索集合里按相关度排在最前的 `k` 条（score 降序，同分保持检索顺序）。"""
    return sorted(searched.values(), key=lambda ref: -ref.score)[:k]


def _selected_ids_from(call: Any) -> list[str]:
    """从终态调用的参数里取 selected_ref_ids；解析失败返回空列表。

    参数已经过工具入参模型校验（`terminal_call` 只在 `result.ok` 时设置），
    这里只是把 JSON 字符串里的列表原样取出来。
    """
    try:
        args = json.loads(call.call.arguments or "{}")
    except json.JSONDecodeError:
        return []
    if not isinstance(args, dict):
        return []
    ids = args.get("selected_ref_ids")
    return [str(ref_id) for ref_id in ids] if isinstance(ids, list) else []


class RetrievalAgent:
    """基于工具回路的检索 Agent 机制实现（不调真实模型，提示词打磨以后再做）。"""

    def __init__(self, llm: LLMClient, ctx: RunContext, system_prompt: str) -> None:
        self.llm = llm
        self.ctx = ctx
        # 选择上限在构造时追加进系统提示（T-028）：上限的唯一来源是配置，
        # 模型第一次就能看到 N，不用靠"选多了被拒 → 重试"学会。
        self.system_prompt = system_prompt + selection_limit_notice(
            ctx.settings.llm.max_refs_in_prompt
        )
        # decide() 之后可读：回路的累计用量，由 ① 节点写进 RetrievalTrace。
        self.last_usage: LLMUsage | None = None
        # decide() 之后可读：本次回路的逐次工具调用追溯（T-025）。
        self.last_tool_calls: list[ToolCallTrace] = []
        # decide() 之后可读：每轮 chat() 的用量与耗时（T-025 第 2 轮）。
        self.last_llm_turns: list[LLMTurnTrace] = []
        # decide() 之后可读：本次回路的逐次工具调用记录（供审计与测试检查回填错误）。
        self.last_invocations: list[Any] = []
        # decide() 之后可读：脱敏后的失败原因（T-027），失败降级时非 None。
        self.last_failure_reason: str | None = None
        # decide() 之后可读：被注入启发式标记为 suspicious 的 ref_id（T-033），
        # 由 ① 节点写进 RetrievalTrace.suspicious_ref_ids；injection_guard 关闭时为空。
        self.last_suspicious_ref_ids: list[str] = []
        # decide() 之后可读：本次回路检索到的**全部**文献（T-034），按首次出现顺序，
        # 不只 Agent 选中的那些；① 节点把它当作在线重排的候选池。
        self.last_searched_refs: list[LiteratureRef] = []

    def decide(self, topic: str) -> RetrievalDecision:
        """跑一次工具回路，把终态调用转成决策；失败/超预算时降级回退。"""
        # 每次决策前清空，避免把上一次 decide() 留下的用量错记到这一次。
        self.last_usage = None
        self.last_tool_calls = []
        self.last_llm_turns = []
        self.last_invocations = []
        self.last_failure_reason = None
        self.last_suspicious_ref_ids = []
        self.last_searched_refs = []

        agent_cfg = self.ctx.settings.retrieval.agent
        max_refs = self.ctx.settings.llm.max_refs_in_prompt
        searched: dict[str, LiteratureRef] = {}
        queries: list[str] = []
        # 注入防护（T-033）：被启发式标记的 ref_id 收集在这里，decide() 后可读。
        flagged: list[str] = []

        registry = build_retrieval_registry(
            searched,
            queries,
            self.ctx,
            max_refs=max_refs,
            max_retrievals=agent_cfg.max_retrievals,
            flagged=flagged,
        )
        tool_ctx = ToolContext(
            workspace_root=self.ctx.settings.project_root,
            settings=self.ctx.settings,
            retriever=self.ctx.retriever,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": topic},
        ]

        try:
            result = run_tool_loop(
                self.llm,
                registry,
                tool_ctx,
                messages,
                LoopBudget(max_steps=agent_cfg.max_steps),
                terminal_tools=DECISION_TOOLS,
            )
        except ToolLoopInterrupted as exc:
            # 回路中途失败：保留已发生的用量与调用记录（节点会写进 RetrievalTrace
            # 与 llm_calls），回退为一次课题原文检索。
            self.last_usage = exc.usage
            self.last_tool_calls = exc.tool_calls
            self.last_llm_turns = exc.llm_turns
            self.last_invocations = exc.invocations
            self.last_failure_reason = f"{exc.error_type}: {exc.error_message}"
            self.last_suspicious_ref_ids = list(flagged)
            self.last_searched_refs = list(searched.values())
            return RetrievalDecision(
                refs=self._original_search(topic, max_refs),
                queries=[],
                stop_reason="agent_error",
                usage=exc.usage,
            )
        except LLMError as exc:
            # 兜底：Agent 自身失败必须降级为一次课题原文检索，不能让整条运行崩掉。
            self.last_failure_reason = f"{type(exc).__name__}: {exc.message}"
            self.last_searched_refs = list(searched.values())
            return RetrievalDecision(
                refs=self._original_search(topic, max_refs),
                queries=[],
                stop_reason="agent_error",
                usage=LLMUsage(attempts=0),
            )

        self.last_usage = result.usage
        self.last_tool_calls = result.tool_calls
        self.last_llm_turns = result.llm_turns
        self.last_invocations = result.invocations
        self.last_suspicious_ref_ids = list(flagged)
        self.last_searched_refs = list(searched.values())

        if result.stop_reason == "terminal" and result.terminal_call is not None:
            selected_ids = _selected_ids_from(result.terminal_call)
            return RetrievalDecision(
                refs=[searched[ref_id] for ref_id in selected_ids if ref_id in searched],
                queries=list(queries),
                stop_reason="agent",
                usage=result.usage,
            )

        # final：模型没调 finish 就结束。区分两种情况（T-027）：尝试过 finish 但
        # 从未成功（选的都是池外 ref_id，被工具拒绝后不再重试）说明是"选不出来"，
        # 显式记 invalid_selection；从未尝试过 finish 才是模型自己的"不选"判断。
        if result.stop_reason == "final":
            attempted_finish = any(
                invocation.call.name == "finish_retrieval" for invocation in result.invocations
            )
            if attempted_finish:
                self.last_failure_reason = (
                    "finish_retrieval 的 selected_ref_ids 全不在已检索集合里，"
                    "模型未能在预算内选出任何有效文献"
                )
            return RetrievalDecision(
                refs=[],
                queries=list(queries),
                stop_reason="invalid_selection" if attempted_finish else "agent",
                usage=result.usage,
            )
        return RetrievalDecision(
            refs=_top_refs(searched, max_refs),
            queries=list(queries),
            stop_reason="budget_exhausted",
            usage=result.usage,
        )

    def _original_search(self, topic: str, k: int) -> list[LiteratureRef]:
        """一次课题原文检索：agent_error 时的兜底，返回检索器的原始结果。"""
        if self.ctx.retriever is None:
            return []
        return self.ctx.retriever.search(topic, k=k)

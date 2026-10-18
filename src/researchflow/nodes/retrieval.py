"""① 检索：通过注入的 `Retriever` 检索并按配置截断摘要。

检索器由 `RunContext.retriever` 注入（Gate 1 是 `LocalCorpusRetriever`），
节点不 import 任何具体检索实现，换检索 API 时这里不用改。

检索 Agent（T-023）：当 `RunContext.retrieval_agent` 被注入时（main.py 只在
`retrieval.agent.enabled=true` 且真实 LLM 模式下构造），走 Agent 多轮检索；
否则保持原来的"课题原文检索一次"行为，Fake 层、Eval 基线、可复现测试都不变。

在线重排（T-034）：`retrieval.rerank.enabled=true` 且注入了 `RunContext.reranker`
时，在 Agent 结束后把本次回路 `search_literature` 返回过的**全部**文献作为候选，
用"课题原文 + Agent 改写查询"做 adaptive 多查询混合重排，取前 `top_k` 篇。
重排是 Agent 的下游一层：默认关闭，关闭时整条路径与 T-033 完全一致。

Agent 选中文献的保护（T-034）：重排结果若与 Agent 选中的文献没有任何交集，
把 Agent 排名第一的那篇替换进最后一位，并在 `RetrievalTrace.rerank` 里记录。
理由见 `docs/reviews/gate3/rerank-eval-20261017.md`：多查询重排在 Recall@10 上
高于 Agent 自选，但 Precision@5 略低，保留 Agent 的最强判断作为保险。
"""

from __future__ import annotations

import time

from langgraph.runtime import Runtime

from ..core.agent_loop import tool_loop_call_record
from ..core.config import RunContext
from ..core.errors import ConfigError
from ..core.schemas import (
    LiteratureRef,
    LLMCallRecord,
    LLMUsage,
    RerankTrace,
    RetrievalTrace,
    Termination,
)
from ..core.step import step

# Agent 降级的停止原因：这些情况下不重排，只在 trace 里写明原因（T-034 A4）。
DEGRADED_STOP_REASONS = frozenset({"agent_error", "invalid_selection"})


def _rerank_queries(topic: str, agent_queries: list[str]) -> list[str]:
    """重排用的查询集合 = 课题原文 + Agent 改写查询，去重保序、丢弃空串。"""
    seen: set[str] = set()
    queries: list[str] = []
    for query in [topic, *agent_queries]:
        text = query.strip()
        if text and text not in seen:
            seen.add(text)
            queries.append(text)
    return queries


def _embedding_model_of(reranker: object) -> str | None:
    """取重排器底座嵌入器的真实模型名（供 tracing 查价格表）。

    包装器 `model` 优先（真实模式返回 `text-embedding-v4` 这类配置值），没有时退回
    `name`（Fake 的 `hash-256`，价格表里没有 → 成本记 null，不假装免费）。
    """
    embedder = getattr(reranker, "embedder", None)
    model = getattr(embedder, "model", None)
    if isinstance(model, str) and model:
        return model
    name = getattr(embedder, "name", None)
    return name if isinstance(name, str) and name else None


def _merge_agent_pick(
    ranked: list[LiteratureRef], selected: list[LiteratureRef]
) -> tuple[list[LiteratureRef], str | None]:
    """重排结果与 Agent 选中的文献合并，返回 `(最终顺序, 被替换进来的 ref_id)`。

    有交集时原样返回重排结果；一篇都没有时把 Agent 排名第一的文献替换进末位。
    """
    final = list(ranked)
    if not selected or not final:
        return final, None
    final_ids = {ref.ref_id for ref in final}
    if final_ids & {ref.ref_id for ref in selected}:
        return final, None
    final[-1] = selected[0]
    return final, selected[0].ref_id


def _apply_rerank(
    topic: str,
    decision: object | None,
    agent: object | None,
    ctx: RunContext,
    refs: list[LiteratureRef],
) -> tuple[list[LiteratureRef], RerankTrace | None, list[LLMCallRecord]]:
    """按配置执行在线重排，返回 `(literature_refs, rerank trace, 用量记录)`。

    `decision=None`（没有 Agent）或 Agent 降级时不重排，返回原顺序并带上原因；
    关闭重排时三样都不产出，保持旧路径不变。
    """
    cfg = ctx.settings.retrieval.rerank
    if not cfg.enabled:
        return refs, None, []

    original: list[LiteratureRef] = list(refs)
    candidates: list[LiteratureRef] = list(getattr(agent, "last_searched_refs", []) or [])
    queries = _rerank_queries(topic, list(getattr(decision, "queries", []) or []))
    trace = RerankTrace(
        enabled=True,
        candidates=len(candidates),
        queries=queries,
        before_ids=[ref.ref_id for ref in candidates],
        after_ids=[ref.ref_id for ref in original],
        embedding_model=_embedding_model_of(ctx.reranker),
    )

    if decision is None:
        trace.reason = "no_agent"
        return original, trace, []
    stop_reason = getattr(decision, "stop_reason", "")
    if stop_reason in DEGRADED_STOP_REASONS:
        trace.reason = f"agent_degraded:{stop_reason}"
        return original, trace, []
    if not candidates:
        trace.reason = "no_candidates"
        return original, trace, []
    if ctx.reranker is None:
        # 开了开关却没注入重排器：明确报错，不静默退回"不重排"。
        raise ConfigError("开启 retrieval.rerank 时缺少重排器：RunContext.reranker 未注入")

    top_k = cfg.top_k or ctx.settings.llm.max_refs_in_prompt
    embedder = getattr(ctx.reranker, "embedder", None)
    reset_usage = getattr(embedder, "reset_usage", None)
    if callable(reset_usage):
        reset_usage()

    started = time.perf_counter()
    ranked = ctx.reranker.rerank(queries, candidates, top_k)
    trace.duration_ms = int((time.perf_counter() - started) * 1000)

    final, replaced = _merge_agent_pick(ranked, original)
    trace.after_ids = [ref.ref_id for ref in final]
    trace.replaced_ref_id = replaced
    trace.agent_pick_kept = bool(original) and (
        replaced is not None
        or bool({ref.ref_id for ref in final} & {ref.ref_id for ref in original})
    )

    usage = getattr(embedder, "last_usage", None) or LLMUsage(attempts=0)
    trace.prompt_tokens = usage.prompt_tokens
    calls = [
        LLMCallRecord(
            step="retrieval",
            response_model="embedding",
            usage=usage,
            duration_ms=getattr(embedder, "last_duration_ms", None),
        )
    ]
    return final, trace, calls


@step("retrieval")
def retrieval_node(state: dict, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    limit = ctx.settings.llm.max_refs_in_prompt
    max_chars = ctx.settings.llm.max_ref_chars
    retriever = ctx.retriever
    if retriever is None:
        # 不静默回退到本地实现：那样"没配检索器"会伪装成"检索成功"。
        raise ConfigError("缺少检索器：RunContext.retriever 未注入")

    agent = ctx.retrieval_agent
    decision = None
    if agent is not None:
        decision = agent.decide(state["topic"])
        refs = decision.refs
        llm_turns = list(getattr(agent, "last_llm_turns", []))
        trace: RetrievalTrace | None = RetrievalTrace(
            queries=decision.queries,
            stop_reason=decision.stop_reason,
            selected_ref_ids=[ref.ref_id for ref in decision.refs],
            usage=decision.usage,
            tool_calls=list(getattr(agent, "last_tool_calls", [])),
            llm_turns=llm_turns,
            failure_reason=getattr(agent, "last_failure_reason", None),
            suspicious_ref_ids=list(getattr(agent, "last_suspicious_ref_ids", [])),
        )
        # Agent 回路的用量记成一条 tool_loop 汇总记录，进 manifest 的 llm_usage
        # （StepRecord.llm_usage 也由此汇总）；trace 里由逐轮 llm:tool_loop span 表示。
        llm_calls = tool_loop_call_record("retrieval", decision.usage, llm_turns)
    else:
        refs = retriever.search(state["topic"], k=limit)
        trace = None
        llm_calls = []

    refs, rerank_trace, rerank_calls = _apply_rerank(state["topic"], decision, agent, ctx, refs)
    if rerank_trace is not None:
        if trace is None:
            # 没有 Agent 时本来不产出 trace；开启重排后产出最小 trace，只为记录
            # "为什么没重排"（A4）。关闭重排时 trace 仍为 None（旧行为不变）。
            trace = RetrievalTrace(stop_reason="no_agent", rerank=rerank_trace)
        else:
            trace.rerank = rerank_trace
    llm_calls.extend(rerank_calls)

    for ref in refs:
        if len(ref.summary) > max_chars:
            ref.summary = ref.summary[:max_chars] + "…"

    if not refs:
        return {
            "step_status": {"retrieval": "empty"},
            "termination": Termination(
                kind="empty", step="retrieval", reason="本地语料未命中任何文献"
            ),
            "retrieval_trace": trace,
            "llm_calls": llm_calls,
        }

    return {
        "literature_refs": refs,
        "step_status": {"retrieval": "ok"},
        "retrieval_trace": trace,
        "llm_calls": llm_calls,
    }

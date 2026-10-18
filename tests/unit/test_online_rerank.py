"""T-034 在线重排：开关、候选与查询集合、Agent 选中保护、降级/无 Agent、用量与 trace。

全部用 Fake Agent 脚本与确定性替身嵌入器，不发真实请求、不调真实模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import runtime_of
from researchflow import main
from researchflow.core.config import RunContext, load_settings
from researchflow.core.embedding import UsageTrackingEmbedder
from researchflow.core.llm import FakeLLMClient
from researchflow.core.retrieval_agent import RetrievalAgent
from researchflow.core.schemas import (
    AssistantTurn,
    LiteratureRef,
    LLMCallRecord,
    LLMUsage,
    RerankTrace,
    RetrievalTrace,
    StepRecord,
    ToolCall,
)
from researchflow.core.tracing import build_trace, load_pricing, trace_summary
from researchflow.main import _build_reranker
from researchflow.nodes.retrieval import retrieval_node
from researchflow.tools.hybrid import HybridReranker
from researchflow.tools.vector_store import VectorStore


def _ref(ref_id: str, title: str, score: float = 1.0) -> LiteratureRef:
    return LiteratureRef(
        ref_id=ref_id,
        title=title,
        summary=f"{title}的摘要",
        score=score,
        retriever="test",
    )


class QueryRetriever:
    """按查询返回预设文献的检索替身；没登记的查询返回空列表。"""

    def __init__(self, results: dict[str, list[LiteratureRef]]) -> None:
        self.results = results

    @property
    def name(self) -> str:
        return "test"

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        return list(self.results.get(query, []))[:k]


class _SpyEmbedder:
    """记录重排器是否真的调过嵌入；用量可选，默认 0。"""

    name = "spy"
    model = None
    dim = 2

    def __init__(self, prompt_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.last_usage: LLMUsage | None = None
        self.last_duration_ms: int | None = 0
        self.calls = 0

    def reset_usage(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.last_usage = LLMUsage(prompt_tokens=self.prompt_tokens, attempts=1)
        return [[1.0, 0.0] for _ in texts]


class SpyReranker:
    """按预设顺序返回候选的重排替身，并记录每次调用的输入。"""

    def __init__(self, order: list[str], prompt_tokens: int = 0) -> None:
        self.order = order
        self.embedder = _SpyEmbedder(prompt_tokens)
        self.calls: list[dict[str, Any]] = []

    def rerank(self, queries: list[str], refs: list[LiteratureRef], k: int) -> list[LiteratureRef]:
        self.calls.append(
            {
                "queries": list(queries),
                "ref_ids": [ref.ref_id for ref in refs],
                "k": k,
            }
        )
        by_id = {ref.ref_id: ref for ref in refs}
        ordered = [by_id[ref_id] for ref_id in self.order if ref_id in by_id]
        return ordered[:k]


def _call(call_id: str, name: str, **kwargs: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(kwargs, ensure_ascii=False))


def _turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(content=None, tool_calls=list(calls))


def _project_settings(project_root: Path, tmp_path: Path):
    settings = load_settings(project_root=project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings.retrieval.agent.enabled = True
    return settings, run_dir


def _agent_ctx(
    tmp_path: Path,
    project_root: Path,
    retriever: Any,
    script: list[AssistantTurn],
    reranker: Any = None,
    *,
    enabled: bool = True,
    top_k: int | None = None,
    max_refs: int = 6,
) -> RunContext:
    settings, run_dir = _project_settings(project_root, tmp_path)
    settings.retrieval.rerank.enabled = enabled
    settings.retrieval.rerank.top_k = top_k
    settings.llm.max_refs_in_prompt = max_refs
    llm = FakeLLMClient(script)
    ctx = RunContext(settings=settings, run_dir=run_dir, llm=llm, retriever=retriever)
    prompt = (project_root / "configs" / "prompts" / "retrieval_system.md").read_text(
        encoding="utf-8"
    )
    ctx.retrieval_agent = RetrievalAgent(llm=llm, ctx=ctx, system_prompt=prompt)
    ctx.reranker = reranker
    return ctx


def _five_refs_script() -> tuple[QueryRetriever, list[AssistantTurn]]:
    """两轮检索共 4 篇候选，Agent 只选其中 2 篇。"""
    retriever = QueryRetriever(
        {
            "课题原文": [_ref("test:1", "A", 4.0), _ref("test:2", "B", 3.0)],
            "english query": [_ref("test:3", "C", 2.0), _ref("test:4", "D", 1.0)],
        }
    )
    script = [
        _turn(_call("c1", "search_literature", query="课题原文", k=6)),
        _turn(_call("c2", "search_literature", query="english query", k=6)),
        _turn(_call("c3", "finish_retrieval", selected_ref_ids=["test:1", "test:2"], reason="够")),
    ]
    return retriever, script


# --- A1：默认关闭，① 的行为与 trace 都不变 -------------------------------------


def test_disabled_by_default_keeps_agent_result_and_no_rerank(tmp_path, project_root) -> None:
    retriever, script = _five_refs_script()
    spy = SpyReranker(["test:4", "test:3", "test:2", "test:1"])
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, spy, enabled=False)

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    assert [ref.ref_id for ref in update["literature_refs"]] == ["test:1", "test:2"]
    assert update["retrieval_trace"].rerank is None
    assert [call.response_model for call in update["llm_calls"]] == ["tool_loop"]
    assert spy.calls == []


def test_disabled_no_agent_path_has_no_trace(tmp_path, project_root) -> None:
    settings, run_dir = _project_settings(project_root, tmp_path)
    settings.retrieval.rerank.enabled = False
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=QueryRetriever({"课题原文": [_ref("test:1", "A")]}),
    )
    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))
    assert update["retrieval_trace"] is None


# --- A2：候选 = 全部检索结果，查询 = 原文 + Agent 改写，结果取 top_k ---------------


def test_rerank_uses_all_searched_candidates_and_topic_plus_agent_queries(
    tmp_path, project_root
) -> None:
    retriever, script = _five_refs_script()
    spy = SpyReranker(["test:2", "test:4", "test:1", "test:3"])
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, spy, top_k=2)

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    assert spy.calls == [
        {
            "queries": ["课题原文", "english query"],
            "ref_ids": ["test:1", "test:2", "test:3", "test:4"],
            "k": 2,
        }
    ]
    trace = update["retrieval_trace"].rerank
    assert trace is not None
    assert trace.enabled is True
    assert trace.candidates == 4
    assert trace.queries == ["课题原文", "english query"]
    assert trace.before_ids == ["test:1", "test:2", "test:3", "test:4"]
    assert trace.after_ids == ["test:2", "test:4"]
    # Agent 选中的 test:2 在重排结果里，不需要兜底替换
    assert trace.agent_pick_kept is True
    assert trace.replaced_ref_id is None
    assert [ref.ref_id for ref in update["literature_refs"]] == ["test:2", "test:4"]


def test_rerank_top_k_defaults_to_max_refs_in_prompt(tmp_path, project_root) -> None:
    retriever, script = _five_refs_script()
    spy = SpyReranker(["test:1", "test:2", "test:3", "test:4"])
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, spy, top_k=None, max_refs=3)

    retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    assert spy.calls[0]["k"] == 3


# --- A3：Agent 选中文献与重排结果无交集时，第一名替换进末位 ----------------------


def test_agent_top_pick_replaces_last_when_no_overlap(tmp_path, project_root) -> None:
    retriever, script = _five_refs_script()
    spy = SpyReranker(["test:3", "test:4"])
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, spy, top_k=2)

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    trace = update["retrieval_trace"].rerank
    assert trace is not None
    assert trace.after_ids == ["test:3", "test:1"]
    assert trace.replaced_ref_id == "test:1"
    assert trace.agent_pick_kept is True
    assert [ref.ref_id for ref in update["literature_refs"]] == ["test:3", "test:1"]


def test_no_replacement_when_any_overlap(tmp_path, project_root) -> None:
    retriever, script = _five_refs_script()
    spy = SpyReranker(["test:4", "test:2"])
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, spy, top_k=2)

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    trace = update["retrieval_trace"].rerank
    assert trace is not None
    assert trace.after_ids == ["test:4", "test:2"]
    assert trace.replaced_ref_id is None
    assert trace.agent_pick_kept is True


# --- A4：Agent 降级 / 没有 Agent 时不重排，trace 写明原因 ------------------------


def test_agent_degraded_skips_rerank_and_records_reason(tmp_path, project_root) -> None:
    retriever = QueryRetriever({"课题原文": [_ref("test:1", "A", 2.0)]})
    spy = SpyReranker(["test:1"])
    ctx = _agent_ctx(tmp_path, project_root, retriever, [], spy)  # 无脚本 → LLMError

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    trace = update["retrieval_trace"]
    assert trace.stop_reason == "agent_error"
    assert trace.rerank is not None
    assert trace.rerank.enabled is True
    assert trace.rerank.reason == "agent_degraded:agent_error"
    assert trace.rerank.after_ids == ["test:1"]
    assert spy.calls == []
    assert [ref.ref_id for ref in update["literature_refs"]] == ["test:1"]


def test_no_agent_skips_rerank_and_records_reason(tmp_path, project_root) -> None:
    settings, run_dir = _project_settings(project_root, tmp_path)
    settings.retrieval.rerank.enabled = True
    spy = SpyReranker(["test:1"])
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=QueryRetriever({"课题原文": [_ref("test:1", "A")]}),
        reranker=spy,
    )

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    trace = update["retrieval_trace"]
    assert trace.stop_reason == "no_agent"
    assert trace.rerank is not None
    assert trace.rerank.reason == "no_agent"
    assert spy.calls == []
    assert [ref.ref_id for ref in update["literature_refs"]] == ["test:1"]


# --- A5：嵌入用量进 llm_calls；tracing 有 rerank span 且按嵌入模型计价 ------------


def test_embedding_usage_recorded_in_llm_calls(tmp_path, project_root) -> None:
    retriever, script = _five_refs_script()
    embedder = UsageTrackingEmbedder(_SpyEmbedder(prompt_tokens=10))
    store = VectorStore(":memory:")
    reranker = HybridReranker(embedder, store)
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, reranker, top_k=2)

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))
    store.close()

    embedding_calls = [call for call in update["llm_calls"] if call.response_model == "embedding"]
    assert len(embedding_calls) == 1
    call = embedding_calls[0]
    assert call.step == "retrieval"
    # 候选块 1 次 + 2 条查询各 1 次，共 3 次 embed()，每次 10 token
    assert call.usage.prompt_tokens == 30
    assert call.usage.attempts == 3
    assert update["retrieval_trace"].rerank.prompt_tokens == 30


def test_trace_has_rerank_span_with_embedding_cost(project_root) -> None:
    step = StepRecord(
        step="retrieval",
        status="ok",
        attempt=1,
        started_at="2026-10-17T10:00:00",
        duration_ms=1000,
    )
    call = LLMCallRecord(
        step="retrieval",
        response_model="embedding",
        usage=LLMUsage(prompt_tokens=1_000_000, attempts=1),
        duration_ms=20,
    )
    retrieval_trace = RetrievalTrace(
        stop_reason="agent",
        rerank=RerankTrace(
            enabled=True,
            candidates=5,
            queries=["课题原文", "english query"],
            before_ids=["test:1", "test:2"],
            after_ids=["test:2", "test:1"],
            agent_pick_kept=True,
            duration_ms=42,
            prompt_tokens=1_000_000,
            embedding_model="text-embedding-v4",
        ),
    )
    state = {
        "run_id": "abc123def456",
        "step_records": [step],
        "llm_calls": [call],
        "retrieval_trace": retrieval_trace,
    }

    trace = build_trace(
        state,
        "abc123def456",
        model="fake",
        pricing=load_pricing(project_root / "configs" / "pricing.yaml"),
    )
    spans = trace["spans"]
    rerank_span = next(span for span in spans if span["name"] == "rerank")
    assert rerank_span["kind"] == "rerank"
    assert rerank_span["duration_ms"] == 42
    assert rerank_span["attributes"]["candidates"] == 5
    assert rerank_span["attributes"]["prompt_tokens"] == 1_000_000

    embedding_span = next(span for span in spans if span["name"] == "llm:embedding")
    assert embedding_span["attributes"]["model"] == "text-embedding-v4"
    assert embedding_span["attributes"]["cost"] == 0.5
    assert embedding_span["attributes"]["currency"] == "CNY"

    summary = trace_summary(trace)
    assert summary["total_cost"] == {"CNY": 0.5}
    assert summary["unpriced_calls"] == 0


# --- A6：旧 checkpoint（没有 rerank 字段）兼容 ----------------------------------


def test_old_checkpoint_without_rerank_field_is_compatible(project_root) -> None:
    trace = RetrievalTrace.model_validate({"queries": [], "stop_reason": "agent"})
    assert trace.rerank is None

    step = StepRecord(
        step="retrieval",
        status="ok",
        attempt=1,
        started_at="2026-10-17T10:00:00",
        duration_ms=100,
    )
    built = build_trace(
        {"step_records": [step], "retrieval_trace": trace}, "abc123def456", model="fake"
    )
    assert all(span["name"] != "rerank" for span in built["spans"])


# --- 配置与构造：rerank 开关默认关闭、top_k 校验、main 构造 ----------------------


def test_rerank_settings_defaults_and_validation(tmp_path) -> None:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "settings.yaml").write_text("llm:\n  mode: auto\n", encoding="utf-8")
    settings = load_settings(project_root=tmp_path, load_env_file=False)
    assert settings.retrieval.rerank.enabled is False
    assert settings.retrieval.rerank.top_k is None

    (tmp_path / "configs" / "settings.yaml").write_text(
        "retrieval:\n  rerank:\n    enabled: true\n    top_k: 3\n", encoding="utf-8"
    )
    settings = load_settings(project_root=tmp_path, load_env_file=False)
    assert settings.retrieval.rerank.enabled is True
    assert settings.retrieval.rerank.top_k == 3

    (tmp_path / "configs" / "settings.yaml").write_text(
        "retrieval:\n  rerank:\n    top_k: 0\n", encoding="utf-8"
    )
    from researchflow.core.config import ConfigError

    with pytest.raises(ConfigError, match="top_k"):
        load_settings(project_root=tmp_path, load_env_file=False)


def test_build_reranker_respects_switch(tmp_path, project_root) -> None:
    settings, run_dir = _project_settings(project_root, tmp_path)
    ctx = RunContext(settings=settings, run_dir=run_dir, llm=FakeLLMClient())
    assert _build_reranker(ctx) is None

    settings.retrieval.rerank.enabled = True
    reranker = _build_reranker(ctx)
    assert isinstance(reranker, HybridReranker)
    assert reranker.embedder.name == "hash-256"


def test_enabled_without_reranker_raises_config_error(tmp_path, project_root) -> None:
    retriever, script = _five_refs_script()
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, reranker=None)

    # @step 会把 ResearchFlowError 收敛成 error state，而不是让它冒出去
    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    assert update["step_status"] == {"retrieval": "error"}
    assert update["error"].code == "RF-001"
    assert "重排器" in update["error"].message


def test_manifest_contains_rerank_trace(tmp_path, project_root) -> None:
    retriever, script = _five_refs_script()
    spy = SpyReranker(["test:2", "test:4", "test:1", "test:3"])
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, spy, top_k=2)
    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    manifest = main._write_manifest(
        ctx.run_dir,
        {"run_id": "r", "topic": "课题原文", "retrieval_trace": update["retrieval_trace"]},
        "fake",
        project_root,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rerank = payload["retrieval_trace"]["rerank"]
    assert rerank["candidates"] == 4
    assert rerank["after_ids"] == ["test:2", "test:4"]
    assert rerank["enabled"] is True

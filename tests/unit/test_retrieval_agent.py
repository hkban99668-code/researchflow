"""T-023：检索 Agent 机制（A2–A8）与默认关闭（A1 的一部分）。

A1（enabled=false 行为不变）由现有全套测试保证，这里只补两条：默认关闭与 Fake
模式下 `_build_retrieval_agent` 返回 None。其余用例都用 Fake 脚本给出 Agent 的
决策，不发真实网络请求、不调真实模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conftest import runtime_of
from researchflow import main
from researchflow.core.config import RunContext, load_settings
from researchflow.core.errors import RetrievalError
from researchflow.core.llm import FakeLLMClient
from researchflow.core.retrieval_agent import RetrievalAgent
from researchflow.core.schemas import AssistantTurn, LiteratureRef, ToolCall
from researchflow.core.tools import ToolContext
from researchflow.main import _build_retrieval_agent
from researchflow.nodes.retrieval import retrieval_node
from researchflow.tools.retrieval_tools import build_retrieval_registry


def _ref(ref_id: str, title: str, score: float = 1.0) -> LiteratureRef:
    return LiteratureRef(
        ref_id=ref_id,
        title=title,
        summary=f"{title}的摘要",
        score=score,
        retriever="test",
    )


class QueryRetriever:
    """按查询返回预设文献的检索替身；没登记的查询返回空列表，并记录调用。"""

    def __init__(self, results: dict[str, list[LiteratureRef]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "test"

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        self.calls.append((query, k))
        return list(self.results.get(query, []))[:k]


class FailingRetriever:
    """对 `error_query` 抛 RetrievalError，其余查询返回预设文献的检索替身。"""

    def __init__(self, error_query: str, results: dict[str, list[LiteratureRef]]) -> None:
        self.error_query = error_query
        self.results = results

    @property
    def name(self) -> str:
        return "test"

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        if query == self.error_query:
            raise RetrievalError("OpenAlex 请求超时（10s）")
        return list(self.results.get(query, []))[:k]


def _call(call_id: str, name: str, **kwargs: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(kwargs, ensure_ascii=False))


def _turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(content=None, tool_calls=list(calls))


def _final(text: str) -> AssistantTurn:
    return AssistantTurn(content=text)


def _agent_ctx(
    tmp_path: Path,
    project_root: Path,
    retriever: Any,
    script: list[AssistantTurn],
    *,
    max_steps: int = 5,
    max_retrievals: int = 4,
    max_refs: int = 6,
) -> RunContext:
    """构造带检索 Agent 的 RunContext（agent 显式构造，绕过 main 的 Fake 拦截）。"""
    settings = load_settings(project_root=project_root)
    settings.retrieval.agent.enabled = True
    settings.retrieval.agent.max_steps = max_steps
    settings.retrieval.agent.max_retrievals = max_retrievals
    settings.llm.max_refs_in_prompt = max_refs
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = FakeLLMClient(script)
    ctx = RunContext(settings=settings, run_dir=run_dir, llm=llm, retriever=retriever)
    prompt = (project_root / "configs" / "prompts" / "retrieval_system.md").read_text(
        encoding="utf-8"
    )
    ctx.retrieval_agent = RetrievalAgent(llm=llm, ctx=ctx, system_prompt=prompt)
    return ctx


# --- A1 的一部分：默认关闭 / Fake 模式不构造 Agent ------------------------------


def test_build_retrieval_agent_none_when_disabled(tmp_path, project_root) -> None:
    settings = load_settings(project_root=project_root)
    assert settings.retrieval.agent.enabled is False
    ctx = RunContext(settings=settings, run_dir=tmp_path, llm=FakeLLMClient())
    assert _build_retrieval_agent(ctx) is None


def test_build_retrieval_agent_none_in_fake_mode(tmp_path, project_root) -> None:
    settings = load_settings(project_root=project_root)
    settings.retrieval.agent.enabled = True
    ctx = RunContext(settings=settings, run_dir=tmp_path, llm=FakeLLMClient())
    assert _build_retrieval_agent(ctx) is None


# --- A1（T-028）：选择上限由配置单一来源，模型第一次调用 finish 前就能看到 --------


def test_system_prompt_notice_follows_configured_limit(tmp_path, project_root) -> None:
    """系统提示里的"最多选 N 条"随 `llm.max_refs_in_prompt` 变化。

    Agent 构造时把 N 追加进提示：换成 N=3 后旧值 6 必须消失，说明提示不是从
    提示词文件里读到的固定数字（写死在文件里的话这里会同时出现 6 和 3）。
    """
    retriever = QueryRetriever({})
    notices: dict[int, str] = {}
    for max_refs in (6, 3):
        ctx = _agent_ctx(tmp_path, project_root, retriever, [_final("不选")], max_refs=max_refs)
        ctx.retrieval_agent.decide("课题")
        system_message = ctx.llm.chat_requests[0][0]
        assert system_message["role"] == "system"
        notices[max_refs] = system_message["content"]

    assert "最多选 6 条" in notices[6]
    assert "最多选 3 条" in notices[3]
    assert "最多选 6 条" not in notices[3]


def test_over_limit_selection_rejected_though_prompt_states_limit(tmp_path, project_root) -> None:
    """复现场景（T-028 F5）：模型第一次选 10 条被上限拒绝（上限 6），
    失败的是"选多了"，而不是"提示里没说"——提示在第一次 finish 前就写了上限。"""
    retriever = QueryRetriever({"q": [_ref(f"test:{i}", f"T{i}") for i in range(1, 11)]})
    script = [
        _turn(_call("c1", "search_literature", query="q", k=10)),
        _turn(
            _call(
                "c2",
                "finish_retrieval",
                selected_ref_ids=[f"test:{i}" for i in range(1, 11)],
                reason="全都要",
            )
        ),
        _turn(
            _call(
                "c3",
                "finish_retrieval",
                selected_ref_ids=[f"test:{i}" for i in range(1, 7)],
                reason="改对了",
            )
        ),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)

    decision = ctx.retrieval_agent.decide("课题")

    first_finish = ctx.retrieval_agent.last_invocations[1]
    assert not first_finish.result.ok
    assert first_finish.result.error is not None
    assert "最多选择 6 条" in first_finish.result.error.message
    # 模型在第一次调用 finish_retrieval 之前就已经能看到上限
    assert "最多选 6 条" in ctx.llm.chat_requests[0][0]["content"]
    assert decision.stop_reason == "agent"
    assert len(decision.refs) == 6


def test_finish_retrieval_schema_and_enforcement_share_limit(tmp_path, project_root) -> None:
    """`selected_ref_ids` 的字段说明写 N，且工具按同一个 N 拒绝超量选择。"""
    settings = load_settings(project_root=project_root)
    run_ctx = RunContext(settings=settings, run_dir=tmp_path)
    run_ctx.retriever = QueryRetriever({"q": [_ref(f"test:{i}", f"T{i}") for i in range(1, 5)]})
    searched: dict[str, LiteratureRef] = {}
    queries: list[str] = []
    registry = build_retrieval_registry(searched, queries, run_ctx, max_refs=3, max_retrievals=4)
    tool_ctx = ToolContext(workspace_root=settings.project_root, settings=settings)

    finish = next(
        tool
        for tool in registry.as_openai_tools()
        if tool["function"]["name"] == "finish_retrieval"
    )
    description = finish["function"]["parameters"]["properties"]["selected_ref_ids"]["description"]
    assert "最多 3 条" in description

    registry.invoke("search_literature", {"query": "q", "k": 4}, tool_ctx)
    allowed = registry.invoke(
        "finish_retrieval",
        {"selected_ref_ids": ["test:1", "test:2", "test:3"], "reason": ""},
        tool_ctx,
    )
    too_many = registry.invoke(
        "finish_retrieval",
        {"selected_ref_ids": ["test:1", "test:2", "test:3", "test:4"], "reason": ""},
        tool_ctx,
    )

    assert allowed.ok
    assert not too_many.ok
    assert too_many.error is not None
    assert too_many.error.code == "invalid_args"
    assert "最多选择 3 条" in too_many.error.message


# --- A2：改写查询、多轮检索、选中文献进入 literature_refs -----------------------


def test_agent_rewrites_query_and_selects(tmp_path, project_root) -> None:
    retriever = QueryRetriever(
        {
            "课题原文": [],
            "english query": [
                _ref("test:1", "A", 3.0),
                _ref("test:2", "B", 2.0),
                _ref("test:3", "C", 1.0),
            ],
        }
    )
    script = [
        _turn(_call("c1", "search_literature", query="课题原文", k=6)),
        _turn(_call("c2", "search_literature", query="english query", k=6)),
        _turn(
            _call(
                "c3",
                "finish_retrieval",
                selected_ref_ids=["test:1", "test:2"],
                reason="够用",
            )
        ),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    assert [ref.ref_id for ref in update["literature_refs"]] == ["test:1", "test:2"]
    assert update["step_status"] == {"retrieval": "ok"}
    trace = update["retrieval_trace"]
    assert trace.queries == ["课题原文", "english query"]
    assert trace.stop_reason == "agent"
    assert trace.selected_ref_ids == ["test:1", "test:2"]


# --- A3：选了没检索到的 id → invalid_args 列出可选 id，改对后成功 -----------------


def test_finish_retrieval_rejects_unretrieved_id_and_lists_available(
    tmp_path, project_root
) -> None:
    settings = load_settings(project_root=project_root)
    run_ctx = RunContext(settings=settings, run_dir=tmp_path)
    run_ctx.retriever = QueryRetriever({"q": [_ref("test:1", "A")]})
    searched: dict[str, LiteratureRef] = {}
    queries: list[str] = []
    registry = build_retrieval_registry(searched, queries, run_ctx, max_refs=6, max_retrievals=4)
    tool_ctx = ToolContext(workspace_root=settings.project_root, settings=settings)

    registry.invoke("search_literature", {"query": "q", "k": 6}, tool_ctx)
    result = registry.invoke(
        "finish_retrieval", {"selected_ref_ids": ["test:999"], "reason": ""}, tool_ctx
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert "test:999" in result.error.message
    assert "test:1" in result.error.message


def test_agent_retries_finish_after_invalid_id(tmp_path, project_root) -> None:
    retriever = QueryRetriever({"q": [_ref("test:1", "A", 2.0), _ref("test:2", "B", 1.0)]})
    script = [
        _turn(_call("c1", "search_literature", query="q", k=6)),
        _turn(_call("c2", "finish_retrieval", selected_ref_ids=["test:999"], reason="错")),
        _turn(_call("c3", "finish_retrieval", selected_ref_ids=["test:1"], reason="对")),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)

    decision = ctx.retrieval_agent.decide("课题")

    assert decision.stop_reason == "agent"
    assert [ref.ref_id for ref in decision.refs] == ["test:1"]
    # 第一次 finish 是 invalid_args，第二次才成功
    assert len(ctx.retrieval_agent.last_invocations) == 3
    assert not ctx.retrieval_agent.last_invocations[1].result.ok


# --- A4：预算耗尽，回退为已检索结果里排在最前的若干条 ---------------------------


def test_budget_exhausted_falls_back_to_top_retrieved(tmp_path, project_root) -> None:
    retriever = QueryRetriever(
        {
            "q1": [_ref("test:1", "A", 1.0)],
            "q2": [_ref("test:2", "B", 2.0)],
            "q3": [_ref("test:3", "C", 3.0)],
            "q4": [_ref("test:4", "D", 4.0)],
            "q5": [_ref("test:5", "E", 5.0)],
        }
    )
    script = [
        _turn(_call("c1", "search_literature", query="q1", k=6)),
        _turn(_call("c2", "search_literature", query="q2", k=6)),
        _turn(_call("c3", "search_literature", query="q3", k=6)),
        _turn(_call("c4", "search_literature", query="q4", k=6)),
        _turn(_call("c5", "search_literature", query="q5", k=6)),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, max_retrievals=4)

    update = retrieval_node({"run_id": "r", "topic": "课题"}, runtime=runtime_of(ctx))

    trace = update["retrieval_trace"]
    assert trace.stop_reason == "budget_exhausted"
    assert trace.queries == ["q1", "q2", "q3", "q4"], "第 5 次检索被预算拒绝，未记录"
    # 已检索集合按 score 降序取最前的 max_refs_in_prompt 条
    assert [ref.ref_id for ref in update["literature_refs"]] == [
        "test:4",
        "test:3",
        "test:2",
        "test:1",
    ]


# --- A5：Agent 抛 LLMError，回退为原文检索，运行不崩 -----------------------------


def test_agent_error_falls_back_to_original_search(tmp_path, project_root) -> None:
    retriever = QueryRetriever({"课题原文": [_ref("test:1", "A", 1.0), _ref("test:2", "B", 2.0)]})
    ctx = _agent_ctx(tmp_path, project_root, retriever, [])  # 无脚本，首次 chat 即 LLMError

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    assert update["step_status"] == {"retrieval": "ok"}
    assert [ref.ref_id for ref in update["literature_refs"]] == ["test:1", "test:2"]
    assert update["retrieval_trace"].stop_reason == "agent_error"


# --- A6：Agent 一条都没选到（全部检索为空），以 empty 结束 -----------------------


def test_agent_selects_nothing_terminates_empty(tmp_path, project_root) -> None:
    retriever = QueryRetriever({})
    script = [
        _turn(_call("c1", "search_literature", query="课题原文", k=6)),
        _turn(_call("c2", "search_literature", query="english", k=6)),
        _final("没有找到相关文献"),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)

    update = retrieval_node({"run_id": "r", "topic": "课题原文"}, runtime=runtime_of(ctx))

    assert update["termination"].kind == "empty"
    assert update["termination"].step == "retrieval"
    assert update.get("literature_refs") is None


# --- A7：检索器抛 RetrievalError，回填可读错误，回路不崩 -------------------------


def test_retriever_error_fed_back_and_loop_recovers(tmp_path, project_root) -> None:
    retriever = FailingRetriever("boom", {"ok": [_ref("test:1", "A", 1.0)]})
    script = [
        _turn(_call("c1", "search_literature", query="boom", k=6)),
        _turn(_call("c2", "search_literature", query="ok", k=6)),
        _turn(_call("c3", "finish_retrieval", selected_ref_ids=["test:1"], reason="")),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)

    decision = ctx.retrieval_agent.decide("课题")

    assert decision.stop_reason == "agent"
    assert [ref.ref_id for ref in decision.refs] == ["test:1"]
    # 第一次检索失败被回填为 execution_error，detail 带 RF-007，回路没有崩
    first = ctx.retrieval_agent.last_invocations[0]
    assert not first.result.ok
    assert first.result.error is not None
    assert first.result.error.code == "execution_error"
    assert first.result.error.detail.get("code") == "RF-007"
    assert "超时" in first.result.error.message


# --- A8：retrieval_trace 进 manifest，且不含 Key 与业务空间 ID -------------------


def test_retrieval_trace_in_manifest_has_no_secrets(tmp_path, project_root) -> None:
    retriever = QueryRetriever({"q": [_ref("test:1", "A", 1.0)]})
    script = [
        _turn(_call("c1", "search_literature", query="q", k=6)),
        _turn(_call("c2", "finish_retrieval", selected_ref_ids=["test:1"], reason="")),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)
    update = retrieval_node({"run_id": "r", "topic": "课题"}, runtime=runtime_of(ctx))
    trace = update["retrieval_trace"]

    # trace 的字段集合不含任何 Key / base_url / 业务空间 ID 字段
    # （tool_calls / llm_turns 是 T-025 新增，failure_reason 是 T-027 新增，
    # suspicious_ref_ids 是 T-033 新增：只含 ref_id，不含摘要文本；
    # rerank 是 T-034 新增：只含 ref_id 与查询词）
    assert set(trace.model_dump()) == {
        "queries",
        "stop_reason",
        "selected_ref_ids",
        "usage",
        "tool_calls",
        "llm_turns",
        "failure_reason",
        "suspicious_ref_ids",
        "rerank",
    }
    assert set(trace.usage.model_dump()) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "attempts",
    }

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = main._write_manifest(
        run_dir, {"run_id": "r", "topic": "课题", "retrieval_trace": trace}, "real", project_root
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "retrieval_trace" in payload
    dumped = json.dumps(payload["retrieval_trace"], ensure_ascii=False)
    for forbidden in ("api_key", "base_url", "authorization", "Bearer"):
        assert forbidden.lower() not in dumped.lower()


# --- T-027：三种失败原因的行为与脱敏记录 -----------------------------------------


def test_llm_error_failure_reason_recorded_sanitized(tmp_path, project_root) -> None:
    """(a) 模型调用抛 LLMError：降级为原文检索，失败原因（类型+一句消息）进追溯。"""
    retriever = QueryRetriever({"课题原文": [_ref("test:1", "A", 1.0)]})
    ctx = _agent_ctx(tmp_path, project_root, retriever, [])  # 无脚本，首次 chat 即 LLMError
    agent = ctx.retrieval_agent

    decision = agent.decide("课题原文")

    assert decision.stop_reason == "agent_error"
    assert [ref.ref_id for ref in decision.refs] == ["test:1"]
    assert agent.last_failure_reason is not None
    assert agent.last_failure_reason.startswith("LLMError:")
    assert "Fake 脚本已耗尽" in agent.last_failure_reason
    # 不含 Key 与绝对路径
    lowered = agent.last_failure_reason.lower()
    assert "sk-" not in lowered
    assert not any(token.startswith("/") for token in agent.last_failure_reason.split())


def test_unparseable_tool_args_do_not_crash_loop(tmp_path, project_root) -> None:
    """(b) 模型连续输出无法解析的参数：错误回填为 invalid_args，回路不崩，
    预算耗尽后按 budget_exhausted 降级为已检索结果的最前若干条。"""
    retriever = QueryRetriever({"q": [_ref("test:1", "A", 1.0)]})
    bad = ToolCall(id="c1", name="search_literature", arguments="{not json")
    script = [_turn(bad), _turn(bad)]  # max_steps=2，两轮都用坏参数
    ctx = _agent_ctx(tmp_path, project_root, retriever, script, max_steps=2)
    agent = ctx.retrieval_agent

    decision = agent.decide("课题")

    assert decision.stop_reason == "budget_exhausted"
    # 两次坏参数都以 invalid_args 记进追溯，回路没有中断
    assert [trace.error_code for trace in agent.last_tool_calls] == ["invalid_args", "invalid_args"]


def test_finish_with_out_of_pool_ids_is_invalid_selection(tmp_path, project_root) -> None:
    """(c) finish_retrieval 选了池外 ref_id：工具报错并说明怎么改，模型可重试；
    模型不再重试而空手而归时 stop_reason=invalid_selection，不再伪装成 agent。"""
    retriever = QueryRetriever({"q": [_ref("test:1", "A", 1.0)]})
    script = [
        _turn(_call("c1", "search_literature", query="q", k=6)),
        _turn(_call("c2", "finish_retrieval", selected_ref_ids=["pool:99"], reason="")),
        _final("好吧"),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)
    agent = ctx.retrieval_agent

    decision = agent.decide("课题")

    # 工具报错且信息告诉模型怎么改（选 search_literature 返回过的 ref_id）
    finish = next(
        invocation
        for invocation in agent.last_invocations
        if invocation.call.name == "finish_retrieval"
    )
    assert not finish.result.ok
    assert finish.result.error is not None
    assert finish.result.error.code == "invalid_args"
    assert "只能选 search_literature 返回过的 ref_id" in finish.result.error.message
    assert "test:1" in finish.result.error.message  # 告诉模型可选哪些 id
    # 最终空手而归：显式 invalid_selection，且失败原因进追溯
    assert decision.stop_reason == "invalid_selection"
    assert decision.refs == []
    assert agent.last_failure_reason is not None
    assert "selected_ref_ids" in agent.last_failure_reason


def test_final_without_finish_attempt_is_still_agent(tmp_path, project_root) -> None:
    """对照组：模型从未尝试 finish 就结束，仍是 agent（模型自己的"不选"判断）。"""
    retriever = QueryRetriever({"q": [_ref("test:1", "A", 1.0)]})
    script = [
        _turn(_call("c1", "search_literature", query="q", k=6)),
        _final("不需要更多文献了"),
    ]
    ctx = _agent_ctx(tmp_path, project_root, retriever, script)

    decision = ctx.retrieval_agent.decide("课题")

    assert decision.stop_reason == "agent"
    assert ctx.retrieval_agent.last_failure_reason is None

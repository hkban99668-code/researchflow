"""T-010 A5–A7：回溯重跑时三类 reducer 的语义。

用 `Command(goto=<node>)` 让一个已经跑完的图从指定节点重跑（调查问题 1 的结论：
`graph.update_state(config, {}, as_node=...)` 之后 `next == ()`，不会触发重跑；
`Command(goto=...)` 才会真正重跑目标节点及其下游）。

- A5：`goto="compute"` 重跑 ④，`artifacts["compute_result"].attempt == 2`，只有一条。
- A6：`goto="retrieval"` 重跑 ①（换一个条数不同的检索器），`literature_refs` 被整体替换。
- A7：`goto="data_review"` 重审一次，`review_log` 只增不减，新记录 `attempt == 2`。
"""

from __future__ import annotations

import json

from langgraph.types import Command

from conftest import StubRetriever, thread_config
from researchflow import main
from researchflow.core.config import RunContext, load_settings
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import LiteratureRef, ParamSet
from researchflow.graphs.orchestrator import build_graph
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"


def _ctx(tmp_path, project_root, retriever=None) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "rerun-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=build_retriever(settings) if retriever is None else retriever,
    )


def _state() -> dict:
    return {
        "run_id": "rerun-run",
        "topic": TOPIC,
        "param_set": ParamSet(
            algorithm="linear_regression",
            params={
                "dataset_path": "examples/sample_dataset.csv",
                "target": "y",
                "features": ["x1", "x2", "x3"],
                "test_size": 0.25,
            },
            seed=42,
        ),
    }


def _run_through(graph, config, ctx) -> dict:
    """跑完整条链（审核放行），返回最终 state。"""
    graph.invoke(_state(), config, context=ctx)
    return graph.invoke(
        Command(resume={"decision": "approve", "reason": "t010"}), config, context=ctx
    )


def _many_refs(count: int) -> list[LiteratureRef]:
    return [
        LiteratureRef(ref_id=f"stub:ref-{index}", title=f"替身文献{index}", retriever="stub")
        for index in range(1, count + 1)
    ]


def test_rerun_compute_overwrites_artifact(tmp_path, project_root):
    """A5：重跑 ④ 后 compute_result 只有一条且 attempt == 2，manifest 同理。"""
    ctx = _ctx(tmp_path, project_root)
    graph = build_graph()
    config = thread_config("t010-a5")

    _run_through(graph, config, ctx)
    rerun = graph.invoke(Command(goto="compute"), config, context=ctx)

    # state 里按 kind 覆盖：compute_result 只有一条，attempt 递增到 2
    assert rerun["artifacts"]["compute_result"].attempt == 2
    assert sorted(rerun["artifacts"]) == sorted(
        ["cleaned_data", "compute_result", "chart", "draft", "validation_report"]
    )
    assert [ref.kind for ref in rerun["artifacts"].values()].count("compute_result") == 1
    # 其他产物也各自只有一条，只是 attempt 跟着重跑递增
    assert rerun["artifacts"]["cleaned_data"].attempt == 1

    # manifest 仍输出为列表，且 compute_result 只有一条、attempt == 2
    manifest = json.loads(
        main._write_manifest(ctx.run_dir, rerun, "fake").read_text(encoding="utf-8")
    )
    kinds = [ref["kind"] for ref in manifest["artifacts"]]
    assert kinds.count("compute_result") == 1
    compute_ref = manifest["artifacts"][kinds.index("compute_result")]
    assert compute_ref["attempt"] == 2


def test_rerun_retrieval_replaces_literature_refs(tmp_path, project_root):
    """A6：重跑 ①（换条数不同的检索器），literature_refs 被整体替换而非追加。"""
    ctx = _ctx(tmp_path, project_root, retriever=StubRetriever())
    graph = build_graph()
    config = thread_config("t010-a6")

    first = _run_through(graph, config, ctx)
    assert len(first["literature_refs"]) == 2

    # 换成 5 条：替换语义下应等于 5，追加语义会变成 2+5=7
    ctx.retriever = StubRetriever(refs=_many_refs(5))
    pending = graph.invoke(Command(goto="retrieval"), config, context=ctx)

    assert "__interrupt__" in pending, "重跑 ① 后应停在 ③ 审核处"
    assert len(pending["literature_refs"]) == 5


def test_rerun_data_review_appends_review_log(tmp_path, project_root):
    """A7：重审一次，review_log 只增不减，新记录 attempt == 2。"""
    ctx = _ctx(tmp_path, project_root)
    graph = build_graph()
    config = thread_config("t010-a7")

    first = _run_through(graph, config, ctx)
    assert len(first["review_log"]) == 1
    assert first["review_log"][0].attempt == 1

    pending = graph.invoke(Command(goto="data_review"), config, context=ctx)
    assert "__interrupt__" in pending, "重跑 data_review 必须再次触发 interrupt"

    result = graph.invoke(
        Command(resume={"decision": "approve", "reason": "t010-again"}), config, context=ctx
    )
    assert len(result["review_log"]) == 2
    assert [record.attempt for record in result["review_log"]] == [1, 2]
    assert result["review_log"][-1].decision == "approve"

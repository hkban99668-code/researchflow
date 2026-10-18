"""T-P0-2：终止语义必须在每个节点后生效。

第 2 轮达标标准：
1. 除 `validation` 外，其余节点的出边都是条件边；
2. ① 检索为空时直接 END（termination.kind == "empty"），不再进入 ③ 触发人工审核；
3. CLI 退出码不得是"审核拒绝"（2）。
"""

from __future__ import annotations

from conftest import thread_config
from researchflow import main
from researchflow.core.config import RunContext, load_settings
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import ParamSet
from researchflow.graphs.orchestrator import build_graph
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"


def _context(tmp_path, project_root, retriever=None) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "routing-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=build_retriever(settings) if retriever is None else retriever,
    )


def _initial_state() -> dict:
    return {
        "run_id": "routing-run",
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


def test_termination_routes_after_every_node(tmp_path, project_root, empty_retriever):
    graph = build_graph()
    structure = graph.get_graph()

    conditional_sources = {edge.source for edge in structure.edges if edge.conditional}
    # `validation` 与 `attribute` 的出边不是条件边：前者是固定的通过/不通过路由，
    # 后者的去向由节点返回的 `Command(goto=...)` 决定（回溯到 data_prepare/compute 或 END）。
    for node_id in set(structure.nodes) - {"__start__", "__end__", "validation", "attribute"}:
        assert node_id in conditional_sources, f"{node_id} 的出边不是条件边"

    ctx = _context(tmp_path, project_root, retriever=empty_retriever)
    config = thread_config("routing-empty")
    result = graph.invoke(_initial_state(), config, context=ctx)

    assert "__interrupt__" not in result, "空检索不应继续触发人工审核"
    termination = result["termination"]
    assert termination is not None
    assert termination.kind == "empty"
    assert termination.step == "retrieval"
    assert result.get("cleaned_data") is None


def test_empty_retrieval_exit_code_is_not_review_rejected(
    tmp_path, project_root, monkeypatch, empty_retriever
):
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *args, **kwargs: settings)
    # 换掉工厂函数，让 CLI 构造的 RunContext 拿到"什么都没命中"的检索器
    monkeypatch.setattr(
        "researchflow.main.build_retriever", lambda *args, **kwargs: empty_retriever
    )

    exit_code = main.main(
        ["run", "--topic", str(project_root / "examples" / "demo_topic.yaml"), "--auto-approve"]
    )

    assert exit_code != main.EXIT_REVIEW_REJECTED
    assert exit_code == main.EXIT_VALIDATION_FAILED

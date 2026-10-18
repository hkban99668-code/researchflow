"""T-007 A1–A3、A5、A7、A8：审核暂停期间文件被改，恢复后必须被 checksum 拦下。

约定：
1. 改文件都在 tmp 里做。原始数据复制到 `tmp_path` 再交给链路，绝不改仓库里的
   `examples/sample_dataset.csv`。
2. 改"一个字节"的方式是把最后一个数字字符换成另一个数字：CSV 仍然合法，
   失败原因只能是 checksum，不会是解析错误。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.types import Command

from conftest import thread_config
from researchflow.core.config import RunContext, load_settings, sha256_file
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import ParamSet
from researchflow.graphs.orchestrator import build_graph
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"
CANDIDATE = "cleaned_candidate.csv"
# A1：拆分后 ③ 占两个节点，加上 attribute 后业务节点共 9 个
BUSINESS_NODES = {
    "retrieval",
    "extraction",
    "data_prepare",
    "data_review",
    "compute",
    "chart",
    "writing",
    "validation",
    "attribute",
}


def _flip_last_digit(path: Path) -> None:
    """把最后一个数字字符换成另一个数字（保证 CSV 仍然合法）。"""
    data = path.read_bytes()
    for index in range(len(data) - 1, -1, -1):
        if data[index : index + 1].isdigit():
            replacement = b"2" if data[index : index + 1] == b"1" else b"1"
            path.write_bytes(data[:index] + replacement + data[index + 1 :])
            return
    raise AssertionError(f"文件里没有数字可改: {path}")


@pytest.fixture
def dataset_path(tmp_path, project_root) -> Path:
    """示例数据集的 tmp 副本：A2/A3 要改文件，不能动仓库里的原件。"""
    target = tmp_path / "sample_dataset.csv"
    target.write_bytes((project_root / "examples" / "sample_dataset.csv").read_bytes())
    return target


@pytest.fixture
def ctx(tmp_path, project_root, dataset_path) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "e2e-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=settings, run_dir=run_dir, llm=FakeLLMClient(), retriever=build_retriever(settings)
    )


def _state(dataset_path: Path) -> dict:
    return {
        "run_id": "e2e-run",
        "topic": TOPIC,
        "param_set": ParamSet(
            algorithm="linear_regression",
            params={
                "dataset_path": str(dataset_path),
                "target": "y",
                "features": ["x1", "x2", "x3"],
                "test_size": 0.25,
            },
            seed=42,
        ),
    }


def _approve(graph, config, ctx):
    return graph.invoke(
        Command(resume={"decision": "approve", "reason": "T-007"}), config, context=ctx
    )


def test_graph_splits_data_pipeline_into_prepare_and_review():
    """A1：图里是 `data_prepare` + `data_review`，加 `attribute` 后业务节点共 9 个。"""
    nodes = set(build_graph().get_graph().nodes)

    assert {"data_prepare", "data_review", "attribute"} <= nodes
    assert "data_pipeline" not in nodes
    assert nodes - {"__start__", "__end__"} == BUSINESS_NODES
    assert len(nodes - {"__start__", "__end__"}) == 9


def test_raw_csv_changed_during_pause_ends_with_rf003(ctx, dataset_path):
    """A2：暂停期间改原始 CSV，恢复后以 RF-003 终止，review_log 为空。"""
    graph = build_graph()
    config = thread_config("t007-raw")
    before = sha256_file(dataset_path)

    pending = graph.invoke(_state(dataset_path), config, context=ctx)
    assert "__interrupt__" in pending
    assert pending["__interrupt__"][0].value["raw_checksum"] == before

    _flip_last_digit(dataset_path)
    result = _approve(graph, config, ctx)

    termination = result["termination"]
    assert termination.kind == "escalated"
    assert termination.step == "attribute"
    assert result["error"] is not None and result["error"].code == "RF-003"
    assert result["review_log"] == [], "还没生成审核记录就失败了"

    detail = result["error"].detail
    assert detail["file"] == "raw"
    assert detail["expected"] == before
    assert detail["actual"] == sha256_file(dataset_path)
    assert result.get("cleaned_data") is None


def test_candidate_changed_during_pause_ends_with_rf003(ctx, dataset_path):
    """A3：暂停期间改候选文件，错误信息指明是候选文件。"""
    graph = build_graph()
    config = thread_config("t007-candidate")

    pending = graph.invoke(_state(dataset_path), config, context=ctx)
    assert "__interrupt__" in pending
    subject_before = pending["__interrupt__"][0].value["subject_checksum"]

    candidate = ctx.artifact_path(CANDIDATE)
    assert sha256_file(candidate) == subject_before
    _flip_last_digit(candidate)

    result = _approve(graph, config, ctx)

    assert result["error"].code == "RF-003"
    assert result["error"].detail["file"] == "cleaned_candidate"
    assert result["error"].detail["expected"] == subject_before
    assert result["review_log"] == []


def test_approved_review_binds_checksums_and_artifact(ctx, dataset_path):
    """A5：审核记录带上被审对象的 checksum，产物指向候选文件。"""
    graph = build_graph()
    config = thread_config("t007-approve")

    graph.invoke(_state(dataset_path), config, context=ctx)
    result = _approve(graph, config, ctx)

    cleaned = result["cleaned_data"]
    record = result["review_log"][0]
    assert record.decision == "approve"
    assert record.subject_checksum == cleaned.checksum
    assert record.raw_checksum == sha256_file(dataset_path)
    # 记录里的 checksum 确实能在磁盘上对上号
    assert record.subject_checksum == sha256_file(ctx.artifact_path(cleaned.path))

    artifact = result["artifacts"]["cleaned_data"]
    assert artifact.path == CANDIDATE
    assert artifact.checksum == cleaned.checksum
    assert artifact.produced_by == "data_review"


def test_rejected_run_keeps_candidate_without_registering_it(ctx, dataset_path):
    """A7：打回时候选文件留在 run 目录，但不登记成 cleaned_data 产物。"""
    graph = build_graph()
    config = thread_config("t007-reject")

    graph.invoke(_state(dataset_path), config, context=ctx)
    result = graph.invoke(
        Command(resume={"decision": "reject", "reason": "缺失率仍偏高"}), config, context=ctx
    )

    assert result.get("cleaned_data") is None
    assert "cleaned_data" not in result["artifacts"]

    candidate = ctx.artifact_path(CANDIDATE)
    assert candidate.exists()
    assert sha256_file(candidate) == result["review_log"][0].subject_checksum
    assert not (ctx.run_dir / "cleaned_data.csv").exists()


def test_data_prepare_does_not_rerun_on_resume(ctx, dataset_path):
    """A8：恢复时 data_prepare 不重跑，step_records 恰好 1 条。"""
    graph = build_graph()
    config = thread_config("t007-once")

    graph.invoke(_state(dataset_path), config, context=ctx)
    result = _approve(graph, config, ctx)

    records = [record for record in result["step_records"] if record.step == "data_prepare"]
    assert len(records) == 1, "重跑会重写候选文件，整个校验设计就不成立"
    assert [record.attempt for record in records] == [1]


def test_raw_csv_deleted_during_pause_ends_with_rf003(ctx, dataset_path):
    """A12：暂停期间**删除**原始 CSV（不只是改动），恢复后以 RF-003 终止。"""
    graph = build_graph()
    config = thread_config("t007-deleted-raw")

    graph.invoke(_state(dataset_path), config, context=ctx)
    dataset_path.unlink()

    result = _approve(graph, config, ctx)

    assert result["termination"].kind == "escalated"
    assert result["termination"].step == "attribute"
    assert result["error"].code == "RF-003"
    assert result["review_log"] == []


def test_review_log_records_subject_path_for_both_decisions(ctx, dataset_path):
    """A13：approve 与 reject 两条路径的 `review_log[0].subject_path` 都是候选文件相对路径。"""
    # approve
    graph = build_graph()
    config = thread_config("t007-a13-approve")
    graph.invoke(_state(dataset_path), config, context=ctx)
    approved = _approve(graph, config, ctx)
    assert approved["review_log"][0].subject_path == CANDIDATE

    # reject
    graph2 = build_graph()
    config2 = thread_config("t007-a13-reject")
    graph2.invoke(_state(dataset_path), config2, context=ctx)
    rejected = graph2.invoke(
        Command(resume={"decision": "reject", "reason": "缺失率仍偏高"}), config2, context=ctx
    )
    assert rejected["review_log"][0].subject_path == CANDIDATE

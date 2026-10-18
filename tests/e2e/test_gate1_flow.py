"""Fake/fixture ①→⑦ 全链路 e2e：审核通过与审核拒绝两条路径。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.types import Command

from conftest import thread_config
from researchflow import main
from researchflow.core.config import RunContext, load_settings
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import ParamSet
from researchflow.graphs.orchestrator import build_graph
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "attempts")


def _manifest_usage(run_dir: Path, result: dict) -> dict:
    """用真实 state 写一次 manifest，取回 llm_usage 一节。"""
    path = main._write_manifest(run_dir, result, "fake")
    return json.loads(path.read_text(encoding="utf-8"))["llm_usage"]


@pytest.fixture
def gate1_ctx(tmp_path, project_root) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "e2e-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=settings, run_dir=run_dir, llm=FakeLLMClient(), retriever=build_retriever(settings)
    )


def _initial_state() -> dict:
    return {
        "run_id": "e2e-run",
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


def test_gate1_flow_completes_when_review_approved(gate1_ctx):
    graph = build_graph()
    config = thread_config("e2e-approve")

    pending = graph.invoke(_initial_state(), config, context=gate1_ctx)
    assert "__interrupt__" in pending, "③ 必须固定触发一次人工审核"
    payload = pending["__interrupt__"][0].value
    assert payload["stage"] == "data_review"
    assert payload["rows"] > 0
    # T-P0-4：审核人必须看到删除缺失行之前的原始数字。
    assert payload["raw_missing_rate"] > 0
    assert payload["dropped_rows"] >= 0
    assert "duplicates_removed" in payload

    result = graph.invoke(
        Command(resume={"decision": "approve", "reason": "e2e"}), config, context=gate1_ctx
    )

    assert "__interrupt__" not in result
    assert result["step_status"] == {
        "retrieval": "ok",
        "extraction": "ok",
        "data_prepare": "ok",
        "data_review": "ok",
        "compute": "ok",
        "chart": "ok",
        "writing": "ok",
        "validation": "ok",
    }
    assert len(result["literature_refs"]) >= 1
    # T-005 A4：每条文献都有稳定 ID，且标了产出它的检索器
    assert all(ref.ref_id and ref.retriever == "local" for ref in result["literature_refs"])
    assert result["research_question"] is not None
    assert result["compute_result"].metrics["r2"] > 0.8
    assert result["validation_result"].passed is True
    assert result["review_log"][-1].decision == "approve"

    kinds = {ref.kind for ref in result["artifacts"].values()}
    assert {"cleaned_data", "compute_result", "chart", "draft", "validation_report"} <= kinds
    # T-006：state 里的产物路径是相对 run_dir 的相对路径，断言前先解析
    for ref in result["artifacts"].values():
        assert gate1_ctx.artifact_path(ref.path).exists(), f"产物缺失: {ref.path}"
        assert ref.checksum

    draft = gate1_ctx.artifact_path(result["draft_path"]).read_text(encoding="utf-8")
    assert "背景" in draft and "局限" in draft

    # T-001 A5：manifest 的 llm_usage.calls 恰好 2 条，total 等于各条之和。
    assert [call.step for call in result["llm_calls"]] == ["extraction", "writing"]
    assert [call.response_model for call in result["llm_calls"]] == [
        "ResearchQuestion",
        "DraftContent",
    ]
    usage = _manifest_usage(gate1_ctx.run_dir, result)
    assert len(usage["calls"]) == 2
    assert set(usage["by_step"]) == {"extraction", "writing"}
    for key in USAGE_KEYS:
        assert usage["total"][key] == sum(call["usage"][key] for call in usage["calls"])
        assert usage["by_step"]["writing"][key] == usage["calls"][1]["usage"][key]


def test_gate1_flow_terminates_when_review_rejected(gate1_ctx):
    graph = build_graph()
    config = thread_config("e2e-reject")

    pending = graph.invoke(_initial_state(), config, context=gate1_ctx)
    assert "__interrupt__" in pending

    result = graph.invoke(
        Command(resume={"decision": "reject", "reason": "缺失率仍偏高"}), config, context=gate1_ctx
    )

    assert result["step_status"]["data_review"] == "rejected"
    termination = result["termination"]
    assert termination.kind == "rejected"
    assert termination.step == "data_review"
    assert termination.reason == "缺失率仍偏高"
    assert result["review_log"][-1].decision == "reject"
    assert result.get("compute_result") is None
    assert result.get("validation_result") is None
    assert not Path(gate1_ctx.run_dir / "cleaned_data.csv").exists()

    # T-001 A6：打回发生在 ③，writing 实际被调用 0 次，by_step 因此不含 writing。
    assert [call.step for call in result["llm_calls"]] == ["extraction"]
    usage = _manifest_usage(gate1_ctx.run_dir, result)
    assert len(usage["calls"]) == 1
    assert "writing" not in usage["by_step"]
    assert usage["by_step"]["extraction"] == usage["calls"][0]["usage"]
    assert usage["total"] == usage["calls"][0]["usage"]

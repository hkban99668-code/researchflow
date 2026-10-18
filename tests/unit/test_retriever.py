"""T-005：Retriever 契约、稳定 `ref_id` 与注入（A2/A3/A5/A8/A10，附 A4/A6/A7）。

① 不再 import 任何检索实现：注入替身检索器后，`literature_refs` 必须就是替身
给的那些，这证明 Gate 2 换真实检索 API 时节点不用改。
"""

from __future__ import annotations

import re
import uuid

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from conftest import runtime_of, thread_config
from researchflow.core.config import RunContext, load_settings
from researchflow.core.llm import FakeLLMClient
from researchflow.core.retrieval import Retriever
from researchflow.core.schemas import LiteratureRef, ParamSet
from researchflow.graphs.orchestrator import build_graph
from researchflow.nodes.retrieval import retrieval_node
from researchflow.tools.literature import (
    LocalCorpusRetriever,
    build_retriever,
    make_ref_id,
    search_local,
)

TOPIC = "科研数据质量与线性回归可复现性评估"
NO_HIT_TOPIC = "与语料完全无关的课题 xyz"


def _state(topic: str = TOPIC) -> dict:
    return {
        "run_id": "retriever-run",
        "topic": topic,
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


def _context(tmp_path, project_root, retriever=None) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "retriever-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=build_retriever(settings) if retriever is None else retriever,
    )


def _run_chain(tmp_path, project_root, topic: str = TOPIC) -> dict:
    """跑完整条 ①→⑦（审核放行），返回最终 state。"""
    ctx = _context(tmp_path, project_root)
    config = thread_config(uuid.uuid4().hex)
    graph = build_graph()
    graph.invoke(_state(topic), config, context=ctx)
    return graph.invoke(
        Command(resume={"decision": "approve", "reason": "T-005"}), config, context=ctx
    )


# --- A2 契约 ------------------------------------------------------------------


def test_local_corpus_retriever_satisfies_protocol() -> None:
    """A2：`LocalCorpusRetriever` 能当作 `Retriever` 通过类型检查。"""
    retriever: Retriever = LocalCorpusRetriever()

    assert retriever.name == "local"
    assert isinstance(retriever.search(TOPIC, k=3), list)


# --- A3 ref_id ----------------------------------------------------------------


def test_make_ref_id_normalizes_title_and_keeps_year() -> None:
    first = make_ref_id("local", "  A  Study ", 2020)
    second = make_ref_id("local", "a study", 2020)

    assert first == second
    assert first != make_ref_id("local", "a study", 2021)
    assert first != make_ref_id("openalex", "a study", 2020)
    assert re.fullmatch(r"local:[0-9a-f]{12}", first), first


# --- A5 诚实返回 0 条 ----------------------------------------------------------


def test_local_retriever_returns_empty_when_nothing_matches() -> None:
    assert LocalCorpusRetriever().search(NO_HIT_TOPIC, k=6) == []


def test_search_local_no_longer_falls_back_to_whole_corpus() -> None:
    """旧函数同样去掉了"0 命中返回全部语料"的兜底。"""
    assert search_local(NO_HIT_TOPIC, limit=6) == []
    assert len(search_local(TOPIC, limit=6)) == 6


# --- A4 稳定 ID ---------------------------------------------------------------


def test_demo_chain_ref_ids_are_stable_across_runs(tmp_path, project_root) -> None:
    first = _run_chain(tmp_path / "a", project_root)["literature_refs"]
    second = _run_chain(tmp_path / "b", project_root)["literature_refs"]

    assert [ref.ref_id for ref in first] == [ref.ref_id for ref in second]
    assert [ref.retriever for ref in first] == ["local"] * len(first)
    assert all(re.fullmatch(r"local:[0-9a-f]{12}", ref.ref_id) for ref in first)


# --- A6 空检索终止 ------------------------------------------------------------


def test_unmatched_topic_terminates_with_empty(tmp_path, project_root) -> None:
    result = _run_chain(tmp_path, project_root, topic=NO_HIT_TOPIC)

    assert result["termination"] is not None
    assert result["termination"].kind == "empty"
    assert result["termination"].step == "retrieval"
    assert "__interrupt__" not in result
    assert result.get("cleaned_data") is None


# --- A7 缺检索器要报错 --------------------------------------------------------


def test_retrieval_without_retriever_raises_config_error(tmp_path, project_root) -> None:
    ctx = _context(tmp_path, project_root)
    ctx.retriever = None  # 模拟没注入检索器

    update = retrieval_node(_state(), runtime=runtime_of(ctx))

    assert update["error"].code == "RF-001"
    assert update["step_status"] == {"retrieval": "error"}
    assert update["termination"].kind == "error"
    assert update.get("literature_refs") is None


# --- A8 注入替身 --------------------------------------------------------------


def test_retrieval_node_uses_injected_retriever(tmp_path, project_root, stub_retriever) -> None:
    ctx = _context(tmp_path, project_root, retriever=stub_retriever)

    update = retrieval_node(_state(), runtime=runtime_of(ctx))

    assert [ref.ref_id for ref in update["literature_refs"]] == ["stub:ref-1", "stub:ref-2"]
    assert [ref.retriever for ref in update["literature_refs"]] == ["stub", "stub"]
    assert update["step_status"] == {"retrieval": "ok"}


# --- A10 ref_id 校验 ----------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "3f9a0c1b2d4e", "local:", "Local:abc", "loc al:abc"])
def test_ref_id_must_match_pattern(bad: str) -> None:
    with pytest.raises(ValidationError):
        LiteratureRef(ref_id=bad, title="t")


def test_ref_id_accepts_all_sources() -> None:
    assert LiteratureRef(ref_id="openalex:W123", title="t").ref_id == "openalex:W123"
    assert LiteratureRef(ref_id="arxiv:2401.01234", title="t").ref_id == "arxiv:2401.01234"

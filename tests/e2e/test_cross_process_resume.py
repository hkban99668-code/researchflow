"""T-008 A3、A8、A9：跨进程恢复（sqlite checkpointer + `RunContext.from_run_id`）。

A3 的"跨进程"用"丢弃全部对象、只留 run_id 与磁盘文件"来模拟：第一段跑完就
`del` 图 / 上下文 / checkpointer 并 `gc.collect()`，第二段只凭 `from_run_id` 与新建
的对象接着跑。A8 验证 checkpoint.sqlite 的字节不含 API Key 与项目根目录绝对路径
（D-009），A9 验证恢复时 LLM 模式不一致会被拒。
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import httpx
import pytest
import respx
from langgraph.types import Command

from conftest import thread_config
from researchflow.core.config import (
    RunContext,
    load_settings,
    run_meta_payload,
    sha256_file,
    write_run_meta,
)
from researchflow.core.errors import ConfigError
from researchflow.core.llm import FakeLLMClient, build_llm_client
from researchflow.core.schemas import ParamSet
from researchflow.graphs.orchestrator import (
    build_graph,
    close_checkpointer,
    make_checkpointer,
)
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"
DEMO_TOPIC = "examples/demo_topic.yaml"

# A3：五类产物的文件名（与 test_reproducibility 的 ARTIFACT_KINDS 对应）
ARTIFACT_FILES = {
    "cleaned_data": "cleaned_candidate.csv",
    "compute_result": "compute_result.json",
    "chart": "chart.png",
    "draft": "draft.md",
    "validation_report": "validation_report.json",
}

VALID_EXTRACTION = json.dumps(
    {
        "question": "变量如何影响目标变量？",
        "variables": ["x1", "x2", "x3", "y"],
        "hypothesis": "存在可量化的线性关系。",
        "method": "单算法回归。",
        "confidence": 0.6,
    }
)


def _settings(tmp_path: Path, project_root: Path):
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    settings.checkpoint.backend = "sqlite"
    return settings


def _context(settings, run_id: str) -> RunContext:
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=build_retriever(settings),
    )


def _initial_state(run_id: str) -> dict:
    return {
        "run_id": run_id,
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


def _digests(run_dir: Path) -> dict:
    return {kind: sha256_file(run_dir / name) for kind, name in ARTIFACT_FILES.items()}


def test_two_segment_resume_matches_one_shot(tmp_path, project_root):
    """A3：两段式（sqlite + from_run_id 恢复）产物与一次性跑完逐字节相同。"""
    settings = _settings(tmp_path, project_root)

    # 一次性跑完的参照：memory checkpointer，同进程内 approve
    ref_ctx = _context(settings, "oneshot")
    ref_graph = build_graph()
    ref_config = thread_config("oneshot")
    ref_graph.invoke(_initial_state("oneshot"), ref_config, context=ref_ctx)
    ref_graph.invoke(
        Command(resume={"decision": "approve", "reason": "A3"}), ref_config, context=ref_ctx
    )
    ref_digests = _digests(ref_ctx.run_dir)

    # 第一段：sqlite 跑到审核处停下
    run_id = "resume"
    ctx = _context(settings, run_id)
    checkpointer = make_checkpointer("sqlite", ctx.run_dir)
    graph = build_graph(checkpointer)
    config = thread_config(run_id)
    pending = graph.invoke(_initial_state(run_id), config, context=ctx)
    assert "__interrupt__" in pending
    close_checkpointer(checkpointer)
    write_run_meta(
        ctx.run_dir,
        run_meta_payload(
            run_id=run_id,
            topic_path=project_root / DEMO_TOPIC,
            llm_mode="fake",
            settings=settings,
            checkpoint_backend="sqlite",
        ),
    )

    # 模拟跨进程：丢掉全部对象，只保留 run_id 与磁盘文件
    del graph, ctx, checkpointer
    gc.collect()

    # 第二段：from_run_id 重建上下文 + 新建的图与 sqlite checkpointer
    ctx2 = RunContext.from_run_id(run_id, settings)
    checkpointer2 = make_checkpointer("sqlite", ctx2.run_dir)
    graph2 = build_graph(checkpointer2)
    graph2.invoke(
        Command(resume={"decision": "approve", "reason": "A3"}),
        thread_config(run_id),
        context=ctx2,
    )
    close_checkpointer(checkpointer2)

    assert _digests(ctx2.run_dir) == ref_digests


def test_checkpoint_sqlite_does_not_leak_secrets(tmp_path, project_root):
    """A8：checkpoint.sqlite 的字节不含项目根目录绝对路径与测试用的假 API Key。"""
    settings = _settings(tmp_path, project_root)
    settings.llm.mode = "real"
    settings.llm.api_key = "sk-test-should-not-leak"
    settings.llm.model = "test-model"
    settings.llm.base_url = "https://secret.test/v1"

    run_dir = settings.runs_dir / "secret-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=build_llm_client(settings),
        retriever=build_retriever(settings),
    )
    checkpointer = make_checkpointer("sqlite", run_dir)
    graph = build_graph(checkpointer)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://secret.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": VALID_EXTRACTION}}]}
            )
        )
        pending = graph.invoke(
            _initial_state("secret-run"), thread_config("secret-run"), context=ctx
        )
        assert "__interrupt__" in pending

    close_checkpointer(checkpointer)

    raw = (run_dir / "checkpoint.sqlite").read_bytes()
    assert b"sk-test-should-not-leak" not in raw
    assert str(project_root).encode() not in raw


def _write_fake_run_meta(settings, ctx, run_id, project_root):
    """给一个已跑到审核处的 run 写 run_meta.json（llm_mode=fake）。"""
    checkpointer = make_checkpointer("sqlite", ctx.run_dir)
    graph = build_graph(checkpointer)
    graph.invoke(_initial_state(run_id), thread_config(run_id), context=ctx)
    close_checkpointer(checkpointer)
    write_run_meta(
        ctx.run_dir,
        run_meta_payload(
            run_id=run_id,
            topic_path=project_root / DEMO_TOPIC,
            llm_mode="fake",
            settings=settings,
            checkpoint_backend="sqlite",
        ),
    )


def test_resume_defaults_to_recorded_fake_mode_ignoring_env(tmp_path, project_root):
    """B17：恢复 Fake 运行默认用 run_meta 记录的 fake，不受 .env 真实模式影响。"""
    settings = _settings(tmp_path, project_root)
    ctx = _context(settings, "mode-run")
    _write_fake_run_meta(settings, ctx, "mode-run", project_root)

    # 当前环境是真实模式（.env 配了 Key 与模型）
    settings.llm.mode = "real"
    settings.llm.api_key = "sk-now-real"
    settings.llm.model = "test-model"

    ctx2 = RunContext.from_run_id("mode-run", settings)
    assert ctx2.llm.mode == "fake"


def test_resume_explicit_mode_mismatch_raises(tmp_path, project_root):
    """A7：显式 --mode real 恢复 Fake 运行时抛 ConfigError。"""
    settings = _settings(tmp_path, project_root)
    ctx = _context(settings, "mode-run2")
    _write_fake_run_meta(settings, ctx, "mode-run2", project_root)

    with pytest.raises(ConfigError):
        RunContext.from_run_id("mode-run2", settings, mode="real")

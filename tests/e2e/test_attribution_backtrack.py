"""T-014 A4–A10：attribute 节点的归因回溯全链路（Fake 脚本，不调真实模型）。

用脚本化 `FakeLLMClient` 扮演归因 Agent 的工具回路：`chat()` 返回预设的工具调用
序列，`complete()` 仍走 Fake 的固定构造器。数据集用 `examples/sample_dataset.csv`
变异出 y 离群值 / y 打乱 / 缺列三种故障，配合 `AgentAttributor` 跑完整图。
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd
from langgraph.types import Command

from conftest import thread_config
from researchflow import main
from researchflow.core.agent_loop import LoopBudget
from researchflow.core.attribution import AgentAttributor
from researchflow.core.config import RunContext, load_settings, run_meta_payload, write_run_meta
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import AssistantTurn, ParamSet, ToolCall
from researchflow.graphs.orchestrator import (
    build_graph,
    close_checkpointer,
    make_checkpointer,
)
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"
SYSTEM_PROMPT = "你是归因 Agent，先用只读工具查明原因，再调用决策工具。"


def _call(call_id: str, name: str, **kwargs) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(kwargs, ensure_ascii=False))


def _turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(content=None, tool_calls=list(calls))


def _final(text: str) -> AssistantTurn:
    return AssistantTurn(content=text)


def _ctx(tmp_path, project_root, *, script=None, retriever=None) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "t014"
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = FakeLLMClient(script=script) if script is not None else FakeLLMClient()
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=llm,
        retriever=build_retriever(settings) if retriever is None else retriever,
    )
    ctx.attributor = AgentAttributor(
        llm=llm, ctx=ctx, budget=LoopBudget(), system_prompt=SYSTEM_PROMPT
    )
    return ctx


def _state(dataset_path) -> dict:
    return {
        "run_id": "t014",
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


def _write_dataset(tmp_path, project_root, fault, *, seed, **params):
    """把示例数据变异成三种故障之一，写到 tmp，返回路径。"""
    frame = pd.read_csv(project_root / "examples" / "sample_dataset.csv")
    out = frame.copy()
    if fault == "y_outliers":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(frame), size=params.get("count", 8), replace=False)
        out.loc[idx, "y"] = out.loc[idx, "y"] * params.get("factor", 20.0)
    elif fault == "y_shuffled":
        rng = np.random.default_rng(seed)
        out["y"] = rng.permutation(out["y"].to_numpy(dtype=float))
    elif fault == "drop_column":
        out = frame.drop(columns=["x2"])
    else:
        raise AssertionError(f"未知故障: {fault}")
    path = tmp_path / "dataset.csv"
    out.to_csv(path, index=False)
    return path


def _drive(graph, config, ctx, state) -> dict:
    """跑完整链路，遇到 interrupt 一律 approve，直到没有 interrupt 或结束。"""
    result = graph.invoke(state, config, context=ctx)
    while graph.get_state(config).interrupts:
        result = graph.invoke(
            Command(resume={"decision": "approve", "reason": "auto"}), config, context=ctx
        )
    return result


def _records(state: dict, step: str) -> list:
    return [record for record in state.get("step_records") or [] if record.step == step]


def test_a4_recovery_backtracks_to_data_prepare_and_passes(tmp_path, project_root):
    """A4：先 inspect_column 再 propose_backtrack(data_prepare, iqr)，最终 ⑦ 通过。"""
    script = [
        _turn(_call("c1", "inspect_column", column="y")),
        _turn(
            _call(
                "c2",
                "propose_backtrack",
                target_step="data_prepare",
                patch={"y_outlier_strategy": "iqr"},
            )
        ),
    ]
    ctx = _ctx(tmp_path, project_root, script=script)
    dataset = _write_dataset(tmp_path, project_root, "y_outliers", seed=201, count=8, factor=20.0)
    graph = build_graph()
    result = _drive(graph, thread_config("t014-a4"), ctx, _state(dataset))

    assert result["validation_result"].passed is True

    attributions = result.get("attributions") or []
    assert len(attributions) == 1
    assert attributions[0].decision.source == "agent"
    assert attributions[0].decision.action == "backtrack"
    assert attributions[0].decision.target_step == "data_prepare"

    assert [record.attempt for record in _records(result, "data_prepare")] == [1, 2]
    assert [record.attempt for record in _records(result, "data_review")] == [1, 2]
    assert [record.attempt for record in result["review_log"]] == [1, 2]


def test_a5_escalate_to_human_exit_code_5(tmp_path, project_root, monkeypatch):
    """A5：Fake 脚本 escalate_to_human，以 escalated 结束，CLI 退出码 5。"""
    dataset = _write_dataset(tmp_path, project_root, "y_shuffled", seed=301)
    topic = tmp_path / "topic.yaml"
    topic.write_text(
        "topic: 数据质量与回归\n"
        "dataset:\n"
        f"  path: {dataset}\n"
        "  target: y\n"
        "  features: [x1, x2, x3]\n"
        "param_set:\n"
        "  algorithm: linear_regression\n"
        "  seed: 42\n",
        encoding="utf-8",
    )
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)

    script = [_turn(_call("c1", "escalate_to_human", reason="y 整列打乱，关系已被破坏"))]

    def build_attributor(ctx):
        return AgentAttributor(
            llm=FakeLLMClient(script), ctx=ctx, budget=LoopBudget(), system_prompt=SYSTEM_PROMPT
        )

    monkeypatch.setattr("researchflow.main._build_attributor", build_attributor)

    exit_code = main.main(["run", "--topic", str(topic), "--mode", "fake", "--auto-approve"])

    assert exit_code == main.EXIT_ESCALATED
    run_dir = next((tmp_path / "runs").glob("*"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["termination"]["kind"] == "escalated"


def test_a6_budget_stops_after_two_backtracks(tmp_path, project_root):
    """A6：Fake 每次都回到 compute，第 3 次进 attribute 不再调 Attributor。"""
    script = [
        _turn(_call("c1", "propose_backtrack", target_step="compute")),
        _turn(_call("c2", "propose_backtrack", target_step="compute")),
    ]
    ctx = _ctx(tmp_path, project_root, script=script)
    dataset = _write_dataset(tmp_path, project_root, "y_outliers", seed=201, count=8, factor=20.0)
    graph = build_graph()
    result = _drive(graph, thread_config("t014-a6"), ctx, _state(dataset))

    assert result["termination"].kind == "budget_exhausted"

    attributions = result.get("attributions") or []
    assert len(attributions) == 2
    assert [record.decision.action for record in attributions] == ["backtrack", "backtrack"]
    # 两次 decide 各一次 chat，第 3 次被预算拦下、没有发请求
    assert len(ctx.llm.chat_requests) == 2


def test_a7_patch_violation_escalates(tmp_path, project_root):
    """A7：target=compute 却给 y_outlier_strategy，不回溯，以 escalated 结束。"""
    script = [
        _turn(
            _call(
                "c1",
                "propose_backtrack",
                target_step="compute",
                patch={"y_outlier_strategy": "iqr"},
            )
        ),
    ]
    ctx = _ctx(tmp_path, project_root, script=script)
    dataset = _write_dataset(tmp_path, project_root, "y_outliers", seed=201, count=8, factor=20.0)
    graph = build_graph()
    result = _drive(graph, thread_config("t014-a7"), ctx, _state(dataset))

    assert result["termination"].kind == "escalated"
    assert "patch 越权" in result["termination"].reason

    attributions = result.get("attributions") or []
    assert len(attributions) == 1
    assert attributions[0].decision.action == "escalate"
    assert "patch 越权" in attributions[0].decision.reason


def test_a8_error_rule_layer_escalates_without_llm(tmp_path, project_root):
    """A8：缺列报错走规则层转人工，source == rule，Fake LLM 的 chat_requests 为 0。"""
    ctx = _ctx(tmp_path, project_root)  # 不带脚本：规则层不应调用 chat()
    dataset = _write_dataset(tmp_path, project_root, "drop_column", seed=0)
    graph = build_graph()
    result = _drive(graph, thread_config("t014-a8"), ctx, _state(dataset))

    assert result["termination"].kind == "escalated"

    attributions = result.get("attributions") or []
    assert len(attributions) == 1
    assert attributions[0].trigger == "error"
    assert attributions[0].decision.source == "rule"
    assert attributions[0].decision.action == "escalate"
    assert len(ctx.llm.chat_requests) == 0


def test_a9_empty_retrieval_skips_attribute(tmp_path, project_root, empty_retriever):
    """A9：检索为空不进 attribute，照旧以 empty 结束，无归因记录。"""
    ctx = _ctx(tmp_path, project_root, retriever=empty_retriever)
    graph = build_graph()
    result = _drive(
        graph,
        thread_config("t014-a9"),
        ctx,
        _state(project_root / "examples" / "sample_dataset.csv"),
    )

    assert result["termination"].kind == "empty"
    assert (result.get("attributions") or []) == []


def test_a10_agent_no_decision_escalates(tmp_path, project_root):
    """A10：模型没调用决策工具就结束，转人工，reason 说明原因。"""
    script = [_final("数据看起来正常，但我无法给出决定。")]
    ctx = _ctx(tmp_path, project_root, script=script)
    dataset = _write_dataset(tmp_path, project_root, "y_outliers", seed=201, count=8, factor=20.0)
    graph = build_graph()
    result = _drive(graph, thread_config("t014-a10"), ctx, _state(dataset))

    assert result["termination"].kind == "escalated"
    assert "未调用决策工具" in result["termination"].reason


def test_agent_llm_error_escalates_with_exit_code_5(tmp_path, project_root, monkeypatch):
    """必改项 1：归因 Agent 自己抛 LLMError 时降级转人工，退出码 5、attributions 一条。"""
    dataset = _write_dataset(tmp_path, project_root, "y_outliers", seed=201, count=8, factor=20.0)
    topic = tmp_path / "topic.yaml"
    topic.write_text(
        "topic: 数据质量与回归\n"
        "dataset:\n"
        f"  path: {dataset}\n"
        "  target: y\n"
        "  features: [x1, x2, x3]\n"
        "param_set:\n"
        "  algorithm: linear_regression\n"
        "  seed: 42\n",
        encoding="utf-8",
    )
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)

    def build_attributor(ctx):
        # 不给 chat 脚本：decide() 第一次 chat 就抛 LLMError("Fake 脚本已耗尽")。
        return AgentAttributor(
            llm=FakeLLMClient(), ctx=ctx, budget=LoopBudget(), system_prompt=SYSTEM_PROMPT
        )

    monkeypatch.setattr("researchflow.main._build_attributor", build_attributor)

    exit_code = main.main(["run", "--topic", str(topic), "--mode", "fake", "--auto-approve"])

    assert exit_code == main.EXIT_ESCALATED
    run_dir = next((tmp_path / "runs").glob("*"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["termination"]["kind"] == "escalated"
    attributions = manifest["attributions"]
    assert len(attributions) == 1
    decision = attributions[0]["decision"]
    assert decision["source"] == "agent"
    assert decision["action"] == "escalate"
    assert decision["reason"].startswith("Agent 调用失败：RF-002")


def test_fake_mode_cli_escalates_without_agent(tmp_path, project_root, monkeypatch):
    """必改项 2：Fake 模式 CLI 不构造 Agent，validation_failed 走"Fake 模式未配置归因 Agent"。"""
    dataset = _write_dataset(tmp_path, project_root, "y_outliers", seed=201, count=8, factor=20.0)
    topic = tmp_path / "topic.yaml"
    topic.write_text(
        "topic: 数据质量与回归\n"
        "dataset:\n"
        f"  path: {dataset}\n"
        "  target: y\n"
        "  features: [x1, x2, x3]\n"
        "param_set:\n"
        "  algorithm: linear_regression\n"
        "  seed: 42\n",
        encoding="utf-8",
    )
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)

    # 不 monkeypatch _build_attributor：Fake 模式下默认返回 None。
    exit_code = main.main(["run", "--topic", str(topic), "--mode", "fake", "--auto-approve"])

    assert exit_code == main.EXIT_ESCALATED
    run_dir = next((tmp_path / "runs").glob("*"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["termination"]["kind"] == "escalated"
    attributions = manifest["attributions"]
    assert len(attributions) == 1
    decision = attributions[0]["decision"]
    assert decision["source"] == "agent"
    assert decision["reason"] == "Fake 模式未配置归因 Agent"


def test_pause_at_review_backtrack_second_review_resumes(tmp_path, project_root):
    """裁定：回溯到 data_prepare 后的第二次审核，可跨进程恢复（attributor=None）继续到 ⑦ 通过。"""
    script = [
        _turn(_call("c1", "inspect_column", column="y")),
        _turn(
            _call(
                "c2",
                "propose_backtrack",
                target_step="data_prepare",
                patch={"y_outlier_strategy": "iqr"},
            )
        ),
    ]
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_id = "t014-pause-resume"
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = FakeLLMClient(script=script)
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=llm,
        retriever=build_retriever(settings),
    )
    ctx.attributor = AgentAttributor(
        llm=llm, ctx=ctx, budget=LoopBudget(), system_prompt=SYSTEM_PROMPT
    )

    dataset = _write_dataset(tmp_path, project_root, "y_outliers", seed=201, count=8, factor=20.0)
    state = {
        "run_id": run_id,
        "topic": TOPIC,
        "param_set": ParamSet(
            algorithm="linear_regression",
            params={
                "dataset_path": str(dataset),
                "target": "y",
                "features": ["x1", "x2", "x3"],
                "test_size": 0.25,
            },
            seed=42,
        ),
    }

    checkpointer = make_checkpointer("sqlite", run_dir)
    graph = build_graph(checkpointer)
    config = thread_config(run_id)

    # 第一段：跑到第一次审核停下；approve 后回溯到 data_prepare，停在第二次审核。
    graph.invoke(state, config, context=ctx)
    assert graph.get_state(config).interrupts
    graph.invoke(Command(resume={"decision": "approve", "reason": "auto"}), config, context=ctx)
    assert graph.get_state(config).interrupts
    close_checkpointer(checkpointer)
    write_run_meta(
        run_dir,
        run_meta_payload(
            run_id=run_id,
            topic_path=project_root / "examples" / "demo_topic.yaml",
            llm_mode="fake",
            settings=settings,
            checkpoint_backend="sqlite",
        ),
    )

    # 模拟 pause-at-review 退出进程：丢弃图和上下文，只留 run_id 与磁盘文件重建。
    del graph, ctx, checkpointer
    gc.collect()
    ctx2 = RunContext.from_run_id(run_id, settings)
    assert ctx2.attributor is None

    checkpointer2 = make_checkpointer("sqlite", ctx2.run_dir)
    graph2 = build_graph(checkpointer2)
    result = graph2.invoke(
        Command(resume={"decision": "approve", "reason": "auto"}), config, context=ctx2
    )
    close_checkpointer(checkpointer2)

    assert result["validation_result"].passed is True

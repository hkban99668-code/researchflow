"""T-025 tracing 的 e2e：真实 Fake 运行产出 trace.json、回溯、暂停恢复、失败、
OTLP 导出与 CLI `trace` 子命令。"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
from langgraph.types import Command

from conftest import thread_config
from researchflow import main
from researchflow.core.agent_loop import LoopBudget
from researchflow.core.attribution import AgentAttributor
from researchflow.core.config import RunContext, Settings, load_settings
from researchflow.core.llm import FakeLLMClient, summarize_llm_usage
from researchflow.core.schemas import (
    AssistantTurn,
    LLMCallRecord,
    LLMUsage,
    ParamSet,
    StepRecord,
    ToolCall,
)
from researchflow.core.tracing import build_trace, export_trace_otlp, trace_summary
from researchflow.graphs.orchestrator import build_graph
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"
SYSTEM_PROMPT = "你是归因 Agent，先用只读工具查明原因，再调用决策工具。"
BUSINESS_NODES = [
    "retrieval",
    "extraction",
    "data_prepare",
    "data_review",
    "compute",
    "chart",
    "writing",
    "validation",
]


def _patched_settings(tmp_path: Path, project_root: Path, monkeypatch) -> Settings:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)
    return settings


def _run_cli(tmp_path: Path, project_root: Path, monkeypatch, *args: str) -> tuple[int, Path]:
    _patched_settings(tmp_path, project_root, monkeypatch)
    exit_code = main.main(
        [
            "run",
            "--topic",
            str(project_root / "examples" / "demo_topic.yaml"),
            "--mode",
            "fake",
            *args,
        ]
    )
    runs = sorted((tmp_path / "runs").glob("*"))
    assert len(runs) == 1, "一次运行只应生成一个 run 目录"
    return exit_code, runs[0]


def _trace(run_dir: Path) -> dict:
    return json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))


def _step_spans(trace: dict) -> list[dict]:
    return [span for span in trace["spans"] if span["kind"] == "step"]


def _llm_spans(trace: dict) -> list[dict]:
    return [span for span in trace["spans"] if span["kind"] == "llm"]


# --- A1：无故障 Fake 运行的 span 树 -------------------------------------------


def test_a1_fake_run_writes_trace_with_8_steps_and_llm_spans(tmp_path, project_root, monkeypatch):
    exit_code, run_dir = _run_cli(tmp_path, project_root, monkeypatch, "--auto-approve")
    assert exit_code == main.EXIT_OK

    trace = _trace(run_dir)
    assert len(trace["trace_id"]) == 32

    steps = _step_spans(trace)
    assert [span["name"] for span in steps] == BUSINESS_NODES
    assert all(span["status"] == "ok" for span in steps)
    # 1 个 run + 8 个 step
    assert trace["spans"][0]["kind"] == "run" and trace["spans"][0]["name"] == "run"

    # ② extraction、⑥ writing 各挂 1 个 llm span，token 与 manifest llm_usage 一致
    llms = _llm_spans(trace)
    assert [span["name"] for span in llms] == ["llm:ResearchQuestion", "llm:DraftContent"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (
        llms[0]["attributes"]["total_tokens"]
        == manifest["llm_usage"]["by_step"]["extraction"]["total_tokens"]
    )
    assert (
        llms[1]["attributes"]["total_tokens"]
        == manifest["llm_usage"]["by_step"]["writing"]["total_tokens"]
    )
    # manifest 增加 trace 节
    assert manifest["trace"]["total_tokens"] == manifest["llm_usage"]["total"]["total_tokens"]


# --- A3/A4：回溯运行 ----------------------------------------------------------


def _call(call_id: str, name: str, **kwargs) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(kwargs, ensure_ascii=False))


def _turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(content=None, tool_calls=list(calls))


def _write_outlier_dataset(tmp_path: Path, project_root: Path) -> Path:
    import numpy as np

    frame = pd.read_csv(project_root / "examples" / "sample_dataset.csv")
    out = frame.copy()
    rng = np.random.default_rng(201)
    idx = rng.choice(len(frame), size=8, replace=False)
    out.loc[idx, "y"] = out.loc[idx, "y"] * 20.0
    path = tmp_path / "dataset.csv"
    out.to_csv(path, index=False)
    return path


def _backtrack_state(dataset: Path) -> dict:
    return {
        "run_id": "a89cf62eee61",
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


def _backtrack_script() -> list[AssistantTurn]:
    return [
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


def _run_attribution(
    tmp_path: Path,
    project_root: Path,
    script: list[AssistantTurn],
) -> tuple[RunContext, dict]:
    """跑一次带归因 Agent 的完整图（回溯到 data_prepare、审核自动通过）。"""
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "t025-backtrack"
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = FakeLLMClient(script=script)
    ctx = RunContext(
        settings=settings, run_dir=run_dir, llm=llm, retriever=build_retriever(settings)
    )
    ctx.attributor = AgentAttributor(
        llm=llm, ctx=ctx, budget=LoopBudget(), system_prompt=SYSTEM_PROMPT
    )
    dataset = _write_outlier_dataset(tmp_path, project_root)
    graph = build_graph()
    config = thread_config("t025-backtrack")
    result = graph.invoke(_backtrack_state(dataset), config, context=ctx)
    while graph.get_state(config).interrupts:
        result = graph.invoke(
            Command(resume={"decision": "approve", "reason": "auto"}), config, context=ctx
        )
    return ctx, result


def _backtrack_result(tmp_path: Path, project_root: Path) -> tuple[RunContext, dict]:
    return _run_attribution(tmp_path, project_root, _backtrack_script())


def test_a3_backtrack_run_has_two_attempts_for_prepare_and_review(tmp_path, project_root):
    _, result = _backtrack_result(tmp_path, project_root)
    trace = build_trace(result, "a89cf62eee61")
    steps = _step_spans(trace)
    by_name: dict[str, list[dict]] = {}
    for span in steps:
        by_name.setdefault(span["name"], []).append(span)
    assert [span["attributes"]["attempt"] for span in by_name["data_prepare"]] == [1, 2]
    assert [span["attributes"]["attempt"] for span in by_name["data_review"]] == [1, 2]


def test_a4_attribution_agent_writes_tool_spans(tmp_path, project_root):
    _, result = _backtrack_result(tmp_path, project_root)
    trace = build_trace(result, "a89cf62eee61")
    tool_spans = [span for span in trace["spans"] if span["kind"] == "tool"]
    assert [span["name"] for span in tool_spans] == [
        "tool:inspect_column",
        "tool:propose_backtrack",
    ]
    invocations = result["attributions"][0].tool_calls
    assert [span["attributes"]["ok"] for span in tool_spans] == [call.ok for call in invocations]
    assert [span["attributes"]["error_code"] for span in tool_spans] == [
        call.error_code for call in invocations
    ]
    # tool span 挂在 attribute step 下
    attr_step = next(span for span in trace["spans"] if span["name"] == "attribute")
    assert all(span["parent_span_id"] == attr_step["span_id"] for span in tool_spans)

    # 第 2 轮起：Agent 每轮 chat() 是一个 llm span，与 tool span 按发生顺序交错；
    # 该 step 子树里的 token 合计等于 LoopResult.usage（由 AttributionRecord.usage 带出）。
    under_attribute = [
        span for span in trace["spans"] if span["parent_span_id"] == attr_step["span_id"]
    ]
    assert [span["name"] for span in under_attribute] == [
        "llm:tool_loop",
        "tool:inspect_column",
        "llm:tool_loop",
        "tool:propose_backtrack",
    ]
    agent_llms = [span for span in under_attribute if span["kind"] == "llm"]
    assertion_usage = result["attributions"][0].usage
    assert sum(span["attributes"]["total_tokens"] for span in agent_llms) == (
        assertion_usage.total_tokens
    )
    # 汇总记录不再单独成 span：llm 数量等于逐轮 chat() 数（没有 tool_loop 重复 span）
    assert len(agent_llms) == len(result["attributions"][0].llm_turns)

    # 必改项 1：同一个量在两处出现就断言相等
    # ① manifest 的 llm_usage 总 token 与 trace 总 token 一致（Agent 用量不再漏记）
    llm_usage = summarize_llm_usage(main._llm_calls_of(result))
    assert llm_usage["total"]["total_tokens"] == trace_summary(trace)["total_tokens"]
    # ② attribute 这次执行的 StepRecord.llm_usage 与 AttributionRecord.usage 一致
    attribute_record = next(
        record for record in main._step_records_of(result) if record.step == "attribute"
    )
    assert attribute_record.llm_usage is not None
    assert attribute_record.llm_usage.total_tokens == assertion_usage.total_tokens
    # ③ Agent 的用量确实进了 by_step（不是只在 trace 里）
    assert llm_usage["by_step"]["attribute"]["total_tokens"] == assertion_usage.total_tokens


def test_agent_failure_still_records_usage_already_spent(tmp_path, project_root):
    """Agent 调用中途失败降级转人工：失败前已发生的用量仍进入 llm_calls 与 trace。"""
    # 脚本只够第 1 轮（inspect_column）；第 2 轮 chat() 耗尽 → Agent 失败转人工
    _, result = _run_attribution(
        tmp_path, project_root, [_turn(_call("c1", "inspect_column", column="y"))]
    )
    attribution = result["attributions"][0]
    assert attribution.usage is not None
    assert attribution.usage.total_tokens > 0
    # 失败前完成的工具调用与轮次都留下来了
    assert [call.name for call in attribution.tool_calls] == ["inspect_column"]
    assert len(attribution.llm_turns) == 1

    tool_loop_records = [
        call for call in main._llm_calls_of(result) if call.response_model == "tool_loop"
    ]
    assert len(tool_loop_records) == 1
    assert tool_loop_records[0].usage.total_tokens == attribution.usage.total_tokens

    trace = build_trace(result, "a89cf62eee61", model="fake")
    agent_llms = [span for span in trace["spans"] if span["name"] == "llm:tool_loop"]
    assert len(agent_llms) == 1
    assert sum(span["attributes"]["total_tokens"] for span in agent_llms) == (
        attribution.usage.total_tokens
    )
    # manifest llm_usage 与 trace 总 token 仍然相等（失败也不出现两个总数）
    llm_usage = summarize_llm_usage(main._llm_calls_of(result))
    assert llm_usage["total"]["total_tokens"] == trace_summary(trace)["total_tokens"]


# --- A5：暂停后 resume，两段 span 合并且不重复 ----------------------------------


def test_a5_resume_regenerates_trace_with_both_segments(tmp_path, project_root, monkeypatch):
    _patched_settings(tmp_path, project_root, monkeypatch)
    exit1 = main.main(
        [
            "run",
            "--topic",
            str(project_root / "examples" / "demo_topic.yaml"),
            "--mode",
            "fake",
            "--pause-at-review",
        ]
    )
    assert exit1 == main.EXIT_AWAITING_REVIEW
    run_dir = next((tmp_path / "runs").glob("*"))
    # 第一段只有 3 个 step（retrieval/extraction/data_prepare）
    first_trace = _trace(run_dir)
    assert [span["name"] for span in _step_spans(first_trace)] == [
        "retrieval",
        "extraction",
        "data_prepare",
    ]

    exit2 = main.main(["resume", "--run-id", run_dir.name, "--decision", "approve"])
    assert exit2 == main.EXIT_OK

    final_trace = _trace(run_dir)
    steps = _step_spans(final_trace)
    assert [span["name"] for span in steps] == BUSINESS_NODES
    # 没有重复：每个节点只出现一次、attempt 都是 1
    assert all(span["attributes"]["attempt"] == 1 for span in steps)


# --- A6：失败运行仍生成 trace.json 且带 error.code -----------------------------


def test_a6_failed_run_writes_error_trace(tmp_path, project_root, monkeypatch):
    frame = pd.read_csv(project_root / "examples" / "sample_dataset.csv")
    dataset = tmp_path / "dataset.csv"
    frame.drop(columns=["x2"]).to_csv(dataset, index=False)
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
    _patched_settings(tmp_path, project_root, monkeypatch)

    exit_code = main.main(["run", "--topic", str(topic), "--mode", "fake", "--auto-approve"])
    assert exit_code == main.EXIT_ESCALATED

    run_dir = next((tmp_path / "runs").glob("*"))
    trace = _trace(run_dir)
    dp = [span for span in _step_spans(trace) if span["name"] == "data_prepare"]
    assert dp and dp[-1]["status"] == "error"
    assert dp[-1]["attributes"]["error_code"] == "RF-003"


# --- A8：CLI trace 子命令 -----------------------------------------------------


def test_a8_trace_cli_prints_indented_tree(tmp_path, project_root, monkeypatch, capsys):
    _, run_dir = _run_cli(tmp_path, project_root, monkeypatch, "--auto-approve")
    capsys.readouterr()  # 丢弃 run 命令本身的输出
    exit_code = main.main(["trace", "--run-id", run_dir.name])
    out = capsys.readouterr().out
    assert exit_code == main.EXIT_OK
    assert out.splitlines()[0].startswith("run ")
    assert "extraction" in out
    assert "llm:ResearchQuestion" in out


def test_a8_trace_cli_rejects_invalid_and_missing_run_id(tmp_path, project_root, monkeypatch):
    _patched_settings(tmp_path, project_root, monkeypatch)
    assert main.main(["trace", "--run-id", "not-hex!!"]) == main.EXIT_ERROR
    assert main.main(["trace", "--run-id", "ffffffffffff"]) == main.EXIT_ERROR


# --- A9：OTLP 导出（惰性导入 + 发出请求 + 失败不改退出码） ---------------------


def test_a9_no_opentelemetry_import_when_endpoint_unset(tmp_path, project_root):
    code = (
        "import os, sys\n"
        "os.environ['RF_LOAD_DOTENV'] = '0'\n"
        "for k in [n for n in os.environ if n.startswith('RF_LLM_')]:\n"
        "    os.environ.pop(k, None)\n"
        "os.environ.pop('RF_OTLP_ENDPOINT', None)\n"
        "from pathlib import Path\n"
        "import researchflow.main as m\n"
        "from researchflow.core.config import load_settings\n"
        f"settings = load_settings(project_root=Path({str(project_root)!r}))\n"
        f"settings.runs_dir = Path({str(tmp_path)!r}) / 'runs'\n"
        "m.load_settings = lambda *a, **k: settings\n"
        f"m.main(['run', '--topic', str(Path({str(project_root)!r}) / 'examples' / "
        f"'demo_topic.yaml'), '--mode', 'fake', '--auto-approve'])\n"
        "print('HAS_OTLP_EXPORTER', "
        "any(k.startswith('opentelemetry.exporter.otlp') for k in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    assert result.returncode == 0, result.stderr
    assert "HAS_OTLP_EXPORTER False" in result.stdout


def _serve(status: int):
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            received.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):  # 静默访问日志
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, received, thread


def test_a9_export_sends_request_to_v1_traces():
    trace = build_trace(
        {
            "run_id": "a89cf62eee61",
            "step_records": [
                StepRecord(
                    step="extraction",
                    status="ok",
                    attempt=1,
                    started_at="2026-10-13T10:00:01.000+08:00",
                    duration_ms=10,
                )
            ],
            "llm_calls": [
                LLMCallRecord(
                    step="extraction",
                    response_model="ResearchQuestion",
                    usage=LLMUsage(prompt_tokens=100, completion_tokens=50),
                    attempt=1,
                    duration_ms=8,
                )
            ],
        },
        "a89cf62eee61",
        model="fake",
    )
    server, received, thread = _serve(200)
    try:
        # 只给服务基址：程序自己拼 /v1/traces
        export_trace_otlp(
            trace,
            endpoint=f"http://127.0.0.1:{server.server_port}",
            headers={"Authorization": "Bearer test-token"},
        )
        assert len(received) == 1
        assert received[0]["path"] == "/v1/traces"
        sent_headers = {key.lower(): value for key, value in received[0]["headers"].items()}
        assert sent_headers.get("authorization") == "Bearer test-token"
        assert "application/json" in sent_headers.get("content-type", "")

        body = json.loads(received[0]["body"])
        spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        # 导出的 id 与 trace.json 完全一致，能和本地文件对上
        assert {span["spanId"] for span in spans} == {span["span_id"] for span in trace["spans"]}
        by_name = {span["name"]: span for span in spans}
        assert by_name["run"]["traceId"] == trace["trace_id"]
        assert by_name["run"]["endTimeUnixNano"]
        assert by_name["llm:ResearchQuestion"]["parentSpanId"] == by_name["extraction"]["spanId"]
        # 建议项：llm / tool 映射为 CLIENT(3)，step / run 为 INTERNAL(1)
        assert by_name["llm:ResearchQuestion"]["kind"] == 3
        assert by_name["extraction"]["kind"] == 1
        attrs = {
            item["key"]: item["value"] for item in by_name["llm:ResearchQuestion"]["attributes"]
        }
        assert attrs["total_tokens"] == {"intValue": "150"}
        assert attrs["model"] == {"stringValue": "fake"}
        assert attrs["cost"] == {"doubleValue": 0.0}
    finally:
        server.shutdown()
        thread.join()


def test_a9_export_failure_does_not_change_exit_code(tmp_path, project_root, monkeypatch, capsys):
    server, received, thread = _serve(500)
    _patched_settings(tmp_path, project_root, monkeypatch)
    monkeypatch.setenv("RF_OTLP_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    try:
        exit_code = main.main(
            [
                "run",
                "--topic",
                str(project_root / "examples" / "demo_topic.yaml"),
                "--mode",
                "fake",
                "--auto-approve",
            ]
        )
    finally:
        server.shutdown()
        thread.join()
    assert exit_code == main.EXIT_OK
    assert "OTLP 导出失败" in capsys.readouterr().err


# --- A10：trace.json 与 manifest 不含敏感信息 ----------------------------------


def test_a10_trace_and_manifest_have_no_secrets(tmp_path, project_root, monkeypatch):
    settings = _patched_settings(tmp_path, project_root, monkeypatch)
    settings.llm.api_key = "sk-test-should-not-leak"
    exit_code, run_dir = _run_cli(tmp_path, project_root, monkeypatch, "--auto-approve")
    assert exit_code == main.EXIT_OK

    trace_text = (run_dir / "trace.json").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "sk-test-should-not-leak" not in trace_text
    assert "sk-test-should-not-leak" not in json.dumps(manifest)
    assert str(project_root) not in trace_text
    assert str(project_root) not in json.dumps(manifest["trace"])

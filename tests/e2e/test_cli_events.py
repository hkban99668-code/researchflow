"""T-004 A1–A5、A7：走 CLI 入口（`main.main`）的运行事件日志。

现有的通过/打回 e2e 直接调 `graph.invoke`，覆盖不到事件日志（事件由 CLI 在消费
输出流时写出），所以这两条路径从 CLI 入口跑一遍；失败路径复用
`test_gate1_failure_paths.py`（它本来就走 `main.main`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchflow import main
from researchflow.core.config import load_settings

APPROVED_STEPS = [
    "retrieval",
    "extraction",
    "data_prepare",
    "data_review",
    "compute",
    "chart",
    "writing",
    "validation",
]
# ③ 拆成两个节点后，interrupt 由 data_review 触发：它前面的 data_prepare 已经
# 跑完（有自己的 step_end），因此 interrupt 前面是 3 条 step_end。
EXPECTED_TYPES = [
    "run_start",
    "step_end",
    "step_end",
    "step_end",
    "interrupt",
    "resume",
    "step_end",
    "step_end",
    "step_end",
    "step_end",
    "step_end",
    "run_end",
]


def _run_cli(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch, *args: str):
    """跑一次 CLI，返回 (退出码, 事件列表, manifest)。"""
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)

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
    events = [
        json.loads(line)
        for line in (runs[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8"))
    return exit_code, events, manifest


def test_approved_run_writes_one_step_end_per_node(tmp_path, project_root, monkeypatch):
    """A1–A5：approve 路径的事件序列、字段、manifest 对照与 step_status 不回归。"""
    exit_code, events, manifest = _run_cli(tmp_path, project_root, monkeypatch, "--auto-approve")

    assert exit_code == main.EXIT_OK
    # A1
    assert [event["type"] for event in events] == EXPECTED_TYPES
    step_events = [event for event in events if event["type"] == "step_end"]
    assert [event["step"] for event in step_events] == APPROVED_STEPS
    assert events[0]["detail"]["topic"] == manifest["topic"]
    assert events[0]["detail"]["llm_mode"] == "fake"
    assert events[4]["detail"]["stage"] == "data_review"
    assert events[5]["detail"]["decision"] == "approve"

    # A2
    for event in step_events:
        assert event["status"] == "ok"
        assert event["attempt"] == 1
        assert event["duration_ms"] >= 0

    # A3
    assert [(s["step"], s["attempt"], s["duration_ms"]) for s in manifest["steps"]] == [
        (event["step"], event["attempt"], event["duration_ms"]) for event in step_events
    ]

    # A4
    assert manifest["step_status"] == dict.fromkeys(APPROVED_STEPS, "ok")

    # A5
    for step in ("extraction", "writing"):
        record = next(s for s in manifest["steps"] if s["step"] == step)
        assert record["llm_usage"] == manifest["llm_usage"]["by_step"][step]
    for step in ("retrieval", "data_prepare", "data_review", "compute", "chart", "validation"):
        record = next(s for s in manifest["steps"] if s["step"] == step)
        assert record["llm_usage"] is None


def test_rejected_run_ends_with_run_end_carrying_rejection(tmp_path, project_root, monkeypatch):
    """A7：打回路径以 run_end 结束，detail 带 termination.kind == rejected。"""
    monkeypatch.setattr("builtins.input", lambda *args: "reject")

    exit_code, events, manifest = _run_cli(tmp_path, project_root, monkeypatch)

    assert exit_code == main.EXIT_REVIEW_REJECTED
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "terminated"
    assert events[-1]["detail"]["termination"]["kind"] == "rejected"
    assert [event["step"] for event in events if event["type"] == "step_end"] == [
        "retrieval",
        "extraction",
        "data_prepare",
        "data_review",
    ]
    assert manifest["step_status"]["data_review"] == "rejected"

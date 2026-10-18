"""T-008 A4–A7：CLI 两段式（`run --pause-at-review` → `resume`）走 `main.main`。

与 test_cli_events 不同，这里不是在同一进程里 auto-approve，而是分两次 `main.main`
调用，模拟"关掉进程、换个进程 resume"。A6 的"不改文件"通过快照 run 目录全部字节
前后一致来断言。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from researchflow import main
from researchflow.core.config import load_settings

DEMO_TOPIC = "examples/demo_topic.yaml"
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


def _patched_settings(tmp_path: Path, project_root: Path, monkeypatch):
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    settings.checkpoint.backend = "sqlite"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)
    return settings


def _run_pause(tmp_path: Path, project_root: Path, monkeypatch, settings=None) -> tuple[int, Path]:
    """跑 `run --pause-at-review`，返回 (退出码, run_dir)。"""
    settings = settings or _patched_settings(tmp_path, project_root, monkeypatch)
    exit_code = main.main(
        [
            "run",
            "--topic",
            str(project_root / DEMO_TOPIC),
            "--mode",
            "fake",
            "--pause-at-review",
        ]
    )
    runs = sorted((tmp_path / "runs").glob("*"))
    assert len(runs) == 1, "一次运行只应生成一个 run 目录"
    return exit_code, runs[0]


def _events(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _snapshot(run_dir: Path) -> dict[str, bytes]:
    """run 目录下所有文件（相对路径 → 字节），用于"不改文件"断言。"""
    return {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def test_pause_at_review_then_resume_approve(tmp_path, project_root, monkeypatch):
    """A4：两段式退出码、事件序列与 manifest 状态。"""
    exit1, run_dir = _run_pause(tmp_path, project_root, monkeypatch)
    assert exit1 == main.EXIT_AWAITING_REVIEW

    run_id = run_dir.name
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "awaiting_review"

    events = _events(run_dir)
    assert events[-1]["type"] == "interrupt"

    exit2 = main.main(["resume", "--run-id", run_id, "--decision", "approve"])
    assert exit2 == main.EXIT_OK

    events = _events(run_dir)
    types = [event["type"] for event in events]
    assert types.count("run_start") == 1
    assert types.count("run_end") == 1
    step_events = [event for event in events if event["type"] == "step_end"]
    assert len(step_events) == 8
    assert [event["step"] for event in step_events] == BUSINESS_NODES
    # 两段合起来是一份连续的日志：中间恰好一对 interrupt / resume
    assert types.index("interrupt") < types.index("resume") < types.index("run_end")


def test_resume_reject_ends_with_rejection(tmp_path, project_root, monkeypatch):
    """A5：`resume --decision reject --reason x` 以打回结束，退出码与打回路径一致。"""
    _, run_dir = _run_pause(tmp_path, project_root, monkeypatch)
    run_id = run_dir.name

    exit2 = main.main(
        ["resume", "--run-id", run_id, "--decision", "reject", "--reason", "缺失率仍偏高"]
    )
    assert exit2 == main.EXIT_REVIEW_REJECTED

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "terminated"
    assert manifest["termination"]["kind"] == "rejected"
    assert manifest["review_log"][-1]["decision"] == "reject"
    assert manifest["review_log"][-1]["reason"] == "缺失率仍偏高"


def test_resume_finished_run_raises_config_error(tmp_path, project_root, monkeypatch):
    """A6：resume 一个已经跑完的 run 抛 ConfigError，且不改该 run 的任何文件。"""
    _patched_settings(tmp_path, project_root, monkeypatch)
    exit1 = main.main(
        [
            "run",
            "--topic",
            str(project_root / DEMO_TOPIC),
            "--mode",
            "fake",
            "--auto-approve",
        ]
    )
    assert exit1 == main.EXIT_OK

    run_dir = next((tmp_path / "runs").glob("*"))
    before = _snapshot(run_dir)

    exit2 = main.main(["resume", "--run-id", run_dir.name, "--decision", "approve"])
    assert exit2 == main.EXIT_ERROR
    assert _snapshot(run_dir) == before


def test_resume_nonexistent_run_raises_config_error(tmp_path, project_root, monkeypatch):
    """A6：resume 不存在的 run_id 抛 ConfigError，不创建任何目录。"""
    _patched_settings(tmp_path, project_root, monkeypatch)
    before = {str(p.relative_to(tmp_path)): p for p in tmp_path.rglob("*") if p.is_file()}

    exit2 = main.main(["resume", "--run-id", "does-not-exist", "--decision", "approve"])
    assert exit2 == main.EXIT_ERROR

    after = {str(p.relative_to(tmp_path)): p for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def test_resume_ignores_real_env_and_restores_fake(tmp_path, project_root, monkeypatch):
    """A7：.env 为真实模式时，resume 不带任何环境变量即可恢复一个 Fake 运行并跑完。"""
    fake_settings = _patched_settings(tmp_path, project_root, monkeypatch)
    exit1, run_dir = _run_pause(tmp_path, project_root, monkeypatch, settings=fake_settings)
    assert exit1 == main.EXIT_AWAITING_REVIEW

    # 模拟 .env 现在是真实模式（配了 Key 与模型）
    real_settings = load_settings(project_root=project_root)
    real_settings.runs_dir = fake_settings.runs_dir
    real_settings.checkpoint.backend = "sqlite"
    real_settings.llm.mode = "real"
    real_settings.llm.api_key = "sk-fake-not-used"
    real_settings.llm.model = "test-model"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: real_settings)

    exit2 = main.main(["resume", "--run-id", run_dir.name, "--decision", "approve"])
    assert exit2 == main.EXIT_OK


def test_resume_explicit_real_mode_on_fake_run_raises(tmp_path, project_root, monkeypatch):
    """A7：显式 --mode real 恢复 Fake 运行抛 ConfigError。"""
    settings = _patched_settings(tmp_path, project_root, monkeypatch)
    exit1, run_dir = _run_pause(tmp_path, project_root, monkeypatch, settings=settings)
    assert exit1 == main.EXIT_AWAITING_REVIEW

    exit2 = main.main(
        ["resume", "--run-id", run_dir.name, "--decision", "approve", "--mode", "real"]
    )
    assert exit2 == main.EXIT_ERROR


def test_resume_after_raw_deleted_ends_with_rf003_and_run_end(tmp_path, project_root, monkeypatch):
    """A12：暂停期间删除原始 CSV，恢复后以 RF-003 结束，events.jsonl 以 run_end 收尾。"""
    _patched_settings(tmp_path, project_root, monkeypatch)

    dataset = tmp_path / "dataset.csv"
    dataset.write_bytes((project_root / "examples" / "sample_dataset.csv").read_bytes())
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

    exit1 = main.main(["run", "--topic", str(topic), "--mode", "fake", "--pause-at-review"])
    assert exit1 == main.EXIT_AWAITING_REVIEW
    run_dir = next((tmp_path / "runs").glob("*"))

    dataset.unlink()

    exit2 = main.main(["resume", "--run-id", run_dir.name, "--decision", "approve"])
    assert exit2 == main.EXIT_ESCALATED

    events = _events(run_dir)
    assert events[-1]["type"] == "run_end"
    assert events[-1]["error"]["code"] == "RF-003"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["error"]["code"] == "RF-003"


def test_run_meta_has_no_api_key_or_absolute_paths(tmp_path, project_root, monkeypatch):
    """A7：run_meta.json 不含 API Key 与项目根目录绝对路径。"""
    settings = _patched_settings(tmp_path, project_root, monkeypatch)
    settings.llm.api_key = "sk-test-should-not-leak"

    exit1, run_dir = _run_pause(tmp_path, project_root, monkeypatch, settings=settings)
    assert exit1 == main.EXIT_AWAITING_REVIEW

    meta = (run_dir / "run_meta.json").read_text(encoding="utf-8")
    assert "sk-test-should-not-leak" not in meta
    assert str(project_root) not in meta
    assert str(os.path.abspath(project_root)) not in meta

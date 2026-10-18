"""T-P0-1：失败的 run 必须留下可定位的 manifest。

第 2 轮达标标准：业务异常经 @step 收敛进 state，manifest 记录 status/error/step。
"""

from __future__ import annotations

import json

from researchflow import main
from researchflow.core.config import load_settings


def test_failed_run_writes_error_manifest(tmp_path, project_root, monkeypatch):
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"

    source_rules = project_root / "configs" / "rules" / "data_acceptance.yaml"
    broken_rules = tmp_path / "data_acceptance.yaml"
    broken_rules.write_text(
        source_rules.read_text(encoding="utf-8").replace("min_rows: 30", "min_rows: 1000"),
        encoding="utf-8",
    )
    settings.raw.setdefault("paths", {})["data_acceptance_rules"] = str(broken_rules)

    monkeypatch.setattr("researchflow.main.load_settings", lambda *args, **kwargs: settings)

    exit_code = main.main(
        ["run", "--topic", str(project_root / "examples" / "demo_topic.yaml"), "--auto-approve"]
    )

    assert exit_code == main.EXIT_ESCALATED

    manifests = sorted((tmp_path / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1, "失败的 run 必须写出 manifest"

    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest.get("status") == "error"
    assert manifest["error"]["code"] == "RF-003"
    assert manifest["error"]["step"] == "data_prepare"
    assert manifest["error"]["message"]
    assert manifest["run_id"] == manifests[0].parent.name
    assert manifest["step_status"]["data_prepare"] == "error"
    # 节点报错走规则层：attribute 把它转成转人工终止（escalated），原始 error 仍保留在 state
    assert manifest["termination"]["kind"] == "escalated"

    # T-004 A6：失败 run 的最后一条事件是 run_end 且带 error.code，
    # 失败节点的 step_end 状态为 error。
    events = [
        json.loads(line)
        for line in (manifests[0].parent / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "error"
    assert events[-1]["error"]["code"] == "RF-003"
    assert events[-1]["detail"]["termination"]["kind"] == "escalated"
    failed = [e for e in events if e["type"] == "step_end" and e["step"] == "data_prepare"]
    assert failed[-1]["status"] == "error"
    assert failed[-1]["error"]["code"] == "RF-003"


def test_rules_file_missing_is_reported_not_silently_passed(tmp_path, project_root, monkeypatch):
    """规则文件缺失必须报错，而不是用空规则判通过（P0-3）。"""
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    settings.raw.setdefault("paths", {})["data_acceptance_rules"] = str(tmp_path / "not-exist.yaml")
    monkeypatch.setattr("researchflow.main.load_settings", lambda *args, **kwargs: settings)

    exit_code = main.main(
        ["run", "--topic", str(project_root / "examples" / "demo_topic.yaml"), "--auto-approve"]
    )

    assert exit_code == main.EXIT_ESCALATED

    manifests = sorted((tmp_path / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["error"]["code"] == "RF-001"
    assert manifest["error"]["step"] == "data_prepare"

"""T-006 A1–A4、A7：五类产物跨目录逐字节一致。

state 与 manifest 里的产物路径改成相对 `run_dir` 之后，同一份输入在两个不同深度的
`runs_dir` 下运行，五类产物的 sha256 必须两两相同，且产物里不再出现项目根目录或
`runs_dir` 的绝对路径。

两条约定：
1. **不**把 sha256 的具体值写死。chart.png 的字节取决于 matplotlib 版本与系统字体，
   本机（macOS）与 CI（Linux）可能不同；只比较同一台机器上两次运行是否一致。
2. 走 CLI 入口（`main.main`）而不是 `graph.invoke`：`events.jsonl` 由 CLI 消费输出流
   时写出（D-007），直接调图不会产生这个文件，A3 就无从断言。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from researchflow import main
from researchflow.core.config import RunContext, load_settings
from researchflow.core.llm import FakeLLMClient
from researchflow.tools.literature import build_retriever

# A1 要求两个**不同深度**的 runs_dir
RUN_ROOTS = ("a/runs", "b/deeper/runs")
ARTIFACT_KINDS = (
    "cleaned_data",
    "compute_result",
    "chart",
    "draft",
    "validation_report",
)


def _run_once(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    runs_root: str,
    *,
    topic: Path | None = None,
) -> dict:
    """在 `tmp_path/<runs_root>` 下从 CLI 跑一次完整链路。"""
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / runs_root
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)

    payloads: list[dict] = []
    monkeypatch.setattr("researchflow.main._print_review_prompt", payloads.append)

    exit_code = main.main(
        [
            "run",
            "--topic",
            str(topic or project_root / "examples" / "demo_topic.yaml"),
            "--mode",
            "fake",
            "--auto-approve",
        ]
    )
    assert exit_code == main.EXIT_OK, "示例链路应完整通过"

    runs = sorted(settings.runs_dir.glob("*"))
    assert len(runs) == 1, "一次运行只应生成一个 run 目录"
    run_dir = runs[0]

    # 只用 ctx 做路径解析，和节点里的用法一致
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=build_retriever(settings),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    digests = {
        ref["kind"]: hashlib.sha256(ctx.artifact_path(ref["path"]).read_bytes()).hexdigest()
        for ref in manifest["artifacts"]
    }

    texts = {
        "manifest.json": (run_dir / "manifest.json").read_text(encoding="utf-8"),
        "events.jsonl": (run_dir / "events.jsonl").read_text(encoding="utf-8"),
    }
    # ③ 拆成 prepare/review 后，清洗数据落盘为候选文件，审核通过也沿用同一个文件
    for name in (
        "cleaned_candidate.csv",
        "compute_result.json",
        "draft.md",
        "validation_report.json",
    ):
        texts[name] = (run_dir / name).read_text(encoding="utf-8")

    return {
        "payload": payloads[0],
        "manifest": manifest,
        "digests": digests,
        "texts": texts,
        "run_dir": run_dir,
        "runs_dir": settings.runs_dir,
        "project_root": project_root,
        "ctx": ctx,
    }


@pytest.fixture
def runs(tmp_path, project_root, monkeypatch) -> list[dict]:
    """在两个不同深度的 runs_dir 下各跑一次。"""
    return [_run_once(tmp_path, project_root, monkeypatch, root) for root in RUN_ROOTS]


def test_five_artifacts_are_byte_identical_across_runs_dirs(runs):
    """A1：五类产物 sha256 两两相同，且确实落在各自的 run 目录下。"""
    assert set(runs[0]["digests"]) == set(ARTIFACT_KINDS)
    assert runs[0]["digests"] == runs[1]["digests"]

    # 两次运行用不同的 runs_dir，不是同一份文件被比较了两次
    assert runs[0]["runs_dir"] != runs[1]["runs_dir"]
    for run in runs:
        for ref in run["manifest"]["artifacts"]:
            path = run["ctx"].artifact_path(ref["path"])
            assert path.exists(), f"产物缺失: {ref['path']}"
            assert run["run_dir"] == path.parent, f"产物不在 run_dir 下: {path}"


def test_manifest_artifacts_section_is_equal(runs):
    """A2：两次运行的 manifest `artifacts` 字段整体相等（kind/path/checksum）。"""
    assert runs[0]["manifest"]["artifacts"] == runs[1]["manifest"]["artifacts"]

    for manifest in (run["manifest"] for run in runs):
        assert [ref["kind"] for ref in manifest["artifacts"]] == list(ARTIFACT_KINDS)
        for ref in manifest["artifacts"]:
            assert ref["checksum"]
            assert not Path(ref["path"]).is_absolute(), f"产物路径仍是绝对路径: {ref['path']}"


def test_no_absolute_install_paths_in_artifacts(runs):
    """A3：manifest / events / run 目录文本文件里没有 project_root 与 runs_dir 绝对路径。"""
    for run in runs:
        secrets = {
            "project_root": str(run["project_root"]),
            "runs_dir": str(run["runs_dir"]),
            "run_dir": str(run["run_dir"]),
        }
        for name, text in run["texts"].items():
            for label, needle in secrets.items():
                assert needle not in text, f"{name} 里出现了 {label} 的绝对路径: {needle}"


def test_draft_has_no_run_id_and_uses_relative_chart_link(runs):
    """A4：draft.md 不含 run_id，图表是 `](chart.png)` 相对引用。"""
    for run in runs:
        run_id = run["manifest"]["run_id"]
        draft = run["texts"]["draft.md"]

        assert run_id not in draft, f"draft.md 里仍写有 run_id: {run_id}"
        assert "](chart.png)" in draft


def test_draft_min_chars_detail_is_identical(runs):
    """A7：两次运行 validation_report 中 `draft_min_chars` 的 detail 完全相同。"""
    details = []
    for run in runs:
        checks = [
            check
            for check in run["manifest"]["validation"]["checks"]
            if check["name"] == "draft_min_chars"
        ]
        assert len(checks) == 1, "validation_report 里应恰好一条 draft_min_chars"
        details.append(checks[0]["detail"])

    assert details[0] == details[1]

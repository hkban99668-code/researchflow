"""T-006 A5、A6、A9：产物路径的相对性约束与唯一解析入口。

- A5：`ArtifactRef` / `CleanedDataRef` 的 `path` 拒绝绝对路径、`..` 与反斜杠。
- A6：`RunContext` 的两个解析方法对非法 / 越界输入抛 `ArtifactError`。
- A9：③ 审核载荷里的 `dataset` / `rules` 相对项目根目录；给外部绝对路径时不报错，
  且该路径不落进 manifest 与 events。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from researchflow import main
from researchflow.core.config import RunContext, load_settings, relative_to_root
from researchflow.core.errors import ArtifactError
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import ArtifactRef, CleanedDataRef
from researchflow.tools.literature import build_retriever

TOPIC = "科研数据质量与线性回归可复现性评估"
# A5：三类非法路径，两个模型都要拦住
ILLEGAL_PATHS = ("/abs/x", "../x", "a\\b")
REF_MODELS = (ArtifactRef, CleanedDataRef)


@pytest.fixture
def ctx(tmp_path, project_root) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "unit-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=FakeLLMClient(),
        retriever=build_retriever(settings),
    )


@pytest.mark.parametrize("model", REF_MODELS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("path", ILLEGAL_PATHS)
def test_artifact_path_rejects_absolute_parent_and_backslash(model, path):
    """A5：绝对路径、`..`、反斜杠三种写法都不能进 state。"""
    with pytest.raises(ValidationError):
        model(kind="chart", path=path)


@pytest.mark.parametrize("model", REF_MODELS, ids=lambda m: m.__name__)
def test_artifact_path_accepts_relative_posix(model):
    """A5 的对照：普通相对路径仍然合法。"""
    assert model(kind="chart", path="chart.png").path == "chart.png"
    assert model(kind="chart", path="nested/chart.png").path == "nested/chart.png"


@pytest.mark.parametrize("model", REF_MODELS, ids=lambda m: m.__name__)
def test_artifact_path_rejects_empty(model):
    """A5 的边界：空路径也不合法。"""
    with pytest.raises(ValidationError):
        model(kind="chart", path="")


def test_artifact_path_rejects_illegal_relative(ctx):
    """A6：`artifact_path` 对非法 rel 抛 `ArtifactError`。"""
    with pytest.raises(ArtifactError) as exc:
        ctx.artifact_path("../x")

    assert exc.value.code == "RF-005"
    assert ".." in exc.value.message

    with pytest.raises(ArtifactError):
        ctx.artifact_path("/abs/x")


def test_artifact_path_resolves_under_run_dir(ctx):
    """A6 的对照：合法 rel 解析到 run_dir 之下。"""
    assert ctx.artifact_path("draft.md") == ctx.run_dir / "draft.md"
    assert ctx.artifact_path("sub/draft.md") == ctx.run_dir / "sub" / "draft.md"


def test_relative_to_run_rejects_path_outside_run_dir(ctx, tmp_path):
    """A6：不在 run_dir 下的路径无法转成相对路径。"""
    outside = tmp_path / "elsewhere" / "draft.md"

    with pytest.raises(ArtifactError) as exc:
        ctx.relative_to_run(outside)

    assert exc.value.code == "RF-005"
    assert "不在 run 目录下" in exc.value.message


def test_relative_to_run_round_trips_with_artifact_path(ctx):
    """A6 的对照：两个方法互为逆运算，结果是不带平台分隔符的 POSIX 字符串。"""
    rel = ctx.relative_to_run(ctx.artifact_path("sub/draft.md"))

    assert rel == "sub/draft.md"
    assert "\\" not in rel
    assert ctx.artifact_path(rel) == ctx.run_dir / "sub" / "draft.md"


def test_relative_to_root_returns_none_outside_root(tmp_path):
    """接口 §4 的两种分支：根目录下给相对路径，根目录外给 None。"""
    root = tmp_path / "project"

    assert relative_to_root(root / "runs" / "abc", root) == "runs/abc"
    assert relative_to_root(tmp_path / "external.csv", root) is None


def _write_topic(path: Path, dataset_path: str) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(
            {
                "topic": TOPIC,
                "dataset": {"path": dataset_path, "target": "y", "features": ["x1", "x2", "x3"]},
                "param_set": {"algorithm": "linear_regression", "params": {"test_size": 0.25}},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _run_cli(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch, topic: Path):
    """跑一次 CLI，返回 (审核载荷, manifest, events 文本)。"""
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)

    payloads: list[dict] = []
    monkeypatch.setattr("researchflow.main._print_review_prompt", payloads.append)

    exit_code = main.main(["run", "--topic", str(topic), "--mode", "fake", "--auto-approve"])
    assert exit_code == main.EXIT_OK

    run_dir = sorted(settings.runs_dir.glob("*"))[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    events_text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    return payloads[0], manifest, manifest_text, events_text


def test_review_payload_uses_project_root_relative_paths(tmp_path, project_root, monkeypatch):
    """A9 前半：项目内的输入在审核载荷里是相对项目根目录的路径。"""
    topic = tmp_path / "topic.yaml"
    _write_topic(topic, "examples/sample_dataset.csv")

    payload, manifest, _, _ = _run_cli(tmp_path, project_root, monkeypatch, topic)

    assert payload["dataset"] == "examples/sample_dataset.csv"
    assert payload["rules"] == "configs/rules/data_acceptance.yaml"


def test_external_dataset_path_is_kept_but_not_recorded(tmp_path, project_root, monkeypatch):
    """A9 后半：外部绝对路径不报错、原样给 CLI 看，但不进 manifest 与 events。"""
    external = tmp_path / "external_dataset.csv"
    external.write_text(
        (project_root / "examples" / "sample_dataset.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    topic = tmp_path / "external_topic.yaml"
    _write_topic(topic, str(external))

    payload, manifest, manifest_text, events_text = _run_cli(
        tmp_path, project_root, monkeypatch, topic
    )

    # 外部路径原样保留，只用于终端显示
    assert payload["dataset"] == str(external)
    # 规则仍是项目内的，走相对路径
    assert payload["rules"] == "configs/rules/data_acceptance.yaml"

    # 载荷里的绝对路径不能污染产物
    assert str(external) not in manifest_text
    assert str(external) not in events_text
    assert str(tmp_path) not in manifest_text

    # runs_dir 不在项目根目录下，manifest 的 run_dir 写 null
    assert manifest["run_dir"] is None


def test_manifest_records_run_dir_relative_to_project_root(tmp_path):
    """接口 §4：runs_dir 在项目根目录下时，manifest 的 run_dir 是相对路径。"""
    root = tmp_path / "project"
    run_dir = root / "runs" / "abc123"
    run_dir.mkdir(parents=True)

    path = main._write_manifest(run_dir, {"run_id": "abc123"}, "fake", root)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["run_dir"] == "runs/abc123"

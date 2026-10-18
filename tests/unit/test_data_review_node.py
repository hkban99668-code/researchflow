"""T-007 A4、A6、A10：`data_review` 无副作用，④ 读取时校验 checksum。

直接调用节点函数，不跑整条图：`interrupt()` 用 monkeypatch 换成返回决定的假函数，
这样既能拿到"恢复后"的那一次执行，又不需要构造图与 checkpoint。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import runtime_of
from researchflow.core.config import RunContext, load_settings, sha256_file
from researchflow.core.errors import ArtifactError
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import CleanedDataRef, ParamSet
from researchflow.nodes import data_pipeline
from researchflow.nodes.compute import compute_node
from researchflow.nodes.data_pipeline import CANDIDATE_FILENAME, data_prepare_node, data_review_node

DATASET = "examples/sample_dataset.csv"


@pytest.fixture
def ctx(tmp_path, project_root) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "unit-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(settings=settings, run_dir=run_dir, llm=FakeLLMClient())


def _state() -> dict:
    return {
        "run_id": "unit-run",
        "param_set": ParamSet(
            params={"dataset_path": DATASET, "target": "y", "features": ["x1", "x2", "x3"]}
        ),
    }


def _prepare(ctx: RunContext) -> CleanedDataRef:
    return data_prepare_node(_state(), runtime=runtime_of(ctx))["cleaned_candidate"]


def _approve(monkeypatch: pytest.MonkeyPatch, decision: str = "approve") -> None:
    """把 `interrupt()` 换成"人已经给了决定"的假函数。"""
    monkeypatch.setattr(
        data_pipeline, "interrupt", lambda payload: {"decision": decision, "reason": "unit"}
    )


def test_data_review_writes_no_files(ctx, monkeypatch):
    """A4：`data_review` 期间 `write_text` / `write_bytes` / `to_csv` 调用次数为 0。"""
    candidate = _prepare(ctx)
    _approve(monkeypatch)

    calls: dict[str, int] = {"write_text": 0, "write_bytes": 0, "to_csv": 0}
    write_text, write_bytes, to_csv = Path.write_text, Path.write_bytes, pd.DataFrame.to_csv

    def count_write_text(self, *args, **kwargs):
        calls["write_text"] += 1
        return write_text(self, *args, **kwargs)

    def count_write_bytes(self, *args, **kwargs):
        calls["write_bytes"] += 1
        return write_bytes(self, *args, **kwargs)

    def count_to_csv(self, *args, **kwargs):
        calls["to_csv"] += 1
        return to_csv(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", count_write_text)
    monkeypatch.setattr(Path, "write_bytes", count_write_bytes)
    monkeypatch.setattr(pd.DataFrame, "to_csv", count_to_csv)

    update = data_review_node({**_state(), "cleaned_candidate": candidate}, runtime=runtime_of(ctx))

    assert update["step_status"] == {"data_review": "ok"}
    assert calls == {"write_text": 0, "write_bytes": 0, "to_csv": 0}


def test_compute_rejects_candidate_modified_after_approval(ctx, monkeypatch):
    """A6：approve 之后候选文件被改，④ 以 RF-005 结束。"""
    candidate = _prepare(ctx)
    _approve(monkeypatch)

    approved = data_review_node(
        {**_state(), "cleaned_candidate": candidate}, runtime=runtime_of(ctx)
    )
    cleaned = approved["cleaned_data"]

    path = ctx.artifact_path(cleaned.path)
    data = path.read_bytes()
    path.write_bytes(data[:-2] + (b"2" if data[-2:-1] == b"1" else b"1") + data[-1:])

    update = compute_node({**_state(), "cleaned_data": cleaned}, runtime=runtime_of(ctx))

    assert update["error"].code == "RF-005"
    assert update["termination"].kind == "error"
    assert update["termination"].error.code == "RF-005"
    assert update["error"].detail["expected"] == cleaned.checksum
    assert update["error"].detail["actual"] == sha256_file(path)


def test_compute_rejects_candidate_deleted_after_approval(ctx, monkeypatch):
    """A12：approve 之后候选文件被**删除**，④ 以 RF-005 结束（不是 FileNotFoundError）。"""
    candidate = _prepare(ctx)
    _approve(monkeypatch)

    approved = data_review_node(
        {**_state(), "cleaned_candidate": candidate}, runtime=runtime_of(ctx)
    )
    cleaned = approved["cleaned_data"]
    ctx.artifact_path(cleaned.path).unlink()

    update = compute_node({**_state(), "cleaned_data": cleaned}, runtime=runtime_of(ctx))

    assert update["error"].code == "RF-005"
    assert update["termination"].kind == "error"
    assert update["termination"].error.code == "RF-005"


def test_read_verified_rejects_empty_checksum(ctx):
    """A10：checksum 为空串时拒绝读取，不放行。"""
    path = ctx.artifact_path("no_checksum.csv")
    path.write_text("x1,y\n1,2\n", encoding="utf-8")

    with pytest.raises(ArtifactError) as exc:
        ctx.read_verified(CleanedDataRef(path="no_checksum.csv"))

    assert exc.value.code == "RF-005"
    assert "checksum" in exc.value.message


def test_read_verified_rejects_changed_file(ctx):
    """A10 的对照：checksum 不一致时报错，一致时返回路径。"""
    path = ctx.artifact_path("cleaned_candidate.csv")
    path.write_text("x1,y\n1,2\n", encoding="utf-8")
    ref = CleanedDataRef(path="cleaned_candidate.csv", checksum=sha256_file(path))

    assert ctx.read_verified(ref) == path

    path.write_text("x1,y\n1,3\n", encoding="utf-8")
    with pytest.raises(ArtifactError) as exc:
        ctx.read_verified(ref)

    assert exc.value.code == "RF-005"
    assert exc.value.detail["expected"] == ref.checksum
    assert exc.value.detail["actual"] == sha256_file(path)


def test_prepare_writes_candidate_with_raw_checksum(ctx):
    """A5 的前置：候选文件落盘，raw_checksum 是原始 CSV 的 sha256。"""
    candidate = _prepare(ctx)

    assert candidate.path == CANDIDATE_FILENAME
    assert ctx.artifact_path(candidate.path).exists()
    assert candidate.checksum == sha256_file(ctx.artifact_path(candidate.path))
    assert candidate.raw_checksum == sha256_file(ctx.settings.resolve_path(DATASET))

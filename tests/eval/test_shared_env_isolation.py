"""T-031 A4：共用 .env 隔离（evals._env）的命令行优先回归——citation 与 rerank_eval 入口。

``.env`` 里是 A 配置、命令行给 B 配置时，真实层拿到的是 B；没给命令行时拿到 A。
测试全程不发请求：citation 打桩 ``build_llm_client``，rerank_eval 打桩 ``build_embedder``
（返回 HashEmbedder）。evals.run 入口的同款回归在 ``test_env_override.py``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import _env as evals_env  # noqa: E402
from evals.citation import run as citation_run  # noqa: E402
from evals.retrieval import rerank_eval  # noqa: E402

from conftest import requires_retrieval_pool  # noqa: E402
from researchflow.core import config as config_mod  # noqa: E402
from researchflow.core.embedding import HashEmbedder  # noqa: E402
from researchflow.core.schemas import DraftContent  # noqa: E402

# 测试里会被 load_dotenv / 写回覆盖值碰到的键，统一交给 monkeypatch 收尾恢复。
_ENV_KEYS = (
    "RF_LLM_MODEL",
    "RF_LLM_BASE_URL",
    "RF_LLM_STRUCTURED_OUTPUT",
    "RF_LLM_API_KEY",
    "RF_EMBED_MODEL",
    "RF_EMBED_BASE_URL",
    "RF_EMBED_API_KEY",
)


def _make_project(tmp_path: Path) -> Path:
    """最小项目根：settings.yaml + .env（模型是 model-a，带 Key 供真实层检查）。"""
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "settings.yaml").write_text("llm:\n  mode: auto\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "RF_LLM_MODEL=model-a\nRF_LLM_BASE_URL=https://env.test/v1\nRF_LLM_API_KEY=sk-env\n",
        encoding="utf-8",
    )
    return tmp_path


def _patch_env_loader(monkeypatch, root: Path, overrides: dict[str, str]) -> None:
    """模拟"模块加载时保存的命令行覆盖值"并把 _env 的配置加载指到临时项目根。"""
    monkeypatch.setattr(evals_env, "_SHELL_OVERRIDES", overrides)
    monkeypatch.setattr(
        evals_env,
        "load_settings",
        lambda **kwargs: config_mod.load_settings(project_root=root, load_env_file=True),
    )
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class _StubLLM:
    """打桩客户端构造函数：记下构造时的 settings，complete 返回无引用草稿。"""

    captured: list[config_mod.Settings] = []

    def __init__(self, settings: config_mod.Settings) -> None:
        _StubLLM.captured.append(settings)

    def complete(self, prompt: str, schema: type, context: dict | None = None) -> DraftContent:
        assert schema is DraftContent
        return DraftContent(background="背景" * 60, findings="发现", limitation="局限")


# --- citation 入口 -------------------------------------------------------------


@requires_retrieval_pool
def test_citation_real_shell_override_wins_over_env(monkeypatch, tmp_path, capsys):
    root = _make_project(tmp_path)
    _patch_env_loader(monkeypatch, root, {"RF_LLM_MODEL": "model-b"})
    _StubLLM.captured = []
    monkeypatch.setattr(citation_run, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(citation_run, "build_llm_client", _StubLLM)
    monkeypatch.setattr(
        citation_run,
        "_pool_topics",
        lambda: [{"id": "t1", "topic": "课题", "rankings": {"agent": []}, "candidates": []}],
    )

    exit_code = citation_run.main(["--layer", "real", "--confirm-real-calls"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "模型名: model-b（来源: 命令行）" in out
    assert _StubLLM.captured and _StubLLM.captured[0].llm.model == "model-b"


@requires_retrieval_pool
def test_citation_real_reads_env_without_override(monkeypatch, tmp_path, capsys):
    root = _make_project(tmp_path)
    _patch_env_loader(monkeypatch, root, {})
    _StubLLM.captured = []
    monkeypatch.setattr(citation_run, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(citation_run, "build_llm_client", _StubLLM)
    monkeypatch.setattr(
        citation_run,
        "_pool_topics",
        lambda: [{"id": "t1", "topic": "课题", "rankings": {"agent": []}, "candidates": []}],
    )

    exit_code = citation_run.main(["--layer", "real", "--confirm-real-calls"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "模型名: model-a（来源: .env）" in out
    assert _StubLLM.captured and _StubLLM.captured[0].llm.model == "model-a"


# --- rerank_eval 入口 -----------------------------------------------------------


def _pool(tmp_path: Path) -> Path:
    pool = {
        "topics": [
            {
                "id": "t1",
                "topic": "缺失值插补",
                "rankings": {"keywords": [], "agent": []},
                "candidates": [
                    {"ref_id": "stub:m1", "title": "缺失值插补", "abstract": ""},
                    {"ref_id": "stub:m2", "title": "unrelated", "abstract": ""},
                ],
            }
        ]
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    return path


def _labels(tmp_path: Path) -> Path:
    rows = [
        "topic_id,topic,ref_id,title,year,abstract,relevant",
        "t1,缺失值插补,stub:m1,t,2020,,1",
        "t1,缺失值插补,stub:m2,t,2020,,0",
    ]
    path = tmp_path / "labels.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_rerank_real_shell_override_wins_over_env(monkeypatch, tmp_path, capsys):
    root = _make_project(tmp_path)
    _patch_env_loader(monkeypatch, root, {"RF_LLM_MODEL": "model-b"})
    captured: list[config_mod.Settings] = []

    def _stub_build_embedder(settings: config_mod.Settings) -> HashEmbedder:
        captured.append(settings)
        return HashEmbedder()

    monkeypatch.setattr(rerank_eval, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(rerank_eval, "build_embedder", _stub_build_embedder)

    exit_code = rerank_eval.main(
        [
            "--pool",
            str(_pool(tmp_path)),
            "--labels",
            str(_labels(tmp_path)),
            "--layer",
            "real",
            "--confirm-real-calls",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "模型名: model-b（来源: 命令行）" in out
    assert captured and captured[0].llm.model == "model-b"


def test_rerank_real_reads_env_without_override(monkeypatch, tmp_path, capsys):
    root = _make_project(tmp_path)
    _patch_env_loader(monkeypatch, root, {})
    captured: list[config_mod.Settings] = []

    def _stub_build_embedder(settings: config_mod.Settings) -> HashEmbedder:
        captured.append(settings)
        return HashEmbedder()

    monkeypatch.setattr(rerank_eval, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(rerank_eval, "build_embedder", _stub_build_embedder)

    exit_code = rerank_eval.main(
        [
            "--pool",
            str(_pool(tmp_path)),
            "--labels",
            str(_labels(tmp_path)),
            "--layer",
            "real",
            "--confirm-real-calls",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "模型名: model-a（来源: .env）" in out
    assert captured and captured[0].llm.model == "model-a"


# --- _env 本身 -------------------------------------------------------------


def test_isolate_dotenv_is_idempotent(monkeypatch):
    """重复 isolate 不用已清空的环境覆盖首次保存的覆盖值。"""
    monkeypatch.setattr(evals_env, "_SHELL_OVERRIDES", {"RF_LLM_MODEL": "model-b"})
    evals_env.isolate_dotenv()
    assert evals_env._SHELL_OVERRIDES == {"RF_LLM_MODEL": "model-b"}


def test_source_of_three_states(monkeypatch):
    """来源三态互斥：命令行 > .env（load_dotenv 写入）> 默认。"""
    # 前序测试可能经 load_dotenv 把 RF_LLM_* 留在 os.environ 里；先清掉，
    # monkeypatch 收尾时恢复原状。
    for key in ("RF_LLM_MODEL", "RF_LLM_BASE_URL", "RF_LLM_STRUCTURED_OUTPUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(evals_env, "_SHELL_OVERRIDES", {"RF_LLM_MODEL": "model-b"})
    monkeypatch.setenv("RF_LLM_BASE_URL", "https://from-env.test/v1")  # 模拟 load_dotenv 写入
    assert evals_env._source_of("RF_LLM_MODEL") == "命令行"
    assert evals_env._source_of("RF_LLM_BASE_URL") == ".env"
    assert evals_env._source_of("RF_LLM_STRUCTURED_OUTPUT") == "默认"


@pytest.mark.parametrize("module", [citation_run, rerank_eval])
def test_entries_isolated_dotenv_at_import(module):
    """入口模块导入后 RF_LOAD_DOTENV=0 已生效（隔离先于一切配置读取）。"""
    import os

    assert os.environ.get("RF_LOAD_DOTENV") == "0"

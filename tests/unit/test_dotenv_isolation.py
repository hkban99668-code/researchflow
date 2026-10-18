"""T-009 A1、A3：测试与 .env 隔离。"""

from __future__ import annotations

import httpx

from researchflow import main as main_mod
from researchflow.core.config import load_settings

LLM_ENV_KEYS = ("RF_LLM_MODE", "RF_LLM_API_KEY", "RF_LLM_MODEL")


def _make_project_root(tmp_path):
    """造一个假项目根目录：最小 settings.yaml + 含真实格式假 Key 的 .env。"""
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "settings.yaml").write_text("llm:\n  mode: auto\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "RF_LLM_MODE=real\nRF_LLM_API_KEY=sk-fake-not-used\nRF_LLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    return tmp_path


def test_dotenv_isolation_respects_flag(tmp_path, monkeypatch):
    """A1：RF_LOAD_DOTENV=0 时 .env 不生效（fake），load_env_file=True 时生效（real）。"""
    root = _make_project_root(tmp_path)
    # monkeypatch 记录这三个键的原始状态（均不存在），load_dotenv 写入后会在收尾恢复
    for key in LLM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("RF_LOAD_DOTENV", "0")
    settings_off = load_settings(project_root=root)
    assert settings_off.resolve_llm_mode() == "fake"
    assert settings_off.llm.api_key is None

    monkeypatch.delenv("RF_LOAD_DOTENV")
    settings_on = load_settings(project_root=root, load_env_file=True)
    assert settings_on.resolve_llm_mode() == "real"
    assert settings_on.llm.api_key == "sk-fake-not-used"


def test_full_e2e_makes_no_network_calls(tmp_path, project_root, monkeypatch):
    """A3：httpx 一被调用就抛异常，fake 模式完整 e2e 不触发，证明测试不联网。"""
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    settings.checkpoint.backend = "sqlite"
    monkeypatch.setattr("researchflow.main.load_settings", lambda *a, **k: settings)

    def _boom(*args, **kwargs):
        raise AssertionError("测试不应发出任何网络请求")

    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr(httpx, "get", _boom)

    exit_code = main_mod.main(
        [
            "run",
            "--topic",
            str(project_root / "examples" / "demo_topic.yaml"),
            "--mode",
            "fake",
            "--auto-approve",
        ]
    )
    assert exit_code == main_mod.EXIT_OK

"""T-017/T-031 A1/A2/A3：真实层命令行环境变量覆盖 .env 里的模型配置，Fake 层不受影响。

T-031 起隔离逻辑统一到 ``evals._env``：覆盖值保存在 ``evals._env._SHELL_OVERRIDES``，
配置在 ``_env.load_real_settings()`` 里加载（monkeypatch 目标相应变化）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import respx

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import _env as evals_env  # noqa: E402
from evals import run as evals_run  # noqa: E402

from researchflow.core import config as config_mod  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "evals" / "baseline_fake.json"

# 本文件测试里 `.env` 会写到的键；load_dotenv 直接改 os.environ，不用 monkeypatch 时
# 会污染其它测试，所以统一记录原始状态，收尾时恢复。
_ENV_KEYS = (
    "RF_LLM_MODEL",
    "RF_LLM_BASE_URL",
    "RF_LLM_STRUCTURED_OUTPUT",
    "RF_LLM_API_KEY",
)


def _make_project(tmp_path: Path) -> Path:
    """造一个最小项目根：settings.yaml + .env（模型是 model-a）。"""
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "settings.yaml").write_text("llm:\n  mode: auto\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "RF_LLM_MODEL=model-a\n"
        "RF_LLM_BASE_URL=https://env.test/v1\n"
        "RF_LLM_STRUCTURED_OUTPUT=json_object\n",
        encoding="utf-8",
    )
    return tmp_path


def _record_env(monkeypatch) -> None:
    """记录 _ENV_KEYS 的原始状态（均不存在），load_dotenv / 写回后由 monkeypatch 收尾恢复。"""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _boom(*args, **kwargs):
    raise AssertionError("评测不应发出任何网络请求")


def test_preview_shell_override_wins_over_env(monkeypatch, tmp_path, capsys):
    """A1（覆盖）：命令行覆盖 RF_LLM_MODEL=model-b 时，预演打印 model-b，来源为"命令行"。"""
    root = _make_project(tmp_path)
    # 模拟模块加载时命令行里有 RF_LLM_MODEL=model-b（真实层才会写回它）。
    monkeypatch.setattr(evals_env, "_SHELL_OVERRIDES", {"RF_LLM_MODEL": "model-b"})
    monkeypatch.setattr(
        evals_env,
        "load_settings",
        lambda **kwargs: config_mod.load_settings(project_root=root, load_env_file=True),
    )
    _record_env(monkeypatch)

    exit_code = evals_run.main(["--layer", "real", "--split", "test"])

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "模型名: model-b（来源: 命令行）" in out
    # base_url 来自 .env，脱敏后打印，来源标注 .env。
    assert "base_url: https://*.test（来源: .env）" in out


def test_preview_reads_env_file_when_no_shell_override(monkeypatch, tmp_path, capsys):
    """A1（不覆盖）：无命令行覆盖时，预演打印 .env 里的 model-a，来源为".env"。"""
    root = _make_project(tmp_path)
    monkeypatch.setattr(evals_env, "_SHELL_OVERRIDES", {})
    monkeypatch.setattr(
        evals_env,
        "load_settings",
        lambda **kwargs: config_mod.load_settings(project_root=root, load_env_file=True),
    )
    _record_env(monkeypatch)

    exit_code = evals_run.main(["--layer", "real", "--split", "test"])

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "模型名: model-a（来源: .env）" in out


def test_preview_makes_no_requests_even_with_shell_key(monkeypatch, tmp_path):
    """A2：预演不发请求；即使命令行覆盖里带了假 Key，请求数也为 0。"""
    root = _make_project(tmp_path)
    monkeypatch.setattr(
        evals_env,
        "_SHELL_OVERRIDES",
        {"RF_LLM_MODEL": "model-b", "RF_LLM_API_KEY": "sk-fake"},
    )
    monkeypatch.setattr(
        evals_env,
        "load_settings",
        lambda **kwargs: config_mod.load_settings(project_root=root, load_env_file=True),
    )
    _record_env(monkeypatch)

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://env.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={})
        )
        exit_code = evals_run.main(["--layer", "real", "--split", "test"])

    assert exit_code == 2
    assert route.call_count == 0


def test_fake_layer_result_unchanged_with_shell_llm_env(monkeypatch):
    """A3：进程环境里有 RF_LLM_* 时，Fake 层判定结果仍与基线一致。"""
    monkeypatch.setattr(
        evals_env,
        "_SHELL_OVERRIDES",
        {"RF_LLM_MODEL": "model-b", "RF_LLM_MODE": "fake"},
    )
    monkeypatch.setenv("RF_LLM_MODEL", "model-b")
    monkeypatch.setenv("RF_LLM_MODE", "fake")
    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr(httpx, "get", _boom)

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_by_id = {c["id"]: c for c in baseline["cases"]}
    subset = [
        c for c in evals_run.load_cases() if c["id"] in {"F-CLEAN-01", "F-OUT-02", "F-SHUFFLE-01"}
    ]
    report = evals_run.run_fake(cases=subset)

    assert report["summary"]["n_cases"] == 3
    for c in report["cases"]:
        b = baseline_by_id[c["id"]]
        assert c["termination_ok"] == b["termination_ok"]
        assert c["attribution_ok"] == b["attribution_ok"]
        assert c["recovered"] == b["recovered"]

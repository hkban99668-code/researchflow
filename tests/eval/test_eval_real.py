"""T-015 A6/A7/A8：真实层（--layer real --llm-scope agent）用 respx 打桩，不调真实模型。"""

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


def _escalate_response(prompt_tokens: int = 10, completion_tokens: int = 5) -> httpx.Response:
    """一条 escalate_to_human 决策的 chat 响应。"""
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "escalate_to_human",
                                    "arguments": json.dumps(
                                        {"reason": "无法修复", "suspected_step": "data_prepare"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def _env(monkeypatch):
    monkeypatch.setenv("RF_LLM_API_KEY", "test-key")
    monkeypatch.setenv("RF_LLM_MODEL", "test-model")
    monkeypatch.setenv("RF_LLM_BASE_URL", "https://real.test/v1")


def _subset(*ids: str) -> list[dict]:
    cases = evals_run.load_cases()
    return [c for c in cases if c["id"] in set(ids)]


def test_real_layer_reports_fields_and_only_agent_cases_call_llm(monkeypatch):
    """A6：报告含第 5 节全部字段；不走到 attribute 的用例不发请求。"""
    _env(monkeypatch)
    subset = _subset("F-CLEAN-01", "F-OUT-02", "F-SHUFFLE-01")

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://real.test/v1/chat/completions").mock(
            side_effect=lambda request: _escalate_response()
        )
        report = evals_run.run_real(cases=subset, split="all")

    assert report is not None
    assert report["layer"] == "real"
    assert report["llm_scope"] == "agent"
    assert report["model"] == "test-model"

    by_id = {r["id"]: r for r in report["cases"]}
    for case_id in ("F-CLEAN-01", "F-OUT-02", "F-SHUFFLE-01"):
        r = by_id[case_id]
        for field in (
            "decision",
            "tool_calls",
            "agent_tokens",
            "agent_latency_ms",
            "model",
            "extra_body",
        ):
            assert field in r, f"{case_id} 缺字段 {field}"

    # clean 用例不进 attribute：无工具调用、token 为 0。
    clean = by_id["F-CLEAN-01"]
    assert clean["tool_calls"] == []
    assert clean["agent_tokens"] == 0

    # 走到 attribute 的 2 条用例各发 1 次请求（1 次 escalate 决策即终态）。
    assert route.call_count == 2

    # 汇总含第 5 节全部字段（含两个基线的并排字段与无效回溯率）。
    summary = report["summary"]
    for field in (
        "attribution_ok",
        "attribution_ok_baseline",
        "attribution_ok_smart_baseline",
        "invalid_backtrack_rate_baseline",
        "invalid_backtrack_rate_smart_baseline",
        "recovered",
        "escalate",
        "invalid_args_rate",
        "tokens",
    ):
        assert field in summary, f"summary 缺字段 {field}"

    # 决策与工具调用字段内容正确（suspected_step 写进 target_step）。
    decision = by_id["F-OUT-02"]["decision"]
    assert decision["action"] == "escalate"
    assert decision["target_step"] == "data_prepare"
    assert by_id["F-OUT-02"]["tool_calls"][0]["name"] == "escalate_to_human"
    assert by_id["F-OUT-02"]["tool_calls"][0]["ok"] is True


def test_real_layer_missing_key_exits_2_without_requests(monkeypatch):
    """A7：没有 Key 时 --layer real（已确认）退出码 2，不发请求。"""
    # 用无 Key 的 settings 替身：真实层读 .env，测试里 .env 可能带真 Key，
    # 所以这里直接把 _env.load_settings 换掉，避免依赖本机 .env 状态（T-031 起真实层
    # 配置在 evals._env.load_real_settings 里加载）。
    settings = config_mod.load_settings(load_env_file=False)
    settings.llm.api_key = None
    settings.llm.model = "test-model"
    monkeypatch.setattr(evals_env, "load_settings", lambda **kwargs: settings)

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://real.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={})
        )
        exit_code = evals_run.main(["--layer", "real", "--split", "test", "--confirm-real-calls"])

    assert exit_code == 2
    assert route.call_count == 0


def test_real_layer_without_confirm_exits_2_even_with_key(monkeypatch, tmp_path):
    """必改项 1：不带 --confirm-real-calls 时，即使 .env 含假 Key 可读，请求数也为 0。"""
    # 用 tmp_path 造一个含假 Key 的 .env 和一个最小 settings.yaml，让 load_settings 可读。
    (tmp_path / ".env").write_text(
        "RF_LLM_API_KEY=sk-fake\nRF_LLM_MODEL=test-model\n", encoding="utf-8"
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "settings.yaml").write_text("{}\n", encoding="utf-8")

    # monkeypatch 记录这两个键的原始状态，load_dotenv 写入后会在收尾恢复，避免污染其它测试。
    for key in ("RF_LLM_API_KEY", "RF_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(
        evals_env,
        "load_settings",
        lambda **kwargs: config_mod.load_settings(project_root=tmp_path, load_env_file=True),
    )

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://real.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={})
        )
        exit_code = evals_run.main(["--layer", "real", "--split", "test"])

    assert exit_code == 2
    assert route.call_count == 0


def test_real_layer_stops_on_token_budget(monkeypatch):
    """A8：--max-total-tokens 生效：单条用量很大时超出即停止，报告标明。"""
    _env(monkeypatch)
    subset = _subset("F-OUT-02", "F-OUT-04", "F-OUT-06")

    with respx.mock(assert_all_called=True) as router:
        router.post("https://real.test/v1/chat/completions").mock(
            side_effect=lambda request: _escalate_response(
                prompt_tokens=25000, completion_tokens=25000
            )
        )
        report = evals_run.run_real(cases=subset, split="all", max_total_tokens=30000)

    assert report is not None
    assert report["summary"]["budget_exhausted_early"] is True
    assert report["summary"]["n_cases"] < len(subset)
    assert report["summary"]["tokens"]["total"] > report["max_total_tokens"]

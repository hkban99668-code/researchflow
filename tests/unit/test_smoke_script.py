"""T-009 A4–A6：冒烟脚本的配置校验、报告结构与 Key 脱敏、单项失败不中断。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import respx

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import smoke_real  # noqa: E402

from researchflow.core.config import load_settings  # noqa: E402

BASE_URL = "https://smoke.test/v1"
KEY = "sk-test-leak"

RQ_JSON = json.dumps(
    {
        "question": "x1、x2、x3 如何影响 y？",
        "variables": ["x1", "x2", "x3", "y"],
        "hypothesis": "存在线性关系",
        "method": "线性回归",
        "confidence": 0.6,
    }
)
DRAFT_JSON = json.dumps({"background": "背景", "findings": "发现", "limitation": "局限"})
S4_TOOL_CALLS = [
    {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "run_direct_compute",
            "arguments": json.dumps(
                {
                    "csv_path": "examples/sample_dataset.csv",
                    "target": "y",
                    "features": ["x1", "x2", "x3"],
                }
            ),
        },
    }
]
S5_TOOL_CALLS = [
    {
        "id": "c1",
        "type": "function",
        "function": {"name": "search_literature", "arguments": json.dumps({"query": "回归"})},
    },
    {
        "id": "c2",
        "type": "function",
        "function": {
            "name": "check_data_quality",
            "arguments": json.dumps({"csv_path": "examples/sample_dataset.csv"}),
        },
    },
]


def _chat_response(content: str | None, tool_calls=None):
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _models_response():
    return httpx.Response(200, json={"data": [{"id": "smoke-model"}]})


def _error_response(status: int = 400):
    return httpx.Response(status, json={"error": {"message": f"mock {status}"}})


def _settings(tmp_path, project_root, base_url=BASE_URL):
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    settings.llm.mode = "real"
    settings.llm.api_key = KEY
    settings.llm.model = "smoke-model"
    settings.llm.base_url = base_url
    settings.llm.structured_output = "json_schema"
    settings.checkpoint.backend = "sqlite"
    return settings


def _all_pass_responses():
    """S1×3、S2、S3×3、S4、S5 首轮、S5 终轮、S9 探测，共 11 个 chat 响应。"""
    return [
        _chat_response(RQ_JSON),
        _chat_response(RQ_JSON),
        _chat_response(RQ_JSON),
        _chat_response(DRAFT_JSON),
        _chat_response(RQ_JSON),
        _chat_response(RQ_JSON),
        _chat_response(RQ_JSON),
        _chat_response(None, S4_TOOL_CALLS),
        _chat_response(None, S5_TOOL_CALLS),
        _chat_response("文献与数据质量检查完成。"),
        _chat_response(RQ_JSON),
    ]


def _install_mock_router(router, chat_side_effect):
    router.get(f"{BASE_URL}/models").mock(return_value=_models_response())
    router.post(f"{BASE_URL}/chat/completions").mock(side_effect=chat_side_effect)


def _fake_main_factory(runs_dir):
    """替身 main.main：造一个带 manifest 与 checkpoint.sqlite 的 run 目录。"""

    def fake_main(argv):
        run_dir = runs_dir / "smoke-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "llm_usage": {
                "total": {"total_tokens": 123},
                "calls": [
                    {"step": "extraction", "structured_output": "json_schema"},
                    {"step": "writing", "structured_output": "json_schema"},
                ],
            }
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "checkpoint.sqlite").write_bytes(b"not-secret")
        return 0

    return fake_main


def _required_fields():
    return {"name", "passed", "detail", "attempts", "usage", "latency_ms"}


def test_main_exits_2_when_key_missing(capsys, monkeypatch):
    """A4：无 Key 时退出码 2，输出指出缺 RF_LLM_API_KEY。"""
    for key in ("RF_LLM_MODE", "RF_LLM_API_KEY", "RF_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)

    assert smoke_real.main([]) == smoke_real.EXIT_MISSING_CONFIG
    assert "RF_LLM_API_KEY" in capsys.readouterr().out


def test_s0_sends_authorization_header():
    """A8：S0 请求带 Authorization 头，报告 detail 里不出现 Key。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.get(f"{BASE_URL}/models").mock(return_value=_models_response())
        result = smoke_real.check_models(BASE_URL, KEY)

    assert result["passed"] is True
    assert route.calls[0].request.headers["Authorization"] == f"Bearer {KEY}"
    assert KEY not in result["detail"]


def test_smoke_checks_report_structure_and_redacts_key(tmp_path, project_root, monkeypatch):
    """A5：各检查函数跑一遍，报告结构一致，Key 不出现在报告文件里。"""
    settings = _settings(tmp_path, project_root)
    monkeypatch.setattr("researchflow.main.main", _fake_main_factory(settings.runs_dir))

    with respx.mock(assert_all_called=True) as router:
        _install_mock_router(router, _all_pass_responses())
        results = smoke_real.run_checks(settings)

    assert [r["name"] for r in results] == [f"S{i}" for i in range(11)]
    for result in results:
        assert _required_fields() <= set(result), result["name"]

    # 人为注入一次 Key，验证 write_reports 的兜底替换
    results[0]["detail"] = f"请求带 Key: {KEY} 应被脱敏"
    json_path, md_path = smoke_real.write_reports(
        results,
        base_url="https://*.smoke.test",
        model="smoke-model",
        repeat=3,
        key=KEY,
        report_dir=tmp_path,
    )
    assert KEY not in json_path.read_text(encoding="utf-8")
    assert KEY not in md_path.read_text(encoding="utf-8")


def test_s4_400_marks_failed_and_rest_still_run(tmp_path, project_root, monkeypatch):
    """A6：S4 返回 400 记为失败，S5–S8 照常执行。"""
    settings = _settings(tmp_path, project_root)
    monkeypatch.setattr("researchflow.main.main", _fake_main_factory(settings.runs_dir))

    responses = _all_pass_responses()
    responses[7] = _error_response(400)

    with respx.mock(assert_all_called=True) as router:
        _install_mock_router(router, responses)
        results = smoke_real.run_checks(settings)

    by_name = {r["name"]: r for r in results}
    assert by_name["S4"]["passed"] is False
    assert by_name["S5"]["passed"] is True
    assert by_name["S6"]["name"] == "S6"
    assert by_name["S7"]["passed"] is True
    assert by_name["S8"]["passed"] is True
    assert by_name["S9"]["passed"] is True
    assert by_name["S10"]["passed"] is True


def test_s5_detail_includes_arguments_without_key(tmp_path, project_root, monkeypatch):
    """T-013 A6：S5 的 detail 包含每次工具调用的 arguments，且不含 Key。"""
    settings = _settings(tmp_path, project_root)
    monkeypatch.setattr("researchflow.main.main", _fake_main_factory(settings.runs_dir))

    with respx.mock(assert_all_called=True) as router:
        _install_mock_router(router, _all_pass_responses())
        results = smoke_real.run_checks(settings)

    s5 = next(r for r in results if r["name"] == "S5")
    assert s5["passed"] is True
    assert "arguments=" in s5["detail"]
    assert "sample_dataset.csv" in s5["detail"]
    assert KEY not in s5["detail"]

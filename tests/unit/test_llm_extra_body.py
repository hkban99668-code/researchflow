"""T-015 A4：`RF_LLM_EXTRA_BODY` 原样合并进请求 body；非法 JSON 抛 ConfigError。"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from researchflow.core.config import PROJECT_ROOT, load_settings
from researchflow.core.errors import ConfigError
from researchflow.core.llm import OpenAICompatClient


def _response() -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )


def test_extra_body_parsed_from_env(monkeypatch):
    """RF_LLM_EXTRA_BODY 是 JSON 对象，读进 settings.llm.extra_body。"""
    monkeypatch.setenv("RF_LLM_EXTRA_BODY", '{"enable_thinking": false}')
    settings = load_settings(project_root=PROJECT_ROOT)
    assert settings.llm.extra_body == {"enable_thinking": False}


def test_extra_body_merged_into_request(monkeypatch):
    """A4：respx 捕获的请求 body 含 extra_body 键，标准字段保持权威。"""
    monkeypatch.setenv("RF_LLM_EXTRA_BODY", '{"enable_thinking": false}')
    settings = load_settings(project_root=PROJECT_ROOT)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://extra.test/v1/chat/completions").mock(return_value=_response())
        client = OpenAICompatClient(
            base_url="https://extra.test/v1",
            api_key="k",
            model="m",
            extra_body=settings.llm.extra_body,
        )
        client.chat([{"role": "user", "content": "hi"}])

        body = json.loads(route.calls[0].request.content)
        assert body["enable_thinking"] is False
        assert body["model"] == "m"  # 标准字段不被 extra_body 覆盖


def test_extra_body_invalid_json_raises_config_error(monkeypatch):
    """A4：非法 JSON 抛 ConfigError。"""
    monkeypatch.setenv("RF_LLM_EXTRA_BODY", "not-json")
    with pytest.raises(ConfigError):
        load_settings(project_root=PROJECT_ROOT)


def test_extra_body_non_object_raises_config_error(monkeypatch):
    """RF_LLM_EXTRA_BODY 是合法 JSON 但不是对象时，抛 ConfigError。"""
    monkeypatch.setenv("RF_LLM_EXTRA_BODY", "[1, 2, 3]")
    with pytest.raises(ConfigError):
        load_settings(project_root=PROJECT_ROOT)

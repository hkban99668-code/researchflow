"""T-003：严格 json_schema 结构化输出与原生 tool calling（A1–A6）。

HTTP 打桩沿用 `test_llm_retry.py` 的 respx 做法，不访问真实网络。
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from researchflow.core.config import PROJECT_ROOT, load_settings
from researchflow.core.errors import ConfigError
from researchflow.core.llm import OpenAICompatClient, probe_structured_output
from researchflow.core.schemas import LLMUsage, ResearchQuestion
from researchflow.core.tools import strict_json_schema

PROMPT = "请根据文献提炼一个可检验的研究问题。"
NULL_PAYLOAD = {
    "question": "q",
    "variables": None,
    "hypothesis": None,
    "method": None,
    "confidence": None,
}
VALID_RQ = {
    "question": "变量如何影响目标变量？",
    "variables": ["x1", "y"],
    "hypothesis": "存在可量化的线性关系。",
    "method": "单算法回归。",
    "confidence": 0.6,
}
TOOLS = [{"type": "function", "function": {"name": "search_literature"}}]


def _response(content: str | None = None, **extra: object) -> httpx.Response:
    message: dict = {"role": "assistant", "content": content}
    message.update(extra)  # type: ignore[arg-type]
    return httpx.Response(200, json={"choices": [{"message": message}]})


def _user_content(route: respx.Route) -> str:
    """取捕获到的请求体里最后一条 user 消息。"""
    body = json.loads(route.calls[0].request.content)
    return [m["content"] for m in body["messages"] if m["role"] == "user"][-1]


def _request_body(route: respx.Route) -> dict:
    return json.loads(route.calls[0].request.content)


# --- A1 / A2 结构化输出的两种模式 ----------------------------------------------


def test_json_schema_mode_sends_strict_schema_and_field_notes():
    """A1：response_format 为 json_schema strict，user 消息含逐字段说明、不含完整 schema 文本。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://schema.test/v1/chat/completions").mock(
            return_value=_response(json.dumps(NULL_PAYLOAD))
        )
        client = OpenAICompatClient(base_url="https://schema.test/v1", api_key="k", model="m")
        client.complete(PROMPT, ResearchQuestion)

        body = _request_body(route)
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "ResearchQuestion",
                "schema": strict_json_schema(ResearchQuestion),
                "strict": True,
            },
        }
        user = _user_content(route)
        assert user.startswith(PROMPT)
        # 每个字段一行说明（字段名 + 类型 + 必填/可选）
        for field in ("question", "variables", "hypothesis", "method", "confidence"):
            assert f"- {field}（" in user
        # 不拼完整 JSON Schema 文本
        assert "JSON Schema" not in user
        assert "additionalProperties" not in user


def test_json_schema_mode_first_try_valid_with_schema_ignoring_provider():
    """A2：模拟「忽略 schema、只看 prompt」的 Provider，prompt 有字段说明后首轮即成功。"""

    def side_effect(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = [m["content"] for m in body["messages"] if m["role"] == "user"][-1]
        # 只有 prompt 里出现了字段说明，才返回合法 JSON；否则返回缺字段的 JSON。
        if all(name in user for name in ("question", "variables", "hypothesis")):
            return _response(json.dumps(VALID_RQ))
        return _response(json.dumps({"question": "只有问题"}))

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://ignore.test/v1/chat/completions").mock(side_effect=side_effect)
        client = OpenAICompatClient(
            base_url="https://ignore.test/v1",
            api_key="k",
            model="m",
            structured_output="json_schema",
        )
        result = client.complete(PROMPT, ResearchQuestion)

        assert result.variables == VALID_RQ["variables"]
        assert route.call_count == 1, "字段说明让首轮就拿到合法输出，不应重试"
        assert client.last_usage is not None and client.last_usage.attempts == 1


def test_json_object_mode_keeps_legacy_request_body():
    """A2：降级模式请求体与本卡之前一致：json_object + prompt 内 schema。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://legacy.test/v1/chat/completions").mock(
            return_value=_response(json.dumps(NULL_PAYLOAD))
        )
        client = OpenAICompatClient(
            base_url="https://legacy.test/v1",
            api_key="k",
            model="m",
            structured_output="json_object",
        )
        client.complete(PROMPT, ResearchQuestion)

        body = _request_body(route)
        assert body["response_format"] == {"type": "json_object"}
        user = _user_content(route)
        assert user.startswith(PROMPT)
        assert "只输出一个 JSON 对象" in user
        assert user.endswith(json.dumps(ResearchQuestion.model_json_schema(), ensure_ascii=False))


# --- A3 strict 模式的 null 落默认值 --------------------------------------------


def test_null_optional_fields_fall_back_to_defaults_on_first_attempt():
    """A3：strict 模式把可选字段填成 null，第一次就该解析成功。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://null.test/v1/chat/completions").mock(
            return_value=_response(json.dumps(NULL_PAYLOAD))
        )
        client = OpenAICompatClient(base_url="https://null.test/v1", api_key="k", model="m")
        result = client.complete(PROMPT, ResearchQuestion)

        assert route.call_count == 1, "不该因为 null 而重试"
        assert result.question == "q"
        assert result.variables == []
        assert result.hypothesis == ""
        assert result.method == ""
        assert result.confidence == 0.0
        assert client.last_usage is not None
        assert client.last_usage.attempts == 1


# --- A4 / A5 chat() ------------------------------------------------------------


def test_chat_parses_tool_calls_and_null_content():
    """A4：content 为 null 时是 None（不是字符串 "None"），tool_calls 原样解析。"""
    payload = {
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
                                "name": "search_literature",
                                "arguments": '{"query": "回归", "limit": 2}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "check_data_quality",
                                "arguments": '{"csv_path": "examples/sample_dataset.csv"}',
                            },
                        },
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12},
    }
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://chat.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        client = OpenAICompatClient(base_url="https://chat.test/v1", api_key="k", model="m")
        turn = client.chat(
            [{"role": "user", "content": "分析这份数据"}], tools=TOOLS, tool_choice="auto"
        )

        assert turn.content is None
        assert [call.id for call in turn.tool_calls] == ["call_1", "call_2"]
        assert [call.name for call in turn.tool_calls] == [
            "search_literature",
            "check_data_quality",
        ]
        assert json.loads(turn.tool_calls[0].arguments) == {"query": "回归", "limit": 2}
        assert client.last_usage == LLMUsage(
            prompt_tokens=30, completion_tokens=12, total_tokens=42, attempts=1
        )
        body = _request_body(route)
        assert body["tools"] == TOOLS
        assert body["tool_choice"] == "auto"


def test_chat_without_tool_support_raises_config_error_before_request():
    """A5：声明不支持 tool calling 时，带 tools 的 chat() 直接报错，不发请求。"""
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://notools.test/v1/chat/completions").mock(
            return_value=_response("ok")
        )
        client = OpenAICompatClient(
            base_url="https://notools.test/v1", api_key="k", model="m", supports_tools=False
        )
        with pytest.raises(ConfigError):
            client.chat([{"role": "user", "content": "hi"}], tools=TOOLS)

        assert route.call_count == 0


# --- A6 能力配置 ---------------------------------------------------------------


def test_invalid_structured_output_setting_raises_config_error(monkeypatch):
    """A6：非法的 RF_LLM_STRUCTURED_OUTPUT 抛 ConfigError。"""
    monkeypatch.setenv("RF_LLM_STRUCTURED_OUTPUT", "yaml")
    with pytest.raises(ConfigError):
        load_settings(project_root=PROJECT_ROOT)


def test_settings_reads_llm_capability_flags(monkeypatch):
    """合法取值要能读进来，supports_tools 的 0/false 都算 False。"""
    monkeypatch.setenv("RF_LLM_STRUCTURED_OUTPUT", "json_object")
    monkeypatch.setenv("RF_LLM_SUPPORTS_TOOLS", "0")
    settings = load_settings(project_root=PROJECT_ROOT)
    assert settings.llm.structured_output == "json_object"
    assert settings.llm.supports_tools is False

    monkeypatch.setenv("RF_LLM_SUPPORTS_TOOLS", "false")
    assert load_settings(project_root=PROJECT_ROOT).llm.supports_tools is False

    monkeypatch.setenv("RF_LLM_SUPPORTS_TOOLS", "1")
    assert load_settings(project_root=PROJECT_ROOT).llm.supports_tools is True


# --- A3 probe_structured_output ----------------------------------------------


def test_probe_structured_output_detects_enforced():
    """A3：执行约束的 Provider（返回全部字段）→ enforced=True。"""
    with respx.mock(assert_all_called=True) as router:
        router.post("https://probe-on.test/v1/chat/completions").mock(
            return_value=_response(json.dumps(VALID_RQ))
        )
        client = OpenAICompatClient(base_url="https://probe-on.test/v1", api_key="k", model="m")
        result = probe_structured_output(client, ResearchQuestion)

        assert result["enforced"] is True
        assert "全部字段都在" in result["detail"]


def test_probe_structured_output_detects_not_enforced():
    """A3：忽略约束的 Provider（缺字段）→ enforced=False，detail 列出缺的字段。"""
    with respx.mock(assert_all_called=True) as router:
        router.post("https://probe-off.test/v1/chat/completions").mock(
            return_value=_response(json.dumps({"question": "只有问题"}))
        )
        client = OpenAICompatClient(base_url="https://probe-off.test/v1", api_key="k", model="m")
        result = probe_structured_output(client, ResearchQuestion)

        assert result["enforced"] is False
        assert "variables" in result["detail"]

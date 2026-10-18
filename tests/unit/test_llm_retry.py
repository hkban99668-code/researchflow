"""T-P0-6：LLM 结构化输出失败必须重试。

第 2 轮达标标准：`OpenAICompatClient.complete` 最多重试 2 次（共 3 次请求），
且重试时把上一次的校验错误摘要回传给模型。`respx` 打桩，不访问真实网络。
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from researchflow.core.errors import LLMError
from researchflow.core.llm import OpenAICompatClient
from researchflow.core.schemas import ResearchQuestion

VALID_PAYLOAD = {
    "question": "变量如何影响目标变量？",
    "variables": ["x1", "y"],
    "hypothesis": "存在可量化的线性关系。",
    "method": "单算法回归。",
    "confidence": 0.6,
}


def _response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_llm_retries_on_invalid_structured_output():
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://retry.test/v1/chat/completions").mock(
            side_effect=[
                _response(json.dumps({"variables": ["x1"]})),
                _response(json.dumps(VALID_PAYLOAD)),
            ]
        )
        client = OpenAICompatClient(base_url="https://retry.test/v1", api_key="k", model="m")
        result = client.complete("prompt", ResearchQuestion, {"topic": "demo"})

        assert result.question == VALID_PAYLOAD["question"]
        assert route.call_count == 2, "应重试一次后成功"

        # 重试请求必须把上一次的校验错误摘要带回给模型。
        body = json.loads(route.calls[1].request.content)
        last_user = body["messages"][-1]["content"]
        assert "上次输出未通过结构化校验" in last_user
        assert "question" in last_user, "错误摘要应指出缺失字段"

    with respx.mock(assert_all_called=True) as exhausted:
        route = exhausted.post("https://exhaust.test/v1/chat/completions").mock(
            side_effect=[_response("bad-1"), _response("bad-2"), _response("bad-3")]
        )
        client = OpenAICompatClient(base_url="https://exhaust.test/v1", api_key="k", model="m")

        with pytest.raises(LLMError) as exc:
            client.complete("prompt", ResearchQuestion, {"topic": "demo"})

        assert route.call_count == 3, "应重试至上限后失败"
        assert exc.value.detail.get("attempts") == 3

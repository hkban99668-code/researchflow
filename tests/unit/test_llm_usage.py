"""T-001：LLM 用量记录（A1–A4、A7）。

HTTP 打桩沿用 `test_llm_retry.py` 的 respx 做法，不访问真实网络。
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from conftest import runtime_of
from researchflow.core.config import RunContext, load_settings
from researchflow.core.errors import LLMError
from researchflow.core.llm import FakeLLMClient, OpenAICompatClient, summarize_llm_usage
from researchflow.core.schemas import LLMCallRecord, LLMUsage, ParamSet, ResearchQuestion
from researchflow.nodes.extraction import _render_prompt, extraction_node

VALID_PAYLOAD = {
    "question": "变量如何影响目标变量？",
    "variables": ["x1", "y"],
    "hypothesis": "存在可量化的线性关系。",
    "method": "单算法回归。",
    "confidence": 0.6,
}


def _response(content: str, usage: dict | None = None) -> httpx.Response:
    payload: dict = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(200, json=payload)


def test_fake_client_usage_matches_character_counts():
    """A1：Fake 用量用字符数伪造，测试可直接用 len() 精确断言。"""
    client = FakeLLMClient()
    assert client.last_usage is None, "complete() 之前不应有用量"

    prompt = "请根据文献提炼一个可检验的研究问题。" * 2
    result = client.complete(
        prompt, ResearchQuestion, {"topic": "demo", "features": ["x1"], "target": "y"}
    )

    usage = client.last_usage
    assert usage is not None
    assert usage.prompt_tokens == len(prompt)
    assert usage.completion_tokens == len(result.model_dump_json())
    assert usage.total_tokens == len(prompt) + len(result.model_dump_json())
    assert usage.attempts == 1


def test_openai_client_sums_usage_across_retries():
    """A2：结构化校验失败重试时，用量累加、attempts 为真实请求次数。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://usage.test/v1/chat/completions").mock(
            side_effect=[
                _response(
                    json.dumps({"variables": ["x1"]}),
                    {"prompt_tokens": 11, "completion_tokens": 7},
                ),
                _response(
                    json.dumps(VALID_PAYLOAD),
                    {"prompt_tokens": 13, "completion_tokens": 29},
                ),
            ]
        )
        client = OpenAICompatClient(base_url="https://usage.test/v1", api_key="k", model="m")
        client.complete("prompt", ResearchQuestion, {"topic": "demo"})

        assert route.call_count == 2
        assert client.last_usage == LLMUsage(
            prompt_tokens=24, completion_tokens=36, total_tokens=60, attempts=2
        )


def test_openai_client_tolerates_missing_usage_field():
    """A3：响应缺少 usage 字段时记 0，不报错。"""
    with respx.mock(assert_all_called=True) as router:
        router.post("https://nousage.test/v1/chat/completions").mock(
            return_value=_response(json.dumps(VALID_PAYLOAD))
        )
        client = OpenAICompatClient(base_url="https://nousage.test/v1", api_key="k", model="m")
        client.complete("prompt", ResearchQuestion, {"topic": "demo"})

        assert client.last_usage == LLMUsage(prompt_tokens=0, completion_tokens=0, attempts=1)


def test_openai_client_keeps_usage_when_retries_exhausted():
    """A4：重试耗尽抛 LLMError 后，仍保留已发生请求的累计用量。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://exhaust.test/v1/chat/completions").mock(
            side_effect=[
                _response("bad-1", {"prompt_tokens": 1, "completion_tokens": 1}),
                _response("bad-2", {"prompt_tokens": 2, "completion_tokens": 2}),
                _response("bad-3", {"prompt_tokens": 3, "completion_tokens": 3}),
            ]
        )
        client = OpenAICompatClient(base_url="https://exhaust.test/v1", api_key="k", model="m")

        with pytest.raises(LLMError):
            client.complete("prompt", ResearchQuestion, {"topic": "demo"})

        assert route.call_count == 3
        assert client.last_usage == LLMUsage(
            prompt_tokens=6, completion_tokens=6, total_tokens=12, attempts=3
        )


def test_summarize_llm_usage_returns_all_zero_for_empty_calls():
    """A7：没有调用时给出全 0 结构。"""
    summary = summarize_llm_usage([])

    assert summary == {
        "calls": [],
        "by_step": {},
        "total": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "attempts": 0,
        },
    }


def test_extraction_prompt_contains_dataset_columns(project_root):
    """A4：② 渲染出的 prompt 包含数据集的全部列名。"""
    template = (project_root / "configs" / "prompts" / "extract_question.md").read_text(
        encoding="utf-8"
    )
    prompt = _render_prompt(template, "课题", [], columns=["x1", "x2", "x3", "y"])

    for column in ("x1", "x2", "x3", "y"):
        assert column in prompt
    assert "variables 只能从这些列中选择" in prompt


def _extraction_update(tmp_path, project_root, llm) -> list[LLMCallRecord]:
    """跑一次 ② 节点，返回它写入 state 的用量记录。"""
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "extraction"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(settings=settings, run_dir=run_dir, llm=llm)
    state = {
        "run_id": "extraction",
        "topic": "科研数据质量与线性回归可复现性评估",
        "param_set": ParamSet(params={"target": "y", "features": ["x1"]}),
    }
    return extraction_node(state, runtime=runtime_of(ctx))["llm_calls"]


def test_extraction_records_structured_output_mode_in_real_mode(tmp_path, project_root):
    """A13：真实模式下 ② 的记录带 `structured_output == "json_schema"`。"""
    with respx.mock(assert_all_called=True) as router:
        router.post("https://so.test/v1/chat/completions").mock(
            return_value=_response(json.dumps(VALID_PAYLOAD))
        )
        client = OpenAICompatClient(base_url="https://so.test/v1", api_key="k", model="m")

        calls = _extraction_update(tmp_path, project_root, client)

    assert client.structured_output == "json_schema"
    assert [call.structured_output for call in calls] == ["json_schema"]


def test_extraction_records_no_structured_output_in_fake_mode(tmp_path, project_root):
    """A13：Fake 不走 response_format，记录里是 None。"""
    calls = _extraction_update(tmp_path, project_root, FakeLLMClient())

    assert [call.structured_output for call in calls] == [None]


def test_summarize_llm_usage_groups_by_step_and_totals():
    """同一步骤多次调用（例如打回后重跑）累加而非覆盖。"""
    calls = [
        LLMCallRecord(
            step="extraction",
            response_model="ResearchQuestion",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=5),
        ),
        LLMCallRecord(
            step="writing",
            response_model="DraftContent",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20),
        ),
        LLMCallRecord(
            step="writing",
            response_model="DraftContent",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=2, attempts=2),
        ),
    ]

    summary = summarize_llm_usage(calls)

    assert len(summary["calls"]) == 3
    assert summary["by_step"]["extraction"] == {
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "total_tokens": 10,
        "attempts": 1,
    }
    assert summary["by_step"]["writing"] == {
        "prompt_tokens": 11,
        "completion_tokens": 22,
        "total_tokens": 33,
        "attempts": 3,
    }
    assert summary["total"] == {
        "prompt_tokens": 16,
        "completion_tokens": 27,
        "total_tokens": 43,
        "attempts": 4,
    }

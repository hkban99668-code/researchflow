"""T-004：装饰器的 StepRecord 产出（A8–A10）。

直接调用装饰后的节点函数，不跑整条图；LLM 打桩沿用 `test_llm_usage.py` 的 respx 做法。
"""

from __future__ import annotations

import httpx
import pytest
import respx

from conftest import runtime_of
from researchflow.core.config import RunContext, load_settings
from researchflow.core.errors import DataValidationError
from researchflow.core.llm import FakeLLMClient, OpenAICompatClient
from researchflow.core.schemas import LLMUsage, ParamSet, StepRecord
from researchflow.core.step import step
from researchflow.nodes.extraction import extraction_node


@pytest.fixture
def ctx(tmp_path, project_root) -> RunContext:
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    run_dir = settings.runs_dir / "unit-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(settings=settings, run_dir=run_dir, llm=FakeLLMClient())


def _extraction_state() -> dict:
    return {
        "run_id": "unit-run",
        "topic": "科研数据质量与线性回归可复现性评估",
        "param_set": ParamSet(params={"target": "y", "features": ["x1", "x2"]}),
    }


def _bad_response(usage: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": "not-json"}}], "usage": usage}
    )


def test_llm_error_records_usage_of_the_failed_call(ctx):
    """A8：LLMError 失败时，那条空 response_model 的用量等于三次响应之和。"""
    ctx.llm = OpenAICompatClient(base_url="https://fail.test/v1", api_key="k", model="m")
    runtime = runtime_of(ctx)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://fail.test/v1/chat/completions").mock(
            side_effect=[
                _bad_response({"prompt_tokens": 1, "completion_tokens": 1}),
                _bad_response({"prompt_tokens": 2, "completion_tokens": 2}),
                _bad_response({"prompt_tokens": 3, "completion_tokens": 3}),
            ]
        )
        update = extraction_node(_extraction_state(), runtime=runtime)
        assert route.call_count == 3

    expected = LLMUsage(prompt_tokens=6, completion_tokens=6, total_tokens=12, attempts=3)
    calls = update["llm_calls"]
    assert [(call.step, call.response_model) for call in calls] == [("extraction", "")]
    assert calls[0].usage == expected

    record = update["step_records"][0]
    assert record.step == "extraction"
    assert record.status == "error"
    assert record.error is not None and record.error.code == "RF-002"
    assert record.error.attempt == 1
    assert record.llm_usage == expected
    assert update["step_status"] == {"extraction": "error"}


def test_non_llm_error_does_not_double_count_usage(ctx):
    """A9：② 成功后 ③ 抛 DataValidationError，用量不会被重复计数。"""
    runtime = runtime_of(ctx)
    extraction_update = extraction_node(_extraction_state(), runtime=runtime)
    assert [call.step for call in extraction_update["llm_calls"]] == ["extraction"]
    assert ctx.llm.last_usage is not None, "前提：② 成功后 last_usage 非空"

    @step("data_prepare")
    def failing(state: dict, *, runtime) -> dict:
        raise DataValidationError("清洗后缺失率超标")

    update = failing({}, runtime=runtime)

    assert update["llm_calls"] == [], "非 LLM 异常不得把上一步的用量再记一遍"
    assert update["step_status"] == {"data_prepare": "error"}
    assert update["step_records"][0].step == "data_prepare"
    assert update["step_records"][0].llm_usage is None
    assert update["error"].code == "RF-003"


def test_attempt_counts_existing_records_of_the_same_step():
    """A10：已有两条同名记录时，新记录 attempt == 3。"""
    state = {
        "step_records": [
            StepRecord(
                step="compute", status="ok", started_at="2026-10-09T00:00:00+00:00", duration_ms=1
            ),
            StepRecord(
                step="compute",
                status="error",
                started_at="2026-10-09T00:00:01+00:00",
                duration_ms=2,
            ),
            StepRecord(
                step="chart", status="ok", started_at="2026-10-09T00:00:02+00:00", duration_ms=3
            ),
        ]
    }

    @step("compute")
    def compute(state: dict, *, runtime) -> dict:
        return {"step_status": {"compute": "ok"}}

    update = compute(state, runtime=runtime_of())

    assert update["step_records"][0].attempt == 3
    assert update["step_records"][0].duration_ms >= 0

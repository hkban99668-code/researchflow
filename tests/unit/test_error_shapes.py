"""T-010 A1、A2、A4：可路由的错误形状。

- A1：每个 ResearchFlowError 子类都显式声明 retryable / default_backtrack_to。
- A2：@step 把异常类属性填进 ErrorRecord。
- A4：owner_steps 缺一类时 load_rules 抛 ConfigError 并指明缺哪一类。
"""

from __future__ import annotations

import pytest
import yaml

from conftest import runtime_of
from researchflow.core.errors import (
    ConfigError,
    DataValidationError,
    LLMError,
    ResearchFlowError,
)
from researchflow.core.step import step
from researchflow.tools.rules_engine import SubmissionRules, load_rules

# code / retryable / default_backtrack_to（与任务卡接口设计表一致）
EXPECTED = {
    "ConfigError": ("RF-001", False, None),
    "LLMError": ("RF-002", True, None),
    "DataValidationError": ("RF-003", False, "data_prepare"),
    "ComputeError": ("RF-004", False, "compute"),
    "ArtifactError": ("RF-005", False, None),
    "ReviewRejectedError": ("RF-006", False, None),
    "RetrievalError": ("RF-007", True, "retrieval"),
}


def test_all_error_subclasses_declare_backtrack_metadata():
    """A1：子类集合与取值都和表一致；新增子类而未声明时集合断言会失败。"""
    subclasses = ResearchFlowError.__subclasses__()
    assert {cls.__name__ for cls in subclasses} == set(EXPECTED)

    for cls in subclasses:
        code, retryable, backtrack_to = EXPECTED[cls.__name__]
        assert cls.code == code
        assert cls.retryable is retryable
        assert cls.default_backtrack_to is backtrack_to


def test_error_record_filled_from_exception_class():
    """A2：DataValidationError → retryable False / backtrack_to data_prepare。"""

    @step("data_prepare")
    def failing(state: dict, *, runtime) -> dict:
        raise DataValidationError("清洗后缺失率超标")

    update = failing({}, runtime=runtime_of())
    error = update["error"]
    assert error.retryable is False
    assert error.backtrack_to == "data_prepare"


def test_llm_error_is_retryable():
    """A2：LLMError → retryable True，且没有回溯起点。"""

    @step("extraction")
    def failing(state: dict, *, runtime) -> dict:
        raise LLMError("真实 LLM 调用失败")

    update = failing({}, runtime=runtime_of())
    error = update["error"]
    assert error.retryable is True
    assert error.backtrack_to is None


def test_missing_metric_owner_raises_config_error(tmp_path, project_root):
    """A4：owner_steps 删掉 metric 一项后 load_rules 抛 ConfigError 并指明缺哪一类。"""
    source = project_root / "configs" / "rules" / "submission_checks.yaml"
    rules = yaml.safe_load(source.read_text(encoding="utf-8"))
    del rules["owner_steps"]["metric"]

    path = tmp_path / "submission_checks.yaml"
    path.write_text(yaml.safe_dump(rules, allow_unicode=True, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_rules(path, model=SubmissionRules)

    assert "metric" in exc.value.message

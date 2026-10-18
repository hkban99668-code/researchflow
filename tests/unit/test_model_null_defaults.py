"""D-006 转入项（A12）：面向模型的模型不得有"可为 None 却默认值非 None"的字段。

strict json_schema 下模型会把没话说的字段填成 null，解析时丢掉值为 None 的键。
若该字段默认值不是 None，null 会被静默改成默认值，模型因此收不到"我说的是空"的
反馈。这条约束由本测试锁住。
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import BaseModel, Field

from researchflow.core.schemas import DraftContent, ResearchQuestion
from researchflow.tools.registry_tools import build_default_registry


def _violations(model: type[BaseModel]) -> list[str]:
    """返回"类型允许 None、默认值却不是 None"的字段名。"""
    return [
        name
        for name, field in model.model_fields.items()
        if type(None) in get_args(field.annotation)
        and field.get_default(call_default_factory=True) is not None
    ]


def _assert_no_violation(model: type[BaseModel]) -> None:
    violations = _violations(model)
    assert violations == [], f"{model.__name__} 的字段 {violations} 可为 None 但默认值不是 None"


def _model_facing_models() -> list[type[BaseModel]]:
    """②⑥ 的结构化输出模型 + 注册表里每个工具的入参模型。"""
    registry = build_default_registry()
    return [
        ResearchQuestion,
        DraftContent,
        *[registry.get(name).args_model for name in registry.names()],
    ]


def test_model_facing_models_have_no_nullable_field_with_non_null_default():
    for model in _model_facing_models():
        _assert_no_violation(model)


def test_violation_message_names_model_and_field():
    """验证这条断言本身有效：违规时能报出模型名与字段名。"""

    class Violating(BaseModel):
        title: str | None = Field("默认标题")

    with pytest.raises(AssertionError) as exc:
        _assert_no_violation(Violating)

    message = str(exc.value)
    assert "Violating" in message, f"报错信息缺少模型名: {message}"
    assert "title" in message, f"报错信息缺少字段名: {message}"

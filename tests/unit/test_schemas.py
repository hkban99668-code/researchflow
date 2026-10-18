"""契约测试：Pydantic 模型、LLM 适配器与 state reducer（先红后绿）。"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel, ValidationError

from researchflow.core.errors import LLMError
from researchflow.core.llm import FakeLLMClient, OpenAICompatClient
from researchflow.core.schemas import ArtifactRef, BacktrackPatch, DraftContent, ResearchQuestion
from researchflow.core.state import append_list, merge_dict


def test_research_question_requires_question_field():
    with pytest.raises(ValidationError):
        ResearchQuestion.model_validate({})


def test_artifact_ref_defaults_are_empty_strings():
    ref = ArtifactRef(kind="chart", path="runs/demo/chart.png")

    assert ref.checksum == ""
    assert ref.produced_by == ""


def test_fake_client_returns_validated_model():
    client = FakeLLMClient()

    question = client.complete(
        "prompt",
        ResearchQuestion,
        {"topic": "demo", "target": "y", "features": ["x1"]},
    )

    assert isinstance(question, ResearchQuestion)
    assert "demo" in question.question
    assert question.variables == ["x1", "y"]


def test_fake_client_is_deterministic():
    client = FakeLLMClient()
    context = {"topic": "demo", "algorithm": "linear_regression", "metrics": {"r2": 0.9}}

    first = client.complete("prompt", DraftContent, context)
    second = client.complete("prompt", DraftContent, context)

    assert first == second


def test_fake_client_rejects_unregistered_model():
    class Unknown(BaseModel):
        value: int

    with pytest.raises(LLMError) as exc:
        FakeLLMClient().complete("prompt", Unknown)

    assert exc.value.code == "RF-002"


@respx.mock
def test_openai_compat_client_parses_json_response():
    payload = {
        "question": "q",
        "variables": ["x1", "y"],
        "hypothesis": "h",
        "method": "m",
        "confidence": 0.5,
    }
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )
    )
    client = OpenAICompatClient("https://api.openai.com/v1", "sk-test", "gpt-4o-mini")

    result = client.complete("prompt", ResearchQuestion)

    assert result.question == "q"
    assert result.variables == ["x1", "y"]


@respx.mock
def test_openai_compat_client_raises_on_schema_mismatch():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    client = OpenAICompatClient("https://api.openai.com/v1", "sk-test", "gpt-4o-mini")

    with pytest.raises(LLMError) as exc:
        client.complete("prompt", ResearchQuestion)

    assert exc.value.code == "RF-002"


def test_reducers_append_and_merge():
    assert append_list([1], [2]) == [1, 2]
    assert append_list(None, [1]) == [1]
    assert merge_dict({"a": "x"}, {"b": "y"}) == {"a": "x", "b": "y"}
    assert merge_dict({"a": "x"}, {"a": "y"}) == {"a": "y"}


def test_backtrack_patch_rejects_unknown_fields():
    """A2：`extra="forbid"` 拒绝白名单外的字段，防止把任意键写进 param_set。"""
    with pytest.raises(ValidationError):
        BacktrackPatch.model_validate({"y_outlier_strategy": "iqr", "unknown": 1})


def test_backtrack_patch_validate_for_rejects_mismatched_target():
    """A2：字段与目标节点不匹配时 `validate_for` 抛 ValueError。"""
    # y_outlier_* 只允许 data_prepare
    with pytest.raises(ValueError):
        BacktrackPatch(y_outlier_strategy="iqr").validate_for("compute")
    # target_transform / exclude_features 只允许 compute
    with pytest.raises(ValueError):
        BacktrackPatch(target_transform="log").validate_for("data_prepare")
    with pytest.raises(ValueError):
        BacktrackPatch(exclude_features=["x1"]).validate_for("data_prepare")


def test_backtrack_patch_rejects_test_size_and_seed():
    """A4：带 test_size 或 seed 的 patch 被拒绝，消息含"随机划分"。"""
    with pytest.raises(ValueError, match="随机划分"):
        BacktrackPatch(test_size=0.3).validate_for("compute")
    with pytest.raises(ValueError, match="随机划分"):
        BacktrackPatch(seed=7).validate_for("compute")
    with pytest.raises(ValueError, match="随机划分"):
        BacktrackPatch(test_size=0.3, seed=7).validate_for("data_prepare")


def test_backtrack_patch_validate_for_rejects_unknown_target():
    """A2：目标不是 data_prepare / compute 时同样拒绝。"""
    with pytest.raises(ValueError):
        BacktrackPatch().validate_for("retrieval")


def test_backtrack_patch_validate_for_accepts_matching_target():
    """A2：字段与目标匹配时校验通过。"""
    BacktrackPatch(y_outlier_strategy="iqr", y_outlier_k=3.0).validate_for("data_prepare")
    BacktrackPatch(target_transform="log").validate_for("compute")
    BacktrackPatch(exclude_features=["x2"]).validate_for("compute", features=["x1", "x2", "x3"])


def test_backtrack_patch_exclude_features_must_be_proper_subset():
    """A3：exclude_features 必须是当前特征的真子集，至少保留 1 个特征。"""
    features = ["x1", "x2", "x3"]
    # 排除不存在的列 → 报错
    with pytest.raises(ValueError, match="不存在的特征列"):
        BacktrackPatch(exclude_features=["zzz"]).validate_for("compute", features=features)
    # 排除全部 → 报错
    with pytest.raises(ValueError, match="全部"):
        BacktrackPatch(exclude_features=["x1", "x2", "x3"]).validate_for(
            "compute", features=features
        )


def test_backtrack_patch_no_op_values_are_not_counted_as_given():
    """strict 模式下模型会把无关字段填成 "none"/[]，这些不应被当作越权字段。"""
    from researchflow.core.schemas import BacktrackPatch

    patch = BacktrackPatch(
        y_outlier_strategy="iqr", y_outlier_k=1.5, target_transform="none", exclude_features=[]
    )
    patch.validate_for("data_prepare", features=["x1", "x2", "x3"])

    patch = BacktrackPatch(target_transform="log", y_outlier_strategy="none")
    patch.validate_for("compute", features=["x1", "x2", "x3"])

    import pytest

    with pytest.raises(ValueError):
        BacktrackPatch(y_outlier_strategy="iqr", target_transform="log").validate_for(
            "data_prepare", features=["x1", "x2", "x3"]
        )

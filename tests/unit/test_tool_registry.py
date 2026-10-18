"""T-002 工具注册表：导出形状、错误映射、与现有 ③④ 节点的一致性。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from langgraph.types import Command
from pydantic import BaseModel

from conftest import thread_config
from researchflow.core.config import RunContext, load_settings
from researchflow.core.errors import ConfigError
from researchflow.core.llm import FakeLLMClient
from researchflow.core.schemas import ParamSet
from researchflow.core.tools import ToolContext, ToolRegistry, ToolSpec
from researchflow.graphs.orchestrator import build_graph
from researchflow.tools.literature import build_retriever
from researchflow.tools.registry_tools import build_default_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = "examples/sample_dataset.csv"


class _NoArgs(BaseModel):
    """测试用空入参模型。"""


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry()


@pytest.fixture
def tool_ctx(project_root: Path) -> ToolContext:
    settings = load_settings(project_root=project_root)
    return ToolContext(
        workspace_root=settings.project_root,
        settings=settings,
        retriever=build_retriever(settings),
    )


@pytest.fixture(scope="module")
def gate1_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """跑一遍 ①②③④（Fake LLM，审核放行），拿 ③ 的 QualitySummary 与 ④ 的 metrics。"""
    settings = load_settings(project_root=PROJECT_ROOT)
    settings.runs_dir = tmp_path_factory.mktemp("runs")
    run_dir = settings.runs_dir / "tool-registry-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        settings=settings, run_dir=run_dir, llm=FakeLLMClient(), retriever=build_retriever(settings)
    )
    config = thread_config("t-002")
    graph = build_graph()

    graph.invoke(
        {
            "run_id": "tool-registry-run",
            "topic": "科研数据质量与线性回归可复现性评估",
            "param_set": ParamSet(
                algorithm="linear_regression",
                params={
                    "dataset_path": SAMPLE_CSV,
                    "target": "y",
                    "features": ["x1", "x2", "x3"],
                    "test_size": 0.25,
                },
                seed=42,
            ),
        },
        config,
        context=ctx,
    )
    return graph.invoke(
        Command(resume={"decision": "approve", "reason": "T-002"}), config, context=ctx
    )


# --- 辅助 ---------------------------------------------------------------------


def _walk(node: Any):
    """递归产出 schema 里的所有 dict 节点（含 $defs、anyOf 分支、数组 items）。"""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _spec(name: str, fn: Any) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="测试用工具",
        args_model=_NoArgs,
        fn=fn,
        side_effect="read",
    )


def _no_op(args: BaseModel, ctx: ToolContext) -> BaseModel:
    return _NoArgs()


def _boom(args: BaseModel, ctx: ToolContext) -> BaseModel:
    raise ValueError("不该泄漏的细节")


def _write_sample(workspace: Path) -> None:
    """把示例 CSV 与准入规则复制进临时工作区（默认 rules_path 是相对工作区的）。"""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "real.csv").write_text(
        (PROJECT_ROOT / SAMPLE_CSV).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (workspace / "rules.yaml").write_text(
        (PROJECT_ROOT / "configs" / "rules" / "data_acceptance.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


# --- A1 默认注册表 -------------------------------------------------------------


def test_default_registry_names(registry: ToolRegistry) -> None:
    assert registry.names() == ["search_literature", "check_data_quality", "run_direct_compute"]
    assert registry.get("nope") is None
    assert registry.get("search_literature") is not None


# --- A2/A3/A4 OpenAI strict 导出 -----------------------------------------------


def test_openai_parameters_pass_draft202012_schema_check(registry: ToolRegistry) -> None:
    for tool in registry.as_openai_tools():
        assert tool["type"] == "function"
        assert tool["function"]["strict"] is True
        jsonschema.Draft202012Validator.check_schema(tool["function"]["parameters"])


def test_openai_parameters_satisfy_strict_rules(registry: ToolRegistry) -> None:
    for tool in registry.as_openai_tools():
        for node in _walk(tool["function"]["parameters"]):
            assert "default" not in node, f"strict schema 不应有 default: {node}"
            assert "title" not in node, f"strict schema 不应有 title: {node}"
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False
                assert sorted(node.get("required", [])) == sorted(node["properties"])


def test_strict_schema_accepts_nulls_and_rejects_unknown_field(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    params = registry.as_openai_tools()[2]["function"]["parameters"]
    payload = {
        "csv_path": SAMPLE_CSV,
        "target": "y",
        "features": ["x1"],
        "test_size": None,
        "seed": None,
    }
    jsonschema.validate(payload, params)

    # null 在 invoke 里被当作"未提供"，落回默认值
    result = registry.invoke("run_direct_compute", payload, tool_ctx)
    assert result.ok is True, result.error
    assert result.output["parameters"]["seed"] == 42
    assert result.output["parameters"]["test_size"] == 0.25

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**payload, "unknown_field": 1}, params)


# --- A5 MCP 导出 ---------------------------------------------------------------


def test_mcp_tools_shape_and_read_only_hint(registry: ToolRegistry) -> None:
    tools = registry.as_mcp_tools()
    assert [tool["name"] for tool in tools] == registry.names()
    for tool in tools:
        assert set(tool) == {"name", "description", "inputSchema", "annotations"}
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"
    assert [tool["annotations"]["readOnlyHint"] for tool in tools] == [True, True, False]


# --- A6 错误映射 ---------------------------------------------------------------


def test_unknown_tool_returns_error(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    result = registry.invoke("nope", {}, tool_ctx)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "unknown_tool"
    assert result.error.retryable is False
    assert result.output is None


def test_invalid_args_on_bad_json(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    result = registry.invoke("run_direct_compute", "{not json", tool_ctx)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True


def test_search_literature_refs_carry_ref_id(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    """T-005 A9：工具返回的每条 ref 都有本地来源的稳定 ID。"""
    result = registry.invoke("search_literature", {"query": "数据质量 回归", "limit": 3}, tool_ctx)

    assert result.ok is True, result.error
    refs = result.output["refs"]
    assert len(refs) == 3
    for ref in refs:
        assert re.fullmatch(r"local:[0-9a-f]{12}", ref["ref_id"]), ref
        assert ref["retriever"] == "local"


def test_search_literature_uses_injected_retriever(
    registry: ToolRegistry, project_root: Path, stub_retriever
) -> None:
    """T-005 A9：`ToolContext` 注入替身时，工具用的是替身而不是本地语料。"""
    ctx = ToolContext(
        workspace_root=project_root,
        settings=load_settings(project_root=project_root),
        retriever=stub_retriever,
    )

    result = registry.invoke("search_literature", {"query": "任意查询", "limit": 6}, ctx)

    assert result.ok is True, result.error
    assert [ref["ref_id"] for ref in result.output["refs"]] == ["stub:ref-1", "stub:ref-2"]


def test_search_literature_errors_without_retriever(
    registry: ToolRegistry, project_root: Path
) -> None:
    """T-022 A7：`ctx.retriever` 为 None 时报错，不回退到本地语料。"""
    ctx = ToolContext(
        workspace_root=project_root,
        settings=load_settings(project_root=project_root),
    )

    result = registry.invoke("search_literature", {"query": "数据质量", "limit": 3}, ctx)

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "execution_error"
    assert result.error.detail["code"] == "RF-001"
    assert "没有配置检索器" in result.error.message


def test_invalid_args_on_non_object_json(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    result = registry.invoke("search_literature", "[1, 2]", tool_ctx)
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True


def test_invalid_args_reports_field_names(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    result = registry.invoke("run_direct_compute", {"target": "y"}, tool_ctx)
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert "csv_path" in result.error.message
    assert result.error.retryable is True


def test_permission_denied_for_path_outside_workspace(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    result = registry.invoke("check_data_quality", {"csv_path": "../../etc/passwd"}, tool_ctx)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "permission_denied"
    assert result.error.retryable is False


def test_execution_error_keeps_original_code(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    # 6 行数据触发 ③ 的 min_rows=30：DataValidationError（RF-003）
    result = registry.invoke(
        "check_data_quality", {"csv_path": "tests/fixtures/minimal_dataset.csv"}, tool_ctx
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "execution_error"
    assert result.error.retryable is False
    assert result.error.detail["code"] == "RF-003"


def test_internal_error_only_reports_type_name(tool_ctx: ToolContext) -> None:
    registry = ToolRegistry()
    registry.register(_spec("boom", _boom))
    result = registry.invoke("boom", {}, tool_ctx)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "internal_error"
    assert result.error.retryable is False
    assert result.error.message == "ValueError"
    assert "不该泄漏的细节" not in result.error.message


# --- A7 raw_args 形式 ----------------------------------------------------------


def test_raw_args_json_string_equals_dict(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    payload = {"query": "数据质量 回归", "limit": 3}
    from_dict = registry.invoke("search_literature", payload, tool_ctx)
    from_str = registry.invoke("search_literature", json.dumps(payload), tool_ctx)
    assert from_dict.ok is True and from_str.ok is True
    assert from_dict.output == from_str.output
    assert len(from_dict.output["refs"]) == 3


# --- A8/A9 与现有节点一致 ------------------------------------------------------


def test_check_data_quality_matches_node(
    registry: ToolRegistry, tool_ctx: ToolContext, gate1_result: dict
) -> None:
    result = registry.invoke("check_data_quality", {"csv_path": SAMPLE_CSV}, tool_ctx)
    assert result.ok is True, result.error
    assert result.output == gate1_result["cleaned_data"].quality.model_dump(mode="json")


def test_run_direct_compute_matches_node(
    registry: ToolRegistry, tool_ctx: ToolContext, gate1_result: dict
) -> None:
    result = registry.invoke(
        "run_direct_compute",
        {"csv_path": SAMPLE_CSV, "target": "y", "features": ["x1", "x2", "x3"]},
        tool_ctx,
    )
    assert result.ok is True, result.error
    assert result.output["metrics"] == gate1_result["compute_result"].metrics


# --- A10 路径边界 --------------------------------------------------------------


def test_absolute_path_outside_workspace_is_denied(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    result = registry.invoke("check_data_quality", {"csv_path": "/etc/passwd"}, tool_ctx)
    assert result.error is not None
    assert result.error.code == "permission_denied"


def test_missing_file_inside_workspace_is_retryable(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    result = registry.invoke("check_data_quality", {"csv_path": "examples/nope.csv"}, tool_ctx)
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True


def test_symlink_inside_workspace_is_accepted(
    registry: ToolRegistry, project_root: Path, tmp_path: Path
) -> None:
    _write_sample(tmp_path)
    (tmp_path / "link.csv").symlink_to(tmp_path / "real.csv")
    ctx = ToolContext(workspace_root=tmp_path, settings=load_settings(project_root=project_root))

    result = registry.invoke(
        "check_data_quality", {"csv_path": "link.csv", "rules_path": "rules.yaml"}, ctx
    )
    assert result.ok is True, result.error
    assert result.output["rows"] == 58


def test_symlink_inside_workspace_pointing_outside_is_denied(
    registry: ToolRegistry, project_root: Path, tmp_path: Path
) -> None:
    """符号链接放在工作区内，但指向工作区外：解析后的真实路径越界，必须拒。"""
    workspace = tmp_path / "ws"
    _write_sample(workspace)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text(
        (PROJECT_ROOT / SAMPLE_CSV).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (workspace / "escape.csv").symlink_to(outside / "secret.csv")
    ctx = ToolContext(workspace_root=workspace, settings=load_settings(project_root=project_root))

    result = registry.invoke(
        "check_data_quality", {"csv_path": "escape.csv", "rules_path": "rules.yaml"}, ctx
    )
    assert result.error is not None
    assert result.error.code == "permission_denied"
    assert result.error.retryable is False


def test_workspace_root_given_as_var_form_is_accepted(
    registry: ToolRegistry, project_root: Path, tmp_path: Path
) -> None:
    """macOS 上 tmp_path 在 /private/var 之下；用 /var 形式给出同一个文件不应误判。"""
    if not str(tmp_path).startswith("/private/var"):
        pytest.skip("当前平台 tmp_path 不在 /private/var 之下")
    _write_sample(tmp_path)
    var_form = Path("/var") / str(tmp_path).removeprefix("/private")
    ctx = ToolContext(workspace_root=tmp_path, settings=load_settings(project_root=project_root))

    result = registry.invoke(
        "check_data_quality",
        {"csv_path": str(var_form / "real.csv"), "rules_path": str(var_form / "rules.yaml")},
        ctx,
    )
    assert result.ok is True, result.error


# --- A11 登记校验 --------------------------------------------------------------


def test_register_rejects_duplicate_and_illegal_names() -> None:
    registry = ToolRegistry()
    registry.register(_spec("alpha", _no_op))
    with pytest.raises(ConfigError):
        registry.register(_spec("alpha", _no_op))
    with pytest.raises(ConfigError):
        registry.register(_spec("has space", _no_op))
    with pytest.raises(ConfigError):
        registry.register(_spec("x" * 65, _no_op))
    assert registry.names() == ["alpha"]


# --- T-013 工具错误清晰化 -------------------------------------------------------


def test_csv_path_directory_returns_invalid_args(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    """A1：csv_path 是目录时返回 invalid_args，消息说清"需要文件，收到目录"。"""
    result = registry.invoke("check_data_quality", {"csv_path": "examples"}, tool_ctx)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True
    assert "需要一个文件，收到的是目录" in result.error.message
    assert "examples" in result.error.message


def test_rules_path_directory_returns_invalid_args(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    """A1：rules_path 是目录时返回 invalid_args，消息说清"需要文件，收到目录"。"""
    result = registry.invoke(
        "check_data_quality",
        {"csv_path": SAMPLE_CSV, "rules_path": "configs/rules"},
        tool_ctx,
    )
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True
    assert "需要一个文件，收到的是目录" in result.error.message
    assert "configs/rules" in result.error.message


def test_csv_path_wrong_suffix_returns_invalid_args(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    """A1：csv_path 不是 CSV 文件时返回 invalid_args，消息说明需要 .csv。"""
    for bad in ("README.md", "examples/demo_topic.yaml"):
        result = registry.invoke("check_data_quality", {"csv_path": bad}, tool_ctx)
        assert result.error is not None, bad
        assert result.error.code == "invalid_args", bad
        assert result.error.retryable is True, bad
        assert "需要后缀为 .csv 的文件" in result.error.message, bad


def test_unparseable_csv_returns_invalid_args(project_root: Path, tmp_path: Path) -> None:
    """设计 #2：.csv 后缀但内容不是 CSV 时返回 invalid_args（"不是可解析的 CSV"）。"""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "bad.csv").write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
    (workspace / "configs" / "rules").mkdir(parents=True, exist_ok=True)
    (workspace / "configs" / "rules" / "data_acceptance.yaml").write_text(
        (PROJECT_ROOT / "configs" / "rules" / "data_acceptance.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    ctx = ToolContext(workspace_root=workspace, settings=load_settings(project_root=project_root))

    result = build_default_registry().invoke("check_data_quality", {"csv_path": "bad.csv"}, ctx)
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True
    assert "不是可解析的 CSV" in result.error.message


def test_unparseable_yaml_returns_invalid_args(project_root: Path, tmp_path: Path) -> None:
    """设计 #2：YAML 解析失败转 invalid_args（"不是可解析的 YAML"）。"""
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "data.csv").write_text(
        (PROJECT_ROOT / SAMPLE_CSV).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (workspace / "rules.yaml").write_text("foo: [unclosed\n", encoding="utf-8")
    ctx = ToolContext(workspace_root=workspace, settings=load_settings(project_root=project_root))

    result = build_default_registry().invoke(
        "check_data_quality", {"csv_path": "data.csv", "rules_path": "rules.yaml"}, ctx
    )
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True
    assert "不是可解析的 YAML" in result.error.message


def test_empty_rules_path_falls_back_to_default(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    """A2：rules_path="" 等价于不传，结果与默认规则路径相同。"""
    with_default = registry.invoke("check_data_quality", {"csv_path": SAMPLE_CSV}, tool_ctx)
    with_empty = registry.invoke(
        "check_data_quality", {"csv_path": SAMPLE_CSV, "rules_path": ""}, tool_ctx
    )
    assert with_default.ok is True, with_default.error
    assert with_empty.ok is True, with_empty.error
    assert with_empty.output == with_default.output


def test_blank_csv_path_returns_invalid_args(registry: ToolRegistry, tool_ctx: ToolContext) -> None:
    """设计 #1：必填 csv_path 为空/纯空白时报 invalid_args。"""
    result = registry.invoke("check_data_quality", {"csv_path": "   "}, tool_ctx)
    assert result.error is not None
    assert result.error.code == "invalid_args"
    assert result.error.retryable is True


def test_run_direct_compute_same_error_behavior(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    """A3：run_direct_compute 对同样的错误输入有同样的行为。"""
    for bad in ("examples", "README.md"):
        result = registry.invoke(
            "run_direct_compute", {"csv_path": bad, "target": "y", "features": ["x1"]}, tool_ctx
        )
        assert result.error is not None, bad
        assert result.error.code == "invalid_args", bad
        assert result.error.retryable is True, bad


def test_internal_error_detail_includes_redacted_args(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A4：包装函数抛 RuntimeError 时，internal_error 的 detail.args 带脱敏参数。"""

    def boom(args: BaseModel, ctx: ToolContext) -> BaseModel:
        raise RuntimeError("boom")

    monkeypatch.setattr("researchflow.tools.registry_tools.check_data_quality", boom)
    registry = build_default_registry()
    settings = load_settings(project_root=project_root)
    ctx = ToolContext(workspace_root=settings.project_root, settings=settings)

    abs_path = str(ctx.workspace_root / "examples" / "sample_dataset.csv")
    result = registry.invoke("check_data_quality", {"csv_path": abs_path}, ctx)

    assert result.error is not None
    assert result.error.code == "internal_error"
    assert result.error.retryable is False
    assert result.error.message == "RuntimeError"
    assert result.error.detail["args"]["csv_path"] == SAMPLE_CSV
    assert str(ctx.workspace_root) not in json.dumps(result.error.detail["args"])


def test_execution_error_redacts_workspace_root(
    registry: ToolRegistry, tool_ctx: ToolContext
) -> None:
    """A5：用 submission_checks.yaml 触发 execution_error，message/detail 无绝对路径。"""
    result = registry.invoke(
        "check_data_quality",
        {"csv_path": SAMPLE_CSV, "rules_path": "configs/rules/submission_checks.yaml"},
        tool_ctx,
    )
    assert result.error is not None
    assert result.error.code == "execution_error"
    root = str(tool_ctx.workspace_root)
    assert root not in result.error.message
    assert root not in json.dumps(result.error.detail)

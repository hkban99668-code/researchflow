"""工具注册表：一份定义，导出为 LLM tools（Chat Completions）与 MCP tools。

每个工具只写一次：Pydantic 入参模型 + 写给模型看的描述 + 实现 + 副作用标签。
注册表负责三件事：

1. 导出 —— `as_openai_tools()` 产出符合 OpenAI strict 模式的 `tools` 列表；
   `as_mcp_tools()` 产出 MCP `tools/list` 形状的列表（不引入 mcp 依赖）。
2. 调用 —— `invoke()` 收模型给的 `raw_args`（dict 或 JSON 字符串），返回 `ToolResult`。
3. 兜错 —— 所有可预期失败都变成 `ToolResult(ok=False, error=ToolError(...))`，
   不抛异常。Agent 回路里错误要以 tool message 回填给模型让它自改参数，抛出会打断回路。

本模块不接 LLM（T-003），不强制 `timeout_s`（真正的超时要子进程，留给 Gate 3 的沙箱）。
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .config import Settings
from .errors import ConfigError, ResearchFlowError

SideEffect = Literal["read", "write", "exec"]
ToolErrorCode = Literal[
    "unknown_tool",
    "invalid_args",
    "permission_denied",
    "execution_error",
    "internal_error",
    # 不是工具本身出错：回路的调用预算用完了，这一条没执行
    "budget_exhausted",
]

# OpenAI 对工具名的限制：^[a-zA-Z0-9_-]{1,64}$
TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
# 错误信息是回填给模型的，超过这个长度就截断
MAX_MESSAGE_LEN = 500

# JSON Schema 里需要递归下钻的组合关键字
_BRANCH_KEYS = ("anyOf", "oneOf", "allOf")


class ToolPathError(Exception):
    """路径越出 `ToolContext.workspace_root`：由包装函数抛出，注册表转成 `permission_denied`。

    不继承 `ResearchFlowError`，否则会被 `invoke` 先归到 `execution_error`。
    """


class ToolInputError(Exception):
    """参数本身有问题但改一改还能重试（如文件不存在）：注册表转成 `invalid_args`。

    与 `ToolPathError` 一样不继承 `ResearchFlowError`，二者区分的是"不许碰"和"再试一次"。
    """


class ToolError(BaseModel):
    """结构化错误，回填给模型。"""

    code: ToolErrorCode
    message: str  # 给模型看的简短说明，构造时截断到 500 字符
    retryable: bool  # 模型改参数后重试是否可能成功
    detail: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """一次工具调用的结果；失败也走这里，不抛异常。"""

    tool: str
    ok: bool
    output: dict[str, Any] | None = None  # 成功时为输出模型的 model_dump(mode="json")
    error: ToolError | None = None
    duration_ms: int = 0

    @model_validator(mode="after")
    def _error_matches_ok(self) -> ToolResult:
        if self.ok and self.error is not None:
            raise ValueError("ok=True 时 error 必须为 None")
        if not self.ok and self.error is None:
            raise ValueError("ok=False 时 error 必须非 None")
        return self


@dataclass(frozen=True)
class ToolContext:
    workspace_root: Path  # 所有路径参数必须解析到它之下
    settings: Settings
    # 文献检索器（`core/retrieval.py::Retriever`）；None 时 `search_literature` 报错不回退
    retriever: Any = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # 写给模型看：做什么、什么时候用、返回什么
    args_model: type[BaseModel]
    fn: Callable[[BaseModel, ToolContext], BaseModel]
    side_effect: SideEffect
    timeout_s: float = 30.0  # 本卡只登记不强制


@dataclass
class ToolRegistry:
    """按登记顺序保存工具，提供导出与调用。"""

    _specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if not TOOL_NAME_RE.match(spec.name):
            raise ConfigError(f"工具名不合法（需匹配 ^[a-zA-Z0-9_-]{{1,64}}$）: {spec.name!r}")
        if spec.name in self._specs:
            raise ConfigError(f"工具重复登记: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def as_openai_tools(self) -> list[dict[str, Any]]:
        """导出 Chat Completions `tools` 参数能直接用的列表（strict 模式）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": strict_json_schema(spec.args_model),
                    "strict": True,
                },
            }
            for spec in self._specs.values()
        ]

    def as_mcp_tools(self) -> list[dict[str, Any]]:
        """导出 MCP `tools/list` 形状的列表；MCP 不要求 strict，用 Pydantic 原生 schema。"""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.args_model.model_json_schema(),
                "annotations": {"readOnlyHint": spec.side_effect == "read"},
            }
            for spec in self._specs.values()
        ]

    def invoke(self, name: str, raw_args: dict[str, Any] | str, ctx: ToolContext) -> ToolResult:
        """调用工具；任何可预期失败都返回 `ok=False`，不抛异常。"""
        started = time.perf_counter()
        spec = self._specs.get(name)
        if spec is None:
            return _failed(name, started, "unknown_tool", f"未登记的工具: {name}", False)

        try:
            payload = _parse_raw_args(raw_args)
        except ValueError as exc:
            return _failed(name, started, "invalid_args", str(exc), True)

        try:
            args = spec.args_model.model_validate(payload)
        except ValidationError as exc:
            return _failed(name, started, "invalid_args", _validation_message(exc), True)

        # internal_error 的 detail 要带参数方便复现（T-013）；先 dump 一份 JSON 安全的。
        args_dump = args.model_dump(mode="json")

        try:
            output = spec.fn(args, ctx)
        except ToolPathError as exc:
            return _failed(name, started, "permission_denied", str(exc), False)
        except ToolInputError as exc:
            # 例如文件不存在：换个路径再试一次可能就成了
            return _failed(
                name, started, "invalid_args", _redact_root(str(exc), ctx.workspace_root), True
            )
        except ResearchFlowError as exc:
            # detail 带上原错误码（如 RF-003），方便上层归因
            return _failed(
                name,
                started,
                "execution_error",
                _redact_root(str(exc), ctx.workspace_root),
                False,
                {"code": exc.code, "detail": _redact_deep(exc.detail, ctx.workspace_root)},
            )
        except Exception as exc:  # 兜住一切：只回类型名，不带栈
            return _failed(
                name,
                started,
                "internal_error",
                type(exc).__name__,
                False,
                {"args": _redact_deep(args_dump, ctx.workspace_root, max_len=200)},
            )

        return ToolResult(
            tool=name,
            ok=True,
            output=output.model_dump(mode="json"),
            duration_ms=_elapsed_ms(started),
        )


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """把 Pydantic schema 改写成 OpenAI strict 模式能接受的形状。

    strict 模式要求：每个 object 都 `additionalProperties: false`、且 `required`
    等于全部属性名；不支持 `default`/`title`。为了"全部必填"又不逼模型编值，
    原本可选的字段改成可为 null 的类型。
    """
    schema = deepcopy(model.model_json_schema())
    _make_strict(schema)
    return schema


def _make_strict(node: dict[str, Any]) -> None:
    """原地改写一个 schema 节点（含 `$defs`）为 strict 形状。"""
    node.pop("title", None)
    node.pop("default", None)

    properties = node.get("properties")
    if isinstance(properties, dict):
        optional = {name for name in properties if name not in node.get("required", [])}
        for name, sub in properties.items():
            if not isinstance(sub, dict):
                continue
            if name in optional:
                _allow_null(sub)
            _make_strict(sub)
        node["required"] = list(properties)
        node["additionalProperties"] = False

    defs = node.get("$defs")
    if isinstance(defs, dict):
        for sub in defs.values():
            if isinstance(sub, dict):
                _make_strict(sub)

    for key in _BRANCH_KEYS:
        for sub in node.get(key) or []:
            if isinstance(sub, dict):
                _make_strict(sub)

    # items 是数组元素 schema；additionalProperties 是 dict 字段的 value schema（map），保持原样
    items = node.get("items")
    if isinstance(items, dict):
        _make_strict(items)


def _allow_null(node: dict[str, Any]) -> None:
    """把 `node` 变成可为 null 的类型，这样"全部 required"也不会逼模型编造值。"""
    if "anyOf" in node:
        branches = node["anyOf"]
        if not any(isinstance(b, dict) and b.get("type") == "null" for b in branches):
            branches.append({"type": "null"})
        return
    if "type" in node:
        current = node["type"]
        if isinstance(current, list):
            if "null" not in current:
                current.append("null")
        else:
            node["type"] = [current, "null"]
        return
    if "enum" in node and None not in node["enum"]:
        node["enum"] = [*node["enum"], None]
        return
    # 只剩 $ref 之类的引用：包一层 anyOf，不动引用本身
    node["anyOf"] = [deepcopy(node), {"type": "null"}]


def _parse_raw_args(raw_args: dict[str, Any] | str) -> dict[str, Any]:
    """把模型给的参数统一成 dict；`null` 视作"未提供"（strict 模式的可选字段）。"""
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"参数不是合法 JSON: {exc.msg}") from exc
    else:
        parsed = raw_args

    if not isinstance(parsed, dict):
        raise ValueError(f"参数必须是 JSON 对象，收到 {type(parsed).__name__}")

    return {key: value for key, value in parsed.items() if value is not None}


def _validation_message(exc: ValidationError) -> str:
    """把 Pydantic 错误压成一句带字段名的话，方便模型改参数。"""
    parts = [
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "参数校验失败: " + "; ".join(parts)


def _redact_root(text: str, root: Path) -> str:
    """把 `root` 的绝对路径前缀替换成相对路径，避免向模型泄漏本机安装位置。

    只做字符串替换：错误消息里拼进去的都是 `_resolve_within` 返回的 resolved 绝对路径，
    因此按 `root.resolve()` 的前缀（含末尾分隔符）替换即可；保留分隔符可避免
    `/root` 误伤 `/rooted` 这类前缀撞车的字符串。
    """
    root_str = str(root.resolve())
    if not root_str:
        return text
    return text.replace(root_str + os.sep, "").replace(root_str, "")


def _redact_deep(value: Any, root: Path, *, max_len: int | None = None) -> Any:
    """递归地把 `value` 里所有字符串脱敏（去根路径前缀），可选截断过长字符串。"""
    if isinstance(value, str):
        redacted = _redact_root(value, root)
        if max_len is not None and len(redacted) > max_len:
            return redacted[:max_len]
        return redacted
    if isinstance(value, dict):
        return {key: _redact_deep(val, root, max_len=max_len) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_deep(val, root, max_len=max_len) for val in value]
    return value


def _failed(
    name: str,
    started: float,
    code: ToolErrorCode,
    message: str,
    retryable: bool,
    detail: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        tool=name,
        ok=False,
        error=ToolError(
            code=code,
            message=message[:MAX_MESSAGE_LEN],
            retryable=retryable,
            detail=detail or {},
        ),
        duration_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

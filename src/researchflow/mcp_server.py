"""MCP Server（stdio）：把注册表工具与运行状态暴露给 MCP 客户端。

选型：用 SDK 低层 `Server`，不用 `MCPServer`（FastMCP v2）。低层 Server 的
`on_list_tools` / `on_call_tool` 回调直接接收 `Tool` 对象，`inputSchema` 可以
直接填 `ToolSpec.args_model.model_json_schema()`，遍历注册表就能动态生成工具，
无需为每个工具手写函数签名；`MCPServer.add_tool(fn)` 是从函数签名反推 schema，
会绕开注册表里那份现成的 schema，做不到"一份定义"。

暴露规则：`side_effect == "read"` 恒暴露；`exec`/`write` 类
（`run_direct_compute`、`start_run`）只在 `--allow-exec` 时暴露。

运行状态用 MCP resources 暴露（不是 tool）：`researchflow://runs` 给最近运行
列表，`researchflow://runs/{run_id}/manifest` 给某次运行的 manifest。
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

import anyio
from mcp import types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field

from .core.config import Settings, load_settings, sanitize_base_url
from .core.tools import ToolContext, ToolRegistry, ToolSpec
from .main import run_fake_paused
from .tools.literature import build_retriever
from .tools.registry_tools import _resolve_within, build_default_registry

RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
MANIFEST_URI_RE = re.compile(r"^researchflow://runs/([0-9a-f]{12})/manifest$")
TOPIC_SUFFIXES = (".yaml", ".yml")

RUNS_URI = "researchflow://runs"
RECENT_RUNS_LIMIT = 20
# JSON-RPC 错误码：MCP 约定 -32002 表示资源不存在，用于非法/不存在的 run_id。
ERROR_RESOURCE_NOT_FOUND = -32002


class StartRunArgs(BaseModel):
    """`start_run` 的入参：要启动的课题 YAML 路径。"""

    topic_path: str = Field(..., min_length=1, description="工作区内的课题 YAML 相对路径或绝对路径")


class StartRunOutput(BaseModel):
    """`start_run` 的输出：run_id、状态与提示。"""

    run_id: str
    status: str
    message: str


def _start_run(args: BaseModel, ctx: ToolContext) -> StartRunOutput:
    """启动一次 Fake 模式运行并停在审核处；路径校验复用 `_resolve_within`。"""
    model = cast(StartRunArgs, args)
    topic_path = _resolve_within(model.topic_path, ctx, label="topic_path", suffixes=TOPIC_SUFFIXES)
    result = run_fake_paused(topic_path, ctx.settings)
    return StartRunOutput(
        run_id=result["run_id"],
        status=result["status"],
        message=result["message"],
    )


def build_mcp_registry() -> ToolRegistry:
    """默认注册表 + `start_run`（`exec`，`--allow-exec` 才暴露）。"""
    registry = build_default_registry()
    registry.register(
        ToolSpec(
            name="start_run",
            description=(
                "以 Fake 模式启动一次完整运行，并在人工审核处停下（不自动通过审核、"
                "不调用真实模型）。什么时候用：需要走一遍科研链路并查看中间产物时。"
                "返回 run_id 与需要人工审核的提示。"
            ),
            args_model=StartRunArgs,
            fn=_start_run,
            side_effect="exec",
        )
    )
    return registry


def _exposed_specs(registry: ToolRegistry, allow_exec: bool) -> list[ToolSpec]:
    """按副作用分级过滤：read 恒暴露，exec/write 只在 allow_exec 时暴露。"""
    specs: list[ToolSpec] = []
    for name in registry.names():
        spec = registry.get(name)
        if spec is None:  # pragma: no cover - names() 只会返回已登记的
            continue
        if allow_exec or spec.side_effect == "read":
            specs.append(spec)
    return specs


def _tool_from_spec(spec: ToolSpec) -> mcp_types.Tool:
    """把注册表里的 ToolSpec 转成 MCP Tool；inputSchema 直接用 Pydantic 原生 schema。"""
    return mcp_types.Tool(
        name=spec.name,
        description=spec.description,
        input_schema=spec.args_model.model_json_schema(),
        annotations=mcp_types.ToolAnnotations(read_only_hint=(spec.side_effect == "read")),
    )


def _error_result(
    code: str, message: str, detail: dict[str, Any] | None = None
) -> mcp_types.CallToolResult:
    """构造 isError=True 的调用结果，带 code/message/detail。"""
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail:
        payload["detail"] = detail
    return mcp_types.CallToolResult(
        is_error=True,
        content=[mcp_types.TextContent(text=json.dumps(payload, ensure_ascii=False))],
    )


def _text_resource(uri: str, text: str) -> mcp_types.ReadResourceResult:
    """把 JSON 文本包成 MCP resource 读取结果。"""
    return mcp_types.ReadResourceResult(
        contents=[mcp_types.TextResourceContents(uri=uri, text=text, mime_type="application/json")]
    )


def _recent_runs(settings: Settings) -> list[dict[str, Any]]:
    """列出 runs_dir 下最近 20 个运行的 {run_id, status, created_at}，按时间倒序。"""
    runs_dir = settings.runs_dir
    if not runs_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        entries.append(
            {
                "run_id": manifest.get("run_id", child.name),
                "status": manifest.get("status", "unknown"),
                "created_at": manifest.get("created_at", ""),
            }
        )
    entries.sort(key=lambda item: item["created_at"], reverse=True)
    return entries[:RECENT_RUNS_LIMIT]


def redact_base_urls(value: Any) -> Any:
    """递归地对所有含 `://` 的字符串过 `sanitize_base_url` 脱敏。

    manifest 里不会有 API Key，但仍做一层 base_url 脱敏兜底：万一某个字段拼进了
    Provider 地址，也不会把业务空间 ID 泄漏给 MCP 客户端。
    """
    if isinstance(value, str):
        return sanitize_base_url(value) if "://" in value else value
    if isinstance(value, dict):
        return {key: redact_base_urls(val) for key, val in value.items()}
    if isinstance(value, list):
        return [redact_base_urls(val) for val in value]
    return value


def _read_manifest(settings: Settings, run_id: str) -> dict[str, Any]:
    """读某次运行的 manifest；run_id 非法或不存在时报错（不读文件）。"""
    if not RUN_ID_RE.fullmatch(run_id):
        raise MCPError(ERROR_RESOURCE_NOT_FOUND, f"非法 run_id: {run_id!r}（需要 12 位十六进制）")
    manifest_path = settings.runs_dir / run_id / "manifest.json"
    if not manifest_path.is_file():
        raise MCPError(ERROR_RESOURCE_NOT_FOUND, f"run 不存在或无 manifest: {run_id}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MCPError(ERROR_RESOURCE_NOT_FOUND, f"manifest 不可读: {run_id}") from exc
    return redact_base_urls(manifest)


def create_server(settings: Settings, allow_exec: bool) -> Server:
    """构造 stdio MCP Server：工具来自注册表，资源来自 runs 目录。"""
    registry = build_mcp_registry()
    exposed = {spec.name: spec for spec in _exposed_specs(registry, allow_exec)}

    async def on_list_tools(ctx: Any, params: Any) -> mcp_types.ListToolsResult:
        return mcp_types.ListToolsResult(tools=[_tool_from_spec(spec) for spec in exposed.values()])

    async def on_call_tool(ctx: Any, params: Any) -> mcp_types.CallToolResult:
        name = params.name
        arguments = params.arguments or {}
        if name not in exposed:
            return _error_result("unknown_tool", f"未登记或未暴露的工具: {name}")
        tool_ctx = ToolContext(
            workspace_root=settings.project_root,
            settings=settings,
            retriever=build_retriever(settings),
        )
        result = registry.invoke(name, arguments, tool_ctx)
        if result.ok:
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(text=json.dumps(result.output, ensure_ascii=False))]
            )
        error = result.error
        if error is None:  # pragma: no cover - ToolResult 校验器保证 ok=False 时 error 非空
            return _error_result("internal_error", "工具调用失败但缺少错误信息")
        return _error_result(error.code, error.message, error.detail)

    async def on_list_resources(ctx: Any, params: Any) -> mcp_types.ListResourcesResult:
        return mcp_types.ListResourcesResult(
            resources=[
                mcp_types.Resource(
                    uri=RUNS_URI,
                    name="runs",
                    description="最近的运行列表（run_id / status / created_at）",
                    mime_type="application/json",
                )
            ]
        )

    async def on_read_resource(ctx: Any, params: Any) -> mcp_types.ReadResourceResult:
        uri = params.uri
        if uri == RUNS_URI:
            return _text_resource(uri, json.dumps(_recent_runs(settings), ensure_ascii=False))
        match = MANIFEST_URI_RE.match(uri)
        if match is not None:
            return _text_resource(
                uri, json.dumps(_read_manifest(settings, match.group(1)), ensure_ascii=False)
            )
        raise MCPError(ERROR_RESOURCE_NOT_FOUND, f"未知资源: {uri}")

    return Server(
        name="researchflow",
        title="ResearchFlow",
        version="0.1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_read_resource=on_read_resource,
    )


async def run_stdio(allow_exec: bool = False) -> None:
    """以 stdio 运行 MCP Server（阻塞直到客户端断开）。"""
    settings = load_settings()
    server = create_server(settings, allow_exec)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_mcp(allow_exec: bool = False) -> None:
    """同步入口：供 `researchflow mcp` 子命令调用。"""
    anyio.run(run_stdio, allow_exec)

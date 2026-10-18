"""T-021 MCP Server（stdio）：内存传输端到端，覆盖验收标准 A1–A7。

全部测试用 `mcp.shared.memory.create_client_server_memory_streams()` 在内存里
跑一遍客户端 ↔ 服务端会话，不起子进程、不连真实模型。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams

from researchflow.core.config import Settings, load_settings
from researchflow.mcp_server import build_mcp_registry, create_server, redact_base_urls

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = "examples/sample_dataset.csv"
SAMPLE_TOPIC = "examples/demo_topic.yaml"


@pytest.fixture
def mcp_settings(tmp_path: Path) -> Settings:
    """MCP 测试用的 settings：runs_dir 指向临时目录，不污染仓库 runs/。"""
    settings = load_settings(project_root=PROJECT_ROOT)
    settings.runs_dir = tmp_path / "runs"
    return settings


async def _run_session(
    settings: Settings, allow_exec: bool, fn: Callable[[ClientSession], Any]
) -> Any:
    """在内存传输上起一次 MCP 会话，把已初始化的 ClientSession 交给 `fn`。"""
    server = create_server(settings, allow_exec)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                server.run(
                    server_streams[0],
                    server_streams[1],
                    server.create_initialization_options(),
                )
            )
            async with ClientSession(*client_streams) as client:
                await client.initialize()
                return await fn(client)


async def _list_tool_names(client: ClientSession) -> list[str]:
    result = await client.list_tools()
    return [tool.name for tool in result.tools]


# --- A1 分级暴露 ---------------------------------------------------------------


def test_list_tools_without_allow_exec(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        assert await _list_tool_names(client) == ["search_literature", "check_data_quality"]

    asyncio.run(_run_session(mcp_settings, False, _check))


def test_list_tools_with_allow_exec(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        assert await _list_tool_names(client) == [
            "search_literature",
            "check_data_quality",
            "run_direct_compute",
            "start_run",
        ]

    asyncio.run(_run_session(mcp_settings, True, _check))


# --- A2 inputSchema 与 as_mcp_tools 一致 --------------------------------------


def test_input_schema_matches_registry(mcp_settings: Settings) -> None:
    expected = {tool["name"]: tool["inputSchema"] for tool in build_mcp_registry().as_mcp_tools()}

    async def _check(client: ClientSession) -> None:
        result = await client.list_tools()
        for tool in result.tools:
            assert tool.input_schema == expected[tool.name], tool.name

    asyncio.run(_run_session(mcp_settings, True, _check))


# --- A3 调用与错误映射 ---------------------------------------------------------


def test_call_check_data_quality_ok(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        result = await client.call_tool("check_data_quality", {"csv_path": SAMPLE_CSV})
        assert result.is_error is False
        payload = json.loads(result.content[0].text)
        assert payload["rows"] == 58
        assert payload["columns"] == 4

    asyncio.run(_run_session(mcp_settings, False, _check))


def test_call_check_data_quality_directory_returns_error(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        result = await client.call_tool("check_data_quality", {"csv_path": "examples"})
        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["code"] == "invalid_args"
        assert "需要一个文件，收到的是目录" in payload["message"]

    asyncio.run(_run_session(mcp_settings, False, _check))


# --- A4 未暴露工具按未知工具处理 ----------------------------------------------


def test_unexposed_tool_returns_unknown_tool(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        result = await client.call_tool(
            "run_direct_compute",
            {"csv_path": SAMPLE_CSV, "target": "y", "features": ["x1"]},
        )
        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["code"] == "unknown_tool"
        assert "run_direct_compute" in payload["message"]

    asyncio.run(_run_session(mcp_settings, False, _check))


# --- A5 start_run 停在审核处 ---------------------------------------------------


def test_start_run_pauses_at_review(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        result = await client.call_tool("start_run", {"topic_path": SAMPLE_TOPIC})
        assert result.is_error is False
        payload = json.loads(result.content[0].text)
        run_id = payload["run_id"]
        assert payload["status"] == "awaiting_review"

        manifest = await client.read_resource(f"researchflow://runs/{run_id}/manifest")
        data = json.loads(manifest.contents[0].text)
        assert data["status"] == "awaiting_review"
        assert data["llm_mode"] == "fake"

    asyncio.run(_run_session(mcp_settings, True, _check))


# --- A6 非法 run_id 返回错误且不读文件 ----------------------------------------


def test_illegal_run_id_returns_error_without_reading(mcp_settings: Settings, monkeypatch) -> None:
    def _boom(run_id: str) -> dict[str, Any]:
        raise AssertionError("非法 run_id 不应走到读取 manifest 的逻辑")

    monkeypatch.setattr("researchflow.mcp_server._read_manifest", _boom)

    async def _check(client: ClientSession) -> None:
        with pytest.raises(MCPError):
            await client.read_resource("researchflow://runs/../x/manifest")

    asyncio.run(_run_session(mcp_settings, True, _check))


def test_nonexistent_run_returns_error(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        with pytest.raises(MCPError):
            await client.read_resource("researchflow://runs/ffffffffffff/manifest")

    asyncio.run(_run_session(mcp_settings, True, _check))


# --- A7 内存传输端到端（list_tools → call_tool → read_resource）---------------


def test_end_to_end_over_memory_transport(mcp_settings: Settings) -> None:
    async def _check(client: ClientSession) -> None:
        tools = await client.list_tools()
        assert "check_data_quality" in [tool.name for tool in tools.tools]

        result = await client.call_tool("check_data_quality", {"csv_path": SAMPLE_CSV})
        assert result.is_error is False

        runs = await client.read_resource("researchflow://runs")
        assert json.loads(runs.contents[0].text) == []

    asyncio.run(_run_session(mcp_settings, False, _check))


# --- redact_base_urls 单元 -----------------------------------------------------


def test_redact_base_urls_sanitizes_urls() -> None:
    payload = {
        "nested": {"base_url": "https://my-space.openai.com/v1/extra"},
        "list": ["https://api.deepseek.com/v1", "plain text"],
        "keep": 42,
        "none": None,
    }
    redacted = redact_base_urls(payload)
    assert redacted["nested"]["base_url"] == "https://*.openai.com"
    assert redacted["list"] == ["https://*.deepseek.com", "plain text"]
    assert redacted["keep"] == 42
    assert redacted["none"] is None

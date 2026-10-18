"""tests/api 共享工具：ASGI 客户端、fake 课题、状态等待与 SSE 读取。

单独成文件而不是塞进 conftest.py：`from conftest import ...` 会与上层
tests/conftest.py 撞名，pytest 的 sys.path 插入顺序不能依赖。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

WAIT_TIMEOUT = 60.0


@asynccontextmanager
async def client_for(app) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30
    ) as client:
        yield client


def topic_body(dataset: str = "examples/sample_dataset.csv") -> dict[str, Any]:
    """与 topic.yaml 同结构的请求体（A1 的 fake 全链路输入）。"""
    return {
        "topic": {
            "topic": "科研数据质量与线性回归可复现性评估",
            "dataset": {"path": dataset, "target": "y", "features": ["x1", "x2", "x3"]},
            "param_set": {"algorithm": "linear_regression", "seed": 42},
        }
    }


async def wait_status(
    client: httpx.AsyncClient, run_id: str, want: str | set[str], timeout: float = WAIT_TIMEOUT
) -> str:
    """轮询 GET /runs/{id} 直到状态进入 want，返回实际状态。"""
    wanted = {want} if isinstance(want, str) else set(want)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        response = await client.get(f"/runs/{run_id}")
        response.raise_for_status()
        status = response.json()["status"]
        if status in wanted:
            return status
        if status in {"ok", "terminated", "error", "interrupted"}:
            raise AssertionError(f"run {run_id} 停在 {status!r}，等不到 {wanted}")
        if loop.time() > deadline:
            raise TimeoutError(f"run {run_id} 等 {wanted} 超时，最后状态 {status!r}")
        await asyncio.sleep(0.05)


async def read_sse(
    client: httpx.AsyncClient, run_id: str, last_event_id: int | None = None
) -> list[dict[str, Any]]:
    """读一次 SSE 直到服务端关流（run_end 之后），返回解析出的消息列表。

    每条消息是 {"id": "行号", "event": 类型, "data": 原始 JSON 字符串}；
    注释行是 {"comment": ": keepalive"}。
    """
    headers = {"Last-Event-ID": str(last_event_id)} if last_event_id is not None else {}
    messages: list[dict[str, Any]] = []
    async with client.stream("GET", f"/runs/{run_id}/events", headers=headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        current: dict[str, Any] = {}
        async for line in response.aiter_lines():
            if line == "":
                if current:
                    messages.append(current)
                    current = {}
                continue
            if line.startswith(":"):
                messages.append({"comment": line})
                continue
            key, _, value = line.partition(": ")
            current[key] = value
        if current:
            messages.append(current)
    return messages


def read_events_file(settings, run_id: str) -> list[dict[str, Any]]:
    """读 run 的 events.jsonl 全部完整行（对照 SSE 的基准数据）。"""
    path = settings.runs_dir / run_id / "events.jsonl"
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()]


async def next_message(generator: Any, timeout: float = 5.0) -> str:
    """带超时地取 SSE 生成器的下一条消息（直连 `iter_events` 的测试用）。"""
    return await asyncio.wait_for(generator.__anext__(), timeout)

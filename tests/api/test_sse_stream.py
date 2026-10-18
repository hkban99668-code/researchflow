"""SSE 验收：A2（重放与 Last-Event-ID）、A3（interrupt 挂起与断开清理）、A4（半行）。"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from api.helpers import client_for, next_message, read_sse, topic_body, wait_status
from researchflow.api.sse import EventStreamCounters, iter_events
from researchflow.core.events import EventLog

pytestmark = pytest.mark.anyio


# --- A2：事件序列与行号 -------------------------------------------------------


async def test_sse_sequence_matches_events_file_and_reconnect(api_env, api_settings):
    async with client_for(api_env) as client:
        run_id = (await client.post("/runs", json=topic_body())).json()["run_id"]
        await wait_status(client, run_id, "awaiting_review")
        await client.post(f"/runs/{run_id}/resume", json={"decision": "approve"})
        await wait_status(client, run_id, "ok")

        messages = await read_sse(client, run_id)
        # 序列、id、data 与 events.jsonl 完全一致
        file_events = [
            raw
            for raw in (api_settings.runs_dir / run_id / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [message["id"] for message in messages] == [
            str(i) for i in range(1, len(file_events) + 1)
        ]
        assert [message["data"] for message in messages] == file_events
        assert [json.loads(raw)["type"] for raw in file_events] == [
            message["event"] for message in messages
        ]
        assert messages[-1]["event"] == "run_end"

        # A2：Last-Event-ID 重连，从 k+1 继续，不重复不遗漏
        k = 3
        again = await read_sse(client, run_id, last_event_id=k)
        assert [message["id"] for message in again] == [
            str(i) for i in range(k + 1, len(file_events) + 1)
        ]
        assert [message["data"] for message in again] == file_events[k:]


# --- A3：interrupt 后不关流，resume 后继续，run_end 后关闭 -----------------------


async def test_stream_stays_open_at_interrupt_then_closes_after_run_end(api_env):
    app = api_env
    async with client_for(app) as client:
        run_id = (await client.post("/runs", json=topic_body())).json()["run_id"]
        await wait_status(client, run_id, "awaiting_review")

        reader = asyncio.create_task(read_sse(client, run_id))
        await asyncio.sleep(0.3)  # 重放完 interrupt，连接保持打开
        assert app.state.counters.active == 1

        await client.post(f"/runs/{run_id}/resume", json={"decision": "approve"})
        messages = await asyncio.wait_for(reader, timeout=60)

    types = [message["event"] for message in messages]
    assert "interrupt" in types and "resume" in types and types[-1] == "run_end"
    await asyncio.sleep(0.2)
    assert app.state.counters.active == 0  # 服务端关流后计数归零，无悬挂协程


async def test_client_disconnect_leaves_no_hanging_task(tmp_path):
    """客户端断开 = Starlette 取消生成器：取消后计数必须归零（A3 的任务计数断言）。

    httpx 的 ASGITransport 不模拟"读到一半断开"（要等 app 跑完才收响应体），
    所以在 iter_events 这一层直接模拟取消——端点挂的就是同一个生成器。
    """
    run_dir = tmp_path / "run-cancel"
    run_dir.mkdir(parents=True)
    EventLog(run_dir / "events.jsonl", "b" * 12).emit("run_start", detail={})
    counters = EventStreamCounters()
    generator = iter_events(run_dir, 0, counters=counters, poll_interval=0.01, keepalive=15.0)

    first = await next_message(generator)  # 重放 run_start
    assert "run_start" in first
    following = asyncio.create_task(generator.__anext__())  # 进入跟随阶段（不会结束）
    await asyncio.sleep(0.1)
    assert counters.active == 1
    following.cancel()  # 模拟客户端断开
    with pytest.raises(asyncio.CancelledError):
        await following
    await asyncio.sleep(0.1)
    assert counters.active == 0  # finally 释放了计数，无悬挂协程
    await generator.aclose()


# --- A4：半行不推送 ------------------------------------------------------------


async def _pump_into(generator, sink: list) -> None:
    """把生成器的消息持续搬进 sink；生成器跑完（run_end）或被取消时返回。"""
    try:
        async for message in generator:
            sink.append(message)
    except asyncio.CancelledError:
        raise


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("等 SSE 消息超时")
        await asyncio.sleep(0.01)


async def test_partial_line_not_pushed_until_complete(tmp_path):
    run_dir = tmp_path / "run-half"
    run_dir.mkdir(parents=True)
    log = EventLog(run_dir / "events.jsonl", "a" * 12)
    log.emit("run_start", detail={})

    sink: list[str] = []
    generator = iter_events(run_dir, 0, poll_interval=0.01, keepalive=15.0)
    pump = asyncio.create_task(_pump_into(generator, sink))
    try:
        await _wait_until(lambda: len(sink) >= 1)
        assert sink[0].startswith("id: 1") and "run_start" in sink[0]

        # 写半行（无换行）：轮询多轮也不推送
        path = run_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"ts": "x", "run_id": "aa')
        await asyncio.sleep(0.2)
        assert len(sink) == 1

        # 补全：只推送一次，id 是行号 2；补全行是 run_end，生成器随之关闭
        with path.open("a", encoding="utf-8") as handle:
            handle.write('aa", "type": "run_end", "detail": {}}\n')
        await _wait_until(lambda: len(sink) >= 2)
        assert sink[1].startswith("id: 2") and '"run_end"' in sink[1]
        await asyncio.sleep(0.2)
        assert len(sink) == 2  # 没有第二次推送
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump


# --- keepalive 与计数器（直连生成器，不真等 15 秒） -------------------------------


async def test_keepalive_emitted_while_idle_and_counter_bookkeeping(tmp_path):
    run_dir = tmp_path / "run-idle"
    run_dir.mkdir(parents=True)
    counters = EventStreamCounters()
    generator = iter_events(run_dir, 0, counters=counters, poll_interval=0.01, keepalive=0.05)

    sink: list[str] = []
    pump = asyncio.create_task(_pump_into(generator, sink))
    try:
        await _wait_until(lambda: len(sink) >= 1)  # 生成器体启动后计数才生效
        assert counters.active == 1
        assert sink[0].startswith(": keepalive")
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump
    await asyncio.sleep(0.1)
    assert counters.active == 0

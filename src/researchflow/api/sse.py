"""SSE：把 `runs/<id>/events.jsonl` 按行号推给客户端（先重放，再跟随）。

为什么数据源是文件而不是内存队列：这份流是 T-004 就落盘的同一份 `events.jsonl`，
进程重启、客户端断线重连都能续上。`id:` 用事件在文件里的行号，客户端带
`Last-Event-ID` 重连时从下一行继续，不重复不遗漏。

半行问题：`EventLog.emit` 是"追加写 + 立即关闭"，读取方可能读到只写了一半的行。
这里只推送**以换行结尾**的完整行，半截行留到下一次轮询读到完整行时再推一次。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

# 轮询间隔与 keepalive：都写在这里，测试可以覆盖成更小的值。
POLL_INTERVAL_SECONDS = 0.1
KEEPALIVE_SECONDS = 15.0


class EventStreamCounters:
    """统计正在跟随事件流的 SSE 连接数。

    只在测试里断言"客户端断开后归零"用；生产路径上它只是一个线程安全的计数器。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1

    def leave(self) -> None:
        with self._lock:
            self._active -= 1

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


def read_complete_lines(path: Path, start_line: int) -> list[tuple[int, str]]:
    """读第 `start_line` 行之后、以换行结尾的完整行，返回 `[(行号, 行内容)]`。

    文件不存在（run 刚提交、run_start 还没落盘）时返回空列表。
    """
    if not path.exists():
        return []
    complete: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if line_no <= start_line:
                continue
            if not raw.endswith("\n"):
                break
            complete.append((line_no, raw.rstrip("\n")))
    return complete


def _event_type(raw: str) -> str:
    """取事件类型；解析不了就退化成 SSE 默认的 `message`。"""
    try:
        value = json.loads(raw).get("type")
    except json.JSONDecodeError:
        return "message"
    return value if isinstance(value, str) else "message"


def format_event(line_no: int, raw: str) -> str:
    """一条 SSE 消息：`id: 行号` / `event: 事件类型` / `data: 事件 JSON 原样`。"""
    return f"id: {line_no}\nevent: {_event_type(raw)}\ndata: {raw}\n\n"


async def iter_events(
    run_dir: Path,
    start_line: int = 0,
    *,
    counters: EventStreamCounters | None = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    keepalive: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """异步生成 SSE 消息：先重放 `start_line` 之后的事件，再跟随新增，直到 `run_end`。

    - 停在 `interrupt` 后**不关闭**连接（客户端通常正在等审核），继续跟随 resume 之后
      的事件；空闲每 `keepalive` 秒发一条 `: keepalive` 注释行。
    - 读文件走 `asyncio.to_thread`，不阻塞事件循环。
    - 客户端断开时 Starlette 会取消本生成器，`finally` 里释放计数，不留悬挂协程。
    """
    if counters is not None:
        counters.enter()
    try:
        path = run_dir / "events.jsonl"
        line_no = start_line
        last_emit = time.monotonic()
        while True:
            lines = await asyncio.to_thread(read_complete_lines, path, line_no)
            for index, raw in lines:
                line_no = index
                yield format_event(index, raw)
                last_emit = time.monotonic()
                if _event_type(raw) == "run_end":
                    return
            if not lines and time.monotonic() - last_emit >= keepalive:
                yield ": keepalive\n\n"
                last_emit = time.monotonic()
            await asyncio.sleep(poll_interval)
    finally:
        if counters is not None:
            counters.leave()

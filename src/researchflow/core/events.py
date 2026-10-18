"""运行事件日志：把一次运行的过程写成 `runs/<id>/events.jsonl`，一行一个事件。

事件由 CLI 在消费图输出流（`stream_mode="updates"`）时写出，节点内部不写文件：
③ 带 `interrupt()`，恢复时 LangGraph 会把整个节点从头再跑一遍，写在装饰器里的
事件就会重复；而 update 只在节点**完成**时产出，被中断的那一半不产出，天然不重复。
这个流也是 Gate 3 SSE 推送要用的同一个流。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .schemas import ErrorRecord, LLMUsage, StepRecord

EventType = Literal["run_start", "step_end", "interrupt", "resume", "run_end"]


class RunEvent(BaseModel):
    """一行事件。

    `step` 及其后的字段只对 `step_end` 有意义；`run_end` 用 `status` 与
    `detail["termination"]["kind"]` 表达收尾，`error` 只在失败时出现。
    """

    # ISO 8601，毫秒精度
    ts: str
    run_id: str
    type: EventType
    step: str | None = None
    status: str | None = None
    attempt: int | None = None
    duration_ms: int | None = None
    error: ErrorRecord | None = None
    llm_usage: LLMUsage | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


def _now_ms() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class EventLog:
    """追加写 `events.jsonl`：每次 `emit` 落盘一行并关闭文件（即 flush）。"""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id

    def emit(self, type: EventType, **fields: Any) -> RunEvent:
        event = RunEvent(ts=_now_ms(), run_id=self.run_id, type=type, **fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        return event

    def from_step_record(self, record: StepRecord) -> RunEvent:
        """把 StepRecord 原样搬成 step_end 事件，不在调用方重新计时。"""
        return self.emit(
            "step_end",
            step=record.step,
            status=record.status,
            attempt=record.attempt,
            duration_ms=record.duration_ms,
            error=record.error,
            llm_usage=record.llm_usage,
        )

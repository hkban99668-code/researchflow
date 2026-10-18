"""应用级运行管理器：线程池 + `run_id → Future` 表 + 从磁盘推导 run 状态。

线程模型（任务卡定）：模块级一个 `ThreadPoolExecutor`，`run_id → Future` 一张表；
超过上限的新 run 排队（状态仍是 `running`）。图与节点保持同步，线程里跑的是
`researchflow.main.drive_run`。

状态**不存内存**：由 `events.jsonl` 最后一个事件与线程表共同决定——
`run_end` 看其 status；最后是 `interrupt` 为 `awaiting_review`；否则在本进程线程表里
是 `running`、不在是 `interrupted`（进程在中途被杀）。
"""

from __future__ import annotations

import json
import re
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..core.config import Settings
from ..graphs.orchestrator import build_graph, close_checkpointer, make_checkpointer
from .errors import NOT_FOUND, ApiError

# run_id 必须是 12 位十六进制：既是路径名也是防目录穿越的第一道闸。
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
RECENT_RUNS_LIMIT = 20
DEFAULT_MAX_WORKERS = 4
# 已经跑完（有 run_end）的三种 status。
FINAL_STATUSES = frozenset({"ok", "terminated", "error"})


def validate_run_id(run_id: str) -> None:
    """`run_id` 非法一律 404，且**不碰文件系统**（`../`、`..%2f` 之类进不来）。"""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ApiError(
            404,
            NOT_FOUND,
            f"非法 run_id: {run_id!r}（需要 12 位十六进制）；"
            "run_id 从 POST /runs 或 GET /runs 拿。",
        )


def _runs_under(path: Path, root: Path) -> bool:
    """`path` 是否真的在 `root` 下（防 `..` 拼出来的路径）。"""
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def last_event(events_path: Path) -> dict[str, Any] | None:
    """`events.jsonl` 里最后一条**完整**事件；文件不存在或没有完整行时返回 None。"""
    if not events_path.is_file():
        return None
    last: dict[str, Any] | None = None
    with events_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.endswith("\n"):
                break  # 半行：写入方还没写完，等下一次
            try:
                last = json.loads(raw)
            except json.JSONDecodeError:
                continue
    return last


def read_manifest(path: Path) -> dict[str, Any] | None:
    """读 manifest.json；不存在或不可读返回 None。"""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class RunManager:
    """一个 API 进程一份：持有线程池与线程表，所有 run 状态从磁盘推。"""

    def __init__(self, settings: Settings, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        self.settings = settings
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rf-run")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    # --- 线程表 ---------------------------------------------------------------

    def submit(self, run_id: str, work: Callable[[], None]) -> bool:
        """原子地占住 `run_id`：该 run 已有未完成的线程时返回 False（调用方回 409）。

        检查与登记在同一把锁里完成，并发 resume 只有一个能拿到 True。
        """
        with self._lock:
            existing = self._futures.get(run_id)
            if existing is not None and not existing.done():
                return False
            self._futures[run_id] = self._executor.submit(self._guard, work)
            return True

    @staticmethod
    def _guard(work: Callable[[], None]) -> None:
        """线程体兜底：正常情况下 `work` 自己把异常落成 `run_end`。

        真走到这里说明连落盘都失败了，至少把栈打出来，不要静默死掉。
        """
        try:
            work()
        except Exception:  # noqa: BLE001 - 线程池不能把异常吞成静默
            traceback.print_exc()

    def is_active(self, run_id: str) -> bool:
        """该 run 在本进程的线程表里是否还有未完成的线程。"""
        with self._lock:
            future = self._futures.get(run_id)
            return future is not None and not future.done()

    def active_count(self) -> int:
        """未完成的线程数（测试用）。"""
        with self._lock:
            return sum(1 for future in self._futures.values() if not future.done())

    def shutdown(self) -> None:
        """等运行中的线程跑完并关闭线程池（应用关闭时调用）。"""
        self._executor.shutdown(wait=True, cancel_futures=False)

    # --- 路径与状态 -----------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        """run 目录。调用前必须先 `validate_run_id`。"""
        return self.settings.runs_dir / run_id

    def exists(self, run_id: str) -> bool:
        """run 目录存在，且确实落在 runs_dir 下。"""
        run_dir = self.run_dir(run_id)
        return run_dir.is_dir() and _runs_under(run_dir, self.settings.runs_dir)

    def status_of(self, run_id: str) -> str | None:
        """从磁盘与线程表推导状态；run 不存在返回 None。

        `run_end` → 它记录的 status；`interrupt` → `awaiting_review`；其他（含还没有
        事件）→ 在本进程线程表里是 `running`，不在是 `interrupted`（进程被杀）。
        """
        run_dir = self.run_dir(run_id)
        if not self.exists(run_id):
            return "running" if self.is_active(run_id) else None
        event = last_event(run_dir / "events.jsonl")
        if event is None:
            return "running" if self.is_active(run_id) else "interrupted"
        event_type = event.get("type")
        if event_type == "run_end":
            status = event.get("status")
            return status if isinstance(status, str) and status else "ok"
        if event_type == "interrupt":
            return "awaiting_review"
        return "running" if self.is_active(run_id) else "interrupted"

    # --- 读盘 ---------------------------------------------------------------

    def manifest(self, run_id: str) -> dict[str, Any] | None:
        return read_manifest(self.run_dir(run_id) / "manifest.json")

    def pending_review(self, run_id: str) -> dict[str, Any] | None:
        """读 checkpoint 里挂起的审核载荷（与 CLI 打印的是同一份信息）。

        只读 `graph.get_state()`，不建 LLM 客户端、不驱动图，所以恢复真实 run 也不
        会发出真实请求。
        """
        checkpointer = make_checkpointer(self.settings.checkpoint.backend, self.run_dir(run_id))
        try:
            graph = build_graph(checkpointer)
            pending = graph.get_state({"configurable": {"thread_id": run_id}})
            if not pending.interrupts:
                return None
            value = pending.interrupts[0].value
            return value if isinstance(value, dict) else {"decision": str(value)}
        finally:
            close_checkpointer(checkpointer)

    def recent_runs(self) -> list[dict[str, Any]]:
        """最近 20 个运行：`{run_id, status, topic, started_at}`，按开始时间倒序。

        沿用 MCP `_recent_runs` 的扫目录逻辑，但**包含进行中的 run**：只看 manifest
        会把"刚 POST 完、还没跑完"的 run 漏掉。没有 manifest 时用 `run_meta.json`
        判存在、用 run_start 事件取 topic。
        """
        runs_dir = self.settings.runs_dir
        if not runs_dir.is_dir():
            return []
        entries: list[dict[str, Any]] = []
        for child in runs_dir.iterdir():
            if not child.is_dir() or not RUN_ID_RE.fullmatch(child.name):
                continue
            manifest = read_manifest(child / "manifest.json")
            meta = read_manifest(child / "run_meta.json")
            if manifest is None and meta is None:
                continue
            if manifest is not None:
                status = manifest.get("status") or self.status_of(child.name) or "unknown"
                topic = manifest.get("topic")
                started_at = manifest.get("created_at", "")
            else:
                status = self.status_of(child.name) or "unknown"
                topic = self._topic_from_events(child)
                started_at = (meta or {}).get("created_at", "")
            entries.append(
                {
                    "run_id": child.name,
                    "status": status,
                    "topic": topic,
                    "started_at": started_at,
                }
            )
        entries.sort(key=lambda item: item["started_at"], reverse=True)
        return entries[:RECENT_RUNS_LIMIT]

    @staticmethod
    def _topic_from_events(run_dir: Path) -> Any:
        """进行中的 run 没有 manifest，topic 从第一条 run_start 事件里取。"""
        events_path = run_dir / "events.jsonl"
        if not events_path.is_file():
            return None
        with events_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.endswith("\n"):
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "run_start":
                    return (event.get("detail") or {}).get("topic")
        return None

"""应用工厂与路由（T-026）：HTTP 发起运行、SSE 看事件、API 形态的人工审核。

端点形状由任务卡定，本文件只做三件事：

1. 把 HTTP 请求翻译成对 `researchflow.main` 运行核心的调用（线程池跑同步图）；
2. run 状态一律从磁盘推（`RunManager`），进程重启后 GET/resume 照常可用；
3. 统一错误体与安全闸：run_id 白名单校验、`real` 模式默认拒绝、响应过 base_url 脱敏。

`serve` 子命令（`main.serve`）负责真正起 uvicorn；测试直接用 ASGI 传输打本应用。
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.types import Command

from ..core.config import Settings, load_settings
from ..core.errors import ConfigError
from ..main import (
    build_run_context,
    build_run_context_from_run_id,
    drive_run,
    initial_state,
    open_run_session,
    record_run_start,
    run_settings,
)
from ..mcp_server import redact_base_urls
from . import sse as sse_module
from .errors import CONFLICT, NOT_FOUND, REAL_NOT_ALLOWED, ApiError, api_error_handler
from .manager import DEFAULT_MAX_WORKERS, FINAL_STATUSES, RunManager, validate_run_id
from .schemas import ResumeRequest, StartRunRequest
from .sse import EventStreamCounters, iter_events

# 环境变量闸门：`mode: "real"` 会调真实模型花钱，默认拒绝（任务卡 A7）。
ALLOW_REAL_ENV = "RF_API_ALLOW_REAL"


def _as_int_header(raw: str | None) -> int:
    """解析 `Last-Event-ID`；缺失返回 0，非数字回 400（不能悄悄从头发一遍造成重复推送）。"""
    if raw is None or raw == "":
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ApiError(
            400,
            "invalid_last_event_id",
            f"Last-Event-ID 必须是事件行号（十进制整数），收到: {raw!r}；"
            "不带该头则从第一个事件开始。",
        ) from exc
    return max(value, 0)


def create_app(
    settings: Settings | None = None, *, max_workers: int = DEFAULT_MAX_WORKERS
) -> FastAPI:
    """构造 FastAPI 应用：一个 RunManager（线程池）+ 一个 SSE 连接计数器。

    `settings` 缺省时 `load_settings()`。测试传 `tmp_path` 下的私有 settings 即可
    与真实 runs/ 隔离。
    """
    app_settings = settings if settings is not None else load_settings()
    manager = RunManager(app_settings, max_workers=max_workers)
    counters = EventStreamCounters()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # 等运行中的线程收尾（线程里自会把异常落成 run_end），再关线程池。
        manager.shutdown()

    app = FastAPI(title="ResearchFlow", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.counters = counters
    app.add_exception_handler(ApiError, api_error_handler)

    def _require_run(run_id: str) -> Path:
        """run_id 合法且存在，返回 run 目录；否则 404（不碰文件系统外的路径）。"""
        validate_run_id(run_id)
        if not manager.exists(run_id):
            raise ApiError(
                404,
                NOT_FOUND,
                f"run 不存在: {run_id}；run_id 从 POST /runs 的响应或 GET /runs 列表里拿。",
            )
        return manager.run_dir(run_id)

    # --- 运行线程体 -----------------------------------------------------------

    def _start_work(
        run_id: str, run_dir: Path, spec: dict[str, Any], state: dict, mode: str
    ) -> None:
        """首跑线程：私有 settings → 上下文 → 图 → 驱动到审核处或结束。

        `drive_run(catch_unexpected=True)`：任何异常都落成 `run_end(status=error)`
        并写 manifest，线程不能静默死掉让 run 永远显示 running（A9）。
        """
        own = run_settings(app_settings, mode)
        ctx = build_run_context(own, run_dir)
        session = open_run_session(own, run_id, run_dir, ctx)
        record_run_start(session, topic_path=run_dir / "topic.yaml", spec=spec)
        drive_run(
            session,
            state,
            pause_at_review=True,
            print_review=False,
            catch_unexpected=True,
            fallback_topic=spec.get("topic"),
        )

    def _resume_work(run_id: str, decision: dict[str, Any]) -> None:
        """恢复线程：与 CLI resume 同一机制（`from_run_id` 读 run_meta.json 重建上下文）。"""
        own = run_settings(app_settings)  # 私有副本：from_run_id 会写 settings.llm.mode
        ctx = build_run_context_from_run_id(run_id, own)
        session = open_run_session(own, run_id, ctx.run_dir, ctx)
        session.log.emit("resume", detail={"decision": decision.get("decision")})
        drive_run(
            session,
            Command(resume=decision),
            decision=decision,
            print_review=False,
            catch_unexpected=True,
        )

    # --- 端点 -----------------------------------------------------------------

    @app.post("/runs", status_code=202)
    def start_run(body: StartRunRequest) -> dict[str, str]:
        if body.mode == "real" and os.getenv(ALLOW_REAL_ENV) != "1":
            raise ApiError(
                403,
                REAL_NOT_ALLOWED,
                "mode=real 会调用真实模型并产生费用，默认拒绝；确认预算后设置环境变量 "
                f"{ALLOW_REAL_ENV}=1 再试，或改用 mode=fake。",
            )
        spec = body.topic.to_spec()
        run_id = uuid.uuid4().hex[:12]
        try:
            # 先在请求线程里把 spec 校验成初始 state（数据集三要素缺一即失败），
            # 失败就不建 run 目录，也不占线程池。
            state = initial_state(run_id, spec, app_settings.seed)
        except ConfigError as exc:
            raise ApiError(
                400,
                "invalid_topic",
                f"topic 不完整: {exc}；dataset 必须提供 path、target 与 features。",
            ) from exc

        run_dir = manager.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        # API 没有 topic.yaml 文件，把请求体里的 spec 落成 run 目录内的一份，
        # run_meta.json 的 topic_path 与跨进程恢复因此和 CLI 完全同构。
        (run_dir / "topic.yaml").write_text(
            yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        manager.submit(run_id, lambda: _start_work(run_id, run_dir, spec, state, body.mode))
        return {"run_id": run_id, "status": "running"}

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return redact_base_urls(manager.recent_runs())

    @app.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        _require_run(run_id)
        status = manager.status_of(run_id)
        body: dict[str, Any] = {"run_id": run_id, "status": status}
        if status == "awaiting_review":
            # 与 CLI 审核提示同一份载荷（graph.get_state 的 interrupt value）。
            body["review"] = manager.pending_review(run_id)
        elif status in FINAL_STATUSES:
            manifest = manager.manifest(run_id) or {}
            validation = manifest.get("validation") or {}
            trace = manifest.get("trace") or {}
            body["summary"] = {
                "status": manifest.get("status", status),
                "termination": manifest.get("termination"),
                "validation": {
                    "passed": validation.get("passed"),
                    "summary": validation.get("summary"),
                },
                "trace": {
                    "total_tokens": trace.get("total_tokens"),
                    # T-027 起 total_cost 是按币种分组的字典；旧 manifest（数字 +
                    # 字符串 currency）原样透传，currency 对新 manifest 为 None。
                    "total_cost": trace.get("total_cost"),
                    "unpriced_calls": trace.get("unpriced_calls"),
                    "currency": trace.get("currency"),
                },
            }
        return redact_base_urls(body)

    @app.post("/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str, body: ResumeRequest) -> dict[str, str]:
        _require_run(run_id)
        status = manager.status_of(run_id)
        if status != "awaiting_review":
            raise ApiError(
                409,
                CONFLICT,
                f"run 当前状态是 {status!r}，只有 awaiting_review 能 resume；"
                f"先 GET /runs/{run_id} 确认状态，审核中的 run 等 interrupt 事件出现后再试。",
            )
        decision: dict[str, Any] = {"decision": body.decision, "reason": body.reason}
        # submit 原子占位：并发两次 resume 只有一个拿到 True，另一个 409（A6）。
        if not manager.submit(run_id, lambda: _resume_work(run_id, decision)):
            raise ApiError(
                409,
                CONFLICT,
                f"run {run_id} 已有线程在运行（可能刚被 resume 过）；"
                "等它结束（GET /runs/{id} 离开 running）再做下一步。",
            )
        return {"run_id": run_id, "status": "running"}

    @app.get("/runs/{run_id}/events")
    def events_stream(
        run_id: str, last_event_id: Annotated[str | None, Header()] = None
    ) -> Response:
        run_dir = _require_run(run_id)
        start = _as_int_header(last_event_id)
        return StreamingResponse(
            iter_events(
                run_dir,
                start,
                counters=counters,
                poll_interval=sse_module.POLL_INTERVAL_SECONDS,
                keepalive=sse_module.KEEPALIVE_SECONDS,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/runs/{run_id}/trace")
    def run_trace(run_id: str) -> JSONResponse:
        run_dir = _require_run(run_id)
        if manager.status_of(run_id) not in FINAL_STATUSES:
            raise ApiError(
                404,
                NOT_FOUND,
                f"run {run_id} 还没结束（trace.json 只在收尾时写）；"
                "订阅 GET /runs/{id}/events 等 run_end 后再取。",
            )
        trace_path = run_dir / "trace.json"
        if not trace_path.is_file():  # pragma: no cover - 终态必有 trace，防御目录被动过
            raise ApiError(404, NOT_FOUND, f"run {run_id} 缺少 trace.json")
        return JSONResponse(redact_base_urls(json.loads(trace_path.read_text(encoding="utf-8"))))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app

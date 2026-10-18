"""CLI 入口，同时提供 CLI / MCP / FastAPI 共用的运行核心。

生成 run_id、驱动 Gate 1 链路、处理人工审核并写 manifest。单次运行与跨进程恢复
共用同一套驱动逻辑：图只描述结构，checkpointer 落在
`runs/<run_id>/checkpoint.sqlite`，运行上下文每次调用经 `context=` 传入，
所以 `resume` 只要 run_id 和磁盘文件就能接着跑（P1-1）。

服务化（T-026）后三个入口共用 `build_run_context` / `open_run_session` /
`record_run_start` / `drive_run`：CLI 负责打印与退出码，MCP 与 FastAPI 只用
`drive_run` 返回的 `RunOutcome`，不再各自复制一遍"建图 → 驱动 → 收尾"。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langgraph.types import Command
from pydantic import BaseModel

from .core.config import (
    RunContext,
    Settings,
    load_settings,
    relative_to_root,
    run_meta_payload,
    write_run_meta,
)
from .core.errors import ConfigError, ResearchFlowError
from .core.events import EventLog
from .core.llm import build_llm_client, summarize_llm_usage
from .core.schemas import (
    ArtifactRef,
    AttributionRecord,
    ErrorRecord,
    LLMCallRecord,
    ParamSet,
    StepRecord,
    Termination,
)
from .core.tracing import (
    build_trace,
    export_trace_otlp,
    load_pricing,
    render_trace_tree,
    trace_summary,
)
from .graphs.orchestrator import build_graph, close_checkpointer, make_checkpointer
from .tools.literature import build_retriever

# `trace --run-id` 与 resume 的 run_id 校验一致：12 位十六进制。
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")

EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_REVIEW_REJECTED = 2
EXIT_ERROR = 3
# 停在审核处并退出进程，等 `researchflow resume` 在新进程里继续
EXIT_AWAITING_REVIEW = 4
# 归因 Agent 判定无法自动修复，转人工
EXIT_ESCALATED = 5

TERMINATION_LABELS = {
    "rejected": "人工审核拒绝",
    "empty": "检索结果为空",
    "error": "链路错误",
    "budget_exhausted": "回溯预算耗尽",
    "escalated": "归因转人工",
}


def _build_attributor(ctx: RunContext) -> Any:
    """构造归因 Agent（机制层）；system_prompt 用最小可用版本，正式打磨在 T-015。

    Fake 模式下返回 None：Fake LLM 没有 chat 脚本，构造了 AgentAttributor 也只会
    在 `decide()` 里抛 `LLMError("Fake 脚本已耗尽")`，让 attribute 节点走"未配置
    Agent → 转人工"分支更清楚。真实 Agent 归因的机制在 e2e 里显式构造。
    """
    if getattr(ctx.llm, "mode", None) == "fake":
        return None
    from .core.agent_loop import LoopBudget
    from .core.attribution import AgentAttributor

    prompt_path = ctx.settings.resolve_path(
        ctx.settings.raw.get("paths", {}).get(
            "attribution_prompt", "configs/prompts/attribution_system.md"
        )
    )
    return AgentAttributor(
        llm=ctx.llm,
        ctx=ctx,
        budget=LoopBudget(),
        system_prompt=prompt_path.read_text(encoding="utf-8"),
    )


def _build_retrieval_agent(ctx: RunContext) -> Any:
    """构造检索 Agent（机制层）；system_prompt 用最小可用版本，正式打磨以后再做。

    默认关闭（`retrieval.agent.enabled=false`）；Fake 模式下也返回 None：Fake LLM
    没有 chat 脚本，构造了 RetrievalAgent 也只会抛 `LLMError("Fake 脚本已耗尽")`，
    让 ① 节点走"原文一次检索"分支更清楚。真实 Agent 的机制在单测里显式构造。
    """
    if not ctx.settings.retrieval.agent.enabled:
        return None
    if getattr(ctx.llm, "mode", None) == "fake":
        return None
    from .core.retrieval_agent import RetrievalAgent

    prompt_path = ctx.settings.resolve_path(
        ctx.settings.raw.get("paths", {}).get(
            "retrieval_prompt", "configs/prompts/retrieval_system.md"
        )
    )
    return RetrievalAgent(
        llm=ctx.llm,
        ctx=ctx,
        system_prompt=prompt_path.read_text(encoding="utf-8"),
    )


def _build_reranker(ctx: RunContext) -> Any:
    """构造在线重排器（T-034）；`retrieval.rerank.enabled=false` 时返回 None。

    嵌入器按模式选：Fake 模式用确定性的 `HashEmbedder`（不联网、用量恒为 0）；
    真实模式用配置构造 `OpenAICompatEmbedder`（`embedding.*`，缺配置时构造期就报
    `ConfigError`，不会等到检索结束才失败）。重排器持有的向量库是进程内 SQLite，
    候选只在本次运行内使用，不落盘。
    """
    if not ctx.settings.retrieval.rerank.enabled:
        return None
    from .core.embedding import HashEmbedder, UsageTrackingEmbedder, build_embedder
    from .tools.hybrid import HybridReranker
    from .tools.vector_store import VectorStore

    embedder: Any = (
        HashEmbedder() if getattr(ctx.llm, "mode", None) == "fake" else build_embedder(ctx.settings)
    )
    hybrid = ctx.settings.retrieval.hybrid
    return HybridReranker(
        UsageTrackingEmbedder(embedder),
        VectorStore(":memory:"),
        rrf_k=hybrid.rrf_k,
        coverage_ref=hybrid.coverage_ref,
        fusion=hybrid.fusion,
    )


def load_topic(path: Path) -> dict:
    """读课题 YAML，校验至少有 topic 字段（API 用同名的内存对象走 `initial_state`）。"""
    if not path.exists():
        raise ConfigError(f"课题文件不存在: {path}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not spec.get("topic"):
        raise ConfigError(f"课题文件缺少 topic 字段: {path}")
    return spec


def initial_state(run_id: str, spec: dict, default_seed: int) -> dict:
    """把课题 spec 转成图的初始 state（数据集三要素缺一即 ConfigError）。"""
    dataset = spec.get("dataset") or {}
    raw_param = spec.get("param_set") or {}
    params = dict(raw_param.get("params") or {})
    params.update(
        {
            "dataset_path": dataset.get("path"),
            "target": dataset.get("target"),
            "features": list(dataset.get("features") or []),
        }
    )
    if not params["dataset_path"] or not params["target"] or not params["features"]:
        raise ConfigError("dataset 必须提供 path、target 与 features")

    return {
        "run_id": run_id,
        "topic": spec["topic"],
        "param_set": ParamSet(
            algorithm=raw_param.get("algorithm", "linear_regression"),
            params=params,
            seed=int(raw_param.get("seed", default_seed)),
        ),
    }


def _dump_model(value: BaseModel | None) -> dict | None:
    """manifest 中统一的结构化输出（兼容字典形态的 state 快照）。"""
    if value is None:
        return None
    return value.model_dump() if isinstance(value, BaseModel) else dict(value)


def _llm_calls_of(state: dict) -> list[LLMCallRecord]:
    """取出 state 里的用量记录；checkpoint 反序列化后可能已是字典。"""
    raw = state.get("llm_calls") or []
    return [
        call if isinstance(call, LLMCallRecord) else LLMCallRecord.model_validate(call)
        for call in raw
    ]


def _termination_of(state: dict) -> Termination | None:
    termination = state.get("termination")
    if termination is None:
        return None
    if isinstance(termination, Termination):
        return termination
    return Termination.model_validate(termination)


def _error_of(state: dict) -> ErrorRecord | None:
    """取 state 里的错误记录；checkpoint 反序列化后可能已是字典。"""
    error = state.get("error")
    if error is None:
        return None
    if isinstance(error, ErrorRecord):
        return error
    return ErrorRecord.model_validate(error)


def _step_records_of(source: dict) -> list[StepRecord]:
    """取 update 或 state 里的单步记录；checkpoint 反序列化后可能已是字典。"""
    raw = source.get("step_records") or []
    return [
        record if isinstance(record, StepRecord) else StepRecord.model_validate(record)
        for record in raw
    ]


def _status_of(state: dict) -> str:
    """manifest 与 run_end 共用的状态判定。"""
    if state.get("error"):
        return "error"
    return "terminated" if state.get("termination") else "ok"


def _termination_reason(termination: Termination | None) -> str:
    if termination is None:
        return ""
    label = TERMINATION_LABELS.get(termination.kind, termination.kind)
    return f"{label}：{termination.reason}" if termination.reason else label


def _exit_code_for(termination: Termination) -> int:
    """按终止语义映射退出码：rejected→2、error→3、escalated→5、其余→1。"""
    if termination.kind == "rejected":
        return EXIT_REVIEW_REJECTED
    if termination.kind == "error":
        return EXIT_ERROR
    if termination.kind == "escalated":
        return EXIT_ESCALATED
    return EXIT_VALIDATION_FAILED


def _error_step(state: dict) -> str:
    """兜底推断出错步骤：取最后一个标记为 error 的步骤，否则 startup。"""
    steps = state.get("step_status") or {}
    for name, status in reversed(list(steps.items())):
        if status == "error":
            return str(name)
    return str(list(steps)[-1]) if steps else "startup"


def _write_manifest(
    run_dir: Path,
    state: dict,
    llm_mode: str,
    project_root: Path | None = None,
    status: str | None = None,
    trace: dict | None = None,
) -> Path:
    validation = state.get("validation_result")
    termination = _termination_of(state)
    error = _error_of(state)
    # `status` 非空时覆盖（例如 `awaiting_review`：链路还没结束，也没有 error）。
    status = status or _status_of(state)
    manifest = {
        "run_id": state.get("run_id"),
        # 相对项目根目录，如 "runs/<run_id>"；runs_dir 在项目根目录之外时写 null。
        "run_dir": relative_to_root(run_dir, project_root) if project_root else None,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "topic": state.get("topic"),
        "llm_mode": llm_mode,
        "step_status": state.get("step_status", {}),
        "steps": [record.model_dump() for record in _step_records_of(state)],
        # state 里 artifacts 是按 kind 覆盖的 dict（P1-3），manifest 仍输出为列表
        "artifacts": [
            ref.model_dump() if isinstance(ref, ArtifactRef) else dict(ref)
            for ref in (state.get("artifacts") or {}).values()
        ],
        "review_log": [record.model_dump() for record in state.get("review_log") or []],
        "attributions": [
            record.model_dump() if isinstance(record, AttributionRecord) else dict(record)
            for record in state.get("attributions") or []
        ],
        "retrieval_trace": _dump_model(state.get("retrieval_trace")),
        "validation": validation.model_dump() if validation else None,
        "error": _dump_model(error),
        "termination": _dump_model(termination),
        "terminal_reason": _termination_reason(termination),
        "compute_summary": (state["compute_result"].summary if state.get("compute_result") else ""),
        "llm_usage": summarize_llm_usage(_llm_calls_of(state)),
    }
    if trace is not None:
        manifest["trace"] = trace_summary(trace)
    path = run_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _trace_model(settings: Settings, llm_mode: str) -> str | None:
    """trace 用模型名：Fake 恒为 "fake"（成本 0），真实取 settings.llm.model。"""
    if llm_mode == "fake":
        return "fake"
    return settings.llm.model


def _write_trace_file(run_dir: Path, state: dict, llm_mode: str, settings: Settings) -> dict:
    """组装 span 树并写 `runs/<id>/trace.json`，返回 trace 字典（供 manifest 与 OTLP）。"""
    model = _trace_model(settings, llm_mode)
    pricing = load_pricing(settings.project_root / "configs" / "pricing.yaml")
    run_id = str(state.get("run_id") or run_dir.name)
    trace = build_trace(state, run_id, model=model, pricing=pricing)
    path = run_dir / "trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace


def _parse_otlp_headers(raw: str) -> dict[str, str]:
    """把 `k=v,k2=v2` 解析成请求头字典；非法片段直接忽略。"""
    headers: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.strip():
            headers[key.strip()] = value.strip()
    return headers


def _maybe_export_otlp(trace: dict) -> None:
    """设置了 `RF_OTLP_ENDPOINT` 才导出；失败只打印一行警告，不影响运行结果与退出码。"""
    endpoint = os.getenv("RF_OTLP_ENDPOINT")
    if not endpoint:
        return
    headers = _parse_otlp_headers(os.getenv("RF_OTLP_HEADERS") or "")
    try:
        export_trace_otlp(trace, endpoint=endpoint, headers=headers)
    except Exception as exc:  # noqa: BLE001 - 导出失败不能影响运行结果
        print(f"OTLP 导出失败: {exc}", file=sys.stderr)


def _print_review_prompt(payload: dict) -> None:
    print("\n=== ③ 数据审核（HITL） ===")
    print(f"数据集: {payload.get('dataset')}")
    print(f"规则: {payload.get('rules')}")
    print(
        f"行数: {payload.get('rows')} / 列数: {payload.get('columns')}"
        f"（清洗前缺失率: {payload.get('raw_missing_rate')}）"
    )
    print(
        f"移除含缺失行: {payload.get('dropped_rows')} / "
        f"移除重复行: {payload.get('duplicates_removed')} / "
        f"清洗后缺失率: {payload.get('missing_rate')}"
    )
    for note in payload.get("notes") or []:
        print(f"- {note}")


def _ask_decision(auto_approve: bool, max_attempts: int = 5) -> dict:
    """人工审核输入：只接受 approve / reject，其余重新询问，超限抛 ConfigError。"""
    if auto_approve:
        return {"decision": "approve", "reason": "auto-approve（脚本化复现，非人工决策）"}

    for _ in range(max_attempts):
        answer = input("审核决定 [approve/reject]: ").strip().lower()
        if answer == "approve":
            return {"decision": "approve", "reason": ""}
        if answer == "reject":
            return {"decision": "reject", "reason": input("拒绝原因: ").strip() or "未提供原因"}
        print(f"无法识别的输入：{answer!r}，请输入 approve 或 reject。")

    raise ConfigError(f"审核输入连续 {max_attempts} 次无法识别，已终止链路")


def _review_detail(payload: dict) -> dict:
    """interrupt 事件的摘要：只放被审核对象的规模字段。

    审核载荷里的 `dataset` / `rules` 是路径（配置，不是被审核对象），`notes`
    是自由文本且可能很长，都不进事件日志。
    """
    keys = ("stage", "rows", "columns", "missing_rate", "dropped_rows", "duplicates_removed")
    return {key: payload.get(key) for key in keys}


def _stream_updates(graph, source: Any, config: dict, ctx: RunContext, log: EventLog) -> None:
    """消费一次 `stream`，把每个节点的 StepRecord 写成 `step_end` 事件。

    update 只在节点**完成**时产出：③ 被 `interrupt()` 打断的那一次不产出
    update，恢复后重跑才产出，因此每个节点恰好一条 `step_end`（T-004 A1）。
    运行上下文经 `context=` 传入，不进 checkpoint（P1-1）。
    """
    for chunk in graph.stream(source, config, context=ctx, stream_mode="updates"):
        for key, update in chunk.items():
            if key == "__interrupt__":
                continue
            for record in _step_records_of(update):
                log.from_step_record(record)


def _drive_graph(
    graph,
    config: dict,
    state: Any,
    ctx: RunContext,
    log: EventLog,
    *,
    auto_approve: bool = False,
    decision: dict | None = None,
    pause_at_review: bool = False,
    print_review: bool = True,
) -> tuple[dict, bool]:
    """驱动图直到结束或停在审核处，返回 `(最终 state, 是否停在审核处)`。

    是否还在等审核一律读 `graph.get_state(config).interrupts`，不看上一次 stream
    的返回值：跨进程恢复（P1-1）时本进程从没跑过第一段，只有 checkpoint 知道状态。

    `decision` 非空时用它（resume 命令），否则按 `auto_approve` 决定是自动放行
    还是在终端里问。`print_review=False` 时既不打印审核载荷也不读终端（API/MCP
    的 stdout 有别的用途，且不能阻塞在 `input()` 上）。
    """
    source: Any = state
    while True:
        _stream_updates(graph, source, config, ctx, log)

        pending = graph.get_state(config)
        if not pending.interrupts:
            break

        payload = pending.interrupts[0].value
        log.emit("interrupt", detail=_review_detail(payload))

        if pause_at_review:
            return dict(pending.values or {}), True

        if print_review:
            _print_review_prompt(payload)
        chosen = decision if decision is not None else _ask_decision(auto_approve)
        log.emit("resume", detail={"decision": chosen.get("decision")})
        source = Command(resume=chosen)

    return dict(graph.get_state(config).values or {}), False


def _emit_run_end(log: EventLog, state: dict, status: str | None = None) -> None:
    """所有退出路径都写一条 run_end：状态与 manifest 一致，失败时带 error。"""
    termination = _termination_of(state)
    detail: dict[str, Any] = {}
    if termination is not None:
        detail["termination"] = {"kind": termination.kind}
    log.emit(
        "run_end",
        status=status or _status_of(state),
        error=_error_of(state),
        detail=detail,
    )


@dataclass
class RunOutcome:
    """一次运行（首跑或恢复）的终态摘要。

    核心驱动层只产出它：不打印、不调用 `sys.exit`。三个入口各自决定怎么用——
    CLI 打印并按 `exit_code` 退出，MCP 取 `status`/`message`，API 落成 HTTP 响应。
    """

    run_id: str
    run_dir: Path
    status: str  # awaiting_review | ok | terminated | error
    llm_mode: str
    message: str
    paused: bool = False
    manifest_path: Path | None = None
    termination_kind: str | None = None
    termination_reason: str = ""
    error_code: str | None = None
    error_message: str = ""
    validation_passed: bool | None = None
    validation_summary: str = ""
    exit_code: int = EXIT_OK
    # 业务异常也被收敛成 outcome（`error_code` 等字段），这里带上原异常供 CLI 重新抛出，
    # 保持 "CLI 失败时报错到 stderr 并给出退出码 3" 的既有行为。
    error_exc: BaseException | None = None


@dataclass
class RunSession:
    """一次运行所需的进程内对象：上下文、图、checkpointer、事件日志。

    `settings` 已经是该 run 私有的副本（`with_llm_mode`），并发 run 之间不共享可变字段。
    """

    settings: Settings
    run_id: str
    run_dir: Path
    llm_mode: str
    ctx: RunContext
    graph: Any
    config: dict
    checkpointer: Any
    log: EventLog

    def close(self) -> None:
        """关闭 sqlite 连接（memory 后端是空操作），保证 run 目录自包含。"""
        close_checkpointer(self.checkpointer)


def run_settings(settings: Settings, mode: str | None = None) -> Settings:
    """返回该 run **私有**的 settings 副本；给 `mode` 时同时覆盖 `llm.mode`。

    多线程并发跑多个 run 时，`settings.llm.mode` 是共享对象上的可变字段：原地改会让
    并发 run 互相串模式（`run_fake_paused` 以前就是原地改），`RunContext.from_run_id`
    恢复时也会把生效模式写回 `settings.llm.mode`。所以每次运行都拿一份副本
    （T-026 调查问题 2）。
    """
    llm = replace(settings.llm) if mode is None else replace(settings.llm, mode=mode)
    return replace(settings, llm=llm)


def build_run_context(settings: Settings, run_dir: Path) -> RunContext:
    """构造一次运行的上下文：LLM、检索器与两个 Agent（CLI/MCP/API 共用）。"""
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=build_llm_client(settings),
        retriever=build_retriever(settings),
    )
    ctx.attributor = _build_attributor(ctx)
    ctx.retrieval_agent = _build_retrieval_agent(ctx)
    ctx.reranker = _build_reranker(ctx)
    return ctx


def build_run_context_from_run_id(
    run_id: str, settings: Settings, *, mode: str | None = None
) -> RunContext:
    """只凭 run_id 与磁盘文件重建上下文（跨进程恢复；CLI 与 API 共用）。

    `settings` 必须是调用方私有的副本（`run_settings`）：`from_run_id` 会按
    `run_meta.json` 把生效模式写回 `settings.llm.mode`。显式给 `mode` 且与
    run_meta 记录不一致时 `from_run_id` 抛 ConfigError。
    """
    ctx = RunContext.from_run_id(run_id, settings, mode=mode)
    ctx.attributor = _build_attributor(ctx)
    ctx.retrieval_agent = _build_retrieval_agent(ctx)
    ctx.reranker = _build_reranker(ctx)
    return ctx


def open_run_session(settings: Settings, run_id: str, run_dir: Path, ctx: RunContext) -> RunSession:
    """按 settings 建 checkpointer 与图，装配 EventLog（首跑与恢复共用）。"""
    checkpointer = make_checkpointer(settings.checkpoint.backend, run_dir)
    return RunSession(
        settings=settings,
        run_id=run_id,
        run_dir=run_dir,
        llm_mode=getattr(ctx.llm, "mode", "unknown"),
        ctx=ctx,
        graph=build_graph(checkpointer),
        config={"configurable": {"thread_id": run_id}},
        checkpointer=checkpointer,
        log=EventLog(run_dir / "events.jsonl", run_id),
    )


def record_run_start(session: RunSession, *, topic_path: Path, spec: dict) -> None:
    """写首跑的 run_start 事件与 run_meta.json（恢复不写这两样）。"""
    session.log.emit("run_start", detail={"topic": spec.get("topic"), "llm_mode": session.llm_mode})
    write_run_meta(
        session.run_dir,
        run_meta_payload(
            run_id=session.run_id,
            topic_path=topic_path,
            llm_mode=session.llm_mode,
            settings=session.settings,
            checkpoint_backend=session.settings.checkpoint.backend,
        ),
    )


def _finalize(session: RunSession, result: dict, *, status: str | None = None) -> Path:
    """写 trace.json 与 manifest.json，返回 manifest 路径（不打印、不发事件）。"""
    trace = _write_trace_file(session.run_dir, result, session.llm_mode, session.settings)
    _maybe_export_otlp(trace)
    return _write_manifest(
        session.run_dir,
        result,
        session.llm_mode,
        session.settings.project_root,
        status=status,
        trace=trace,
    )


def _outcome_of(
    session: RunSession,
    state: dict,
    *,
    paused: bool,
    manifest_path: Path,
    error_code: str | None = None,
    error_message: str = "",
    error_exc: BaseException | None = None,
) -> RunOutcome:
    """把 state（或构造出来的失败 state）折叠成 RunOutcome 的字段。"""
    termination = _termination_of(state)
    validation = state.get("validation_result")
    if paused:
        status = "awaiting_review"
    else:
        status = _status_of(state)

    if paused:
        exit_code = EXIT_AWAITING_REVIEW
        message = (
            "已停在人工审核处，需要人工审核；继续："
            f"researchflow resume --run-id {session.run_id} --decision approve"
        )
    elif termination is not None:
        exit_code = _exit_code_for(termination)
        if error_code:
            message = f"{error_code}: {error_message}"
        else:
            message = _termination_reason(termination) or status
    elif validation is None:
        exit_code = EXIT_VALIDATION_FAILED
        message = status
    else:
        exit_code = EXIT_OK if validation.passed else EXIT_VALIDATION_FAILED
        message = status

    return RunOutcome(
        run_id=session.run_id,
        run_dir=session.run_dir,
        status=status,
        llm_mode=session.llm_mode,
        message=message,
        paused=paused,
        manifest_path=manifest_path,
        termination_kind=termination.kind if termination is not None else None,
        termination_reason=_termination_reason(termination),
        error_code=error_code,
        error_message=error_message,
        validation_passed=validation.passed if validation is not None else None,
        validation_summary=validation.summary if validation is not None else "",
        exit_code=exit_code,
        error_exc=error_exc,
    )


def _record_failure(
    session: RunSession, exc: Exception, snapshot: dict, fallback_topic: Any
) -> RunOutcome:
    """把异常落成 state 里的 error/termination，写 trace、manifest 与 run_end。"""
    snapshot.setdefault("run_id", session.run_id)
    snapshot.setdefault("topic", fallback_topic)
    if isinstance(exc, ResearchFlowError):
        error = ErrorRecord(
            code=exc.code, step=_error_step(snapshot), message=exc.message, detail=exc.detail
        )
    else:
        # 未知异常（非 RF 体系）也必须有可定位的 manifest，否则 run 会永远停在 running。
        error = ErrorRecord(
            code="RF-000",
            step=_error_step(snapshot),
            message=str(exc) or type(exc).__name__,
            detail={"type": type(exc).__name__},
        )
    snapshot["error"] = error
    snapshot["termination"] = Termination(
        kind="error", step=error.step, reason=error.message, error=error
    )
    manifest = _finalize(session, snapshot)
    _emit_run_end(session.log, snapshot)
    return _outcome_of(
        session,
        snapshot,
        paused=False,
        manifest_path=manifest,
        error_code=error.code,
        error_message=error.message,
        error_exc=exc,
    )


def drive_run(
    session: RunSession,
    source: Any,
    *,
    pause_at_review: bool = False,
    decision: dict | None = None,
    auto_approve: bool = False,
    print_review: bool = True,
    catch_unexpected: bool = False,
    fallback_topic: Any = None,
) -> RunOutcome:
    """驱动图到结束或停在审核处，返回 `RunOutcome`。

    - 不打印（`print_review` 只控制是否打印审核载荷）、不调用 `sys.exit`。
    - `ResearchFlowError` 一定被收敛成 `status="error"` 的 outcome，并写完 manifest 与
      `run_end`（首跑与恢复的失败收尾统一在这里）。
    - 未知异常默认原样抛出（保持 CLI 行为）；`catch_unexpected=True` 时也收敛成
      `error`，这是 API 线程池需要的：线程静默死掉会让 run 永远显示 running（A9）。
    - `session` 在这里关闭，调用方不要再 close。
    """
    try:
        result, paused = _drive_graph(
            session.graph,
            session.config,
            source,
            session.ctx,
            session.log,
            auto_approve=auto_approve,
            decision=decision,
            pause_at_review=pause_at_review,
            print_review=print_review,
        )
    except Exception as exc:
        # 先取快照（读 checkpoint，需要连接还开着），再关闭连接。
        snapshot = _state_snapshot(session.graph, session.config)
        session.close()
        if not isinstance(exc, ResearchFlowError) and not catch_unexpected:
            raise
        return _record_failure(session, exc, snapshot, fallback_topic)

    session.close()

    if paused:
        # 停在审核处：不写 run_end（链路没结束），只更新 manifest 状态。
        manifest = _finalize(session, result, status="awaiting_review")
        return _outcome_of(session, result, paused=True, manifest_path=manifest)

    manifest = _finalize(session, result)
    _emit_run_end(session.log, result)
    return _outcome_of(session, result, paused=False, manifest_path=manifest)


def _report_cli_outcome(outcome: RunOutcome) -> int:
    """CLI 的输出与退出码；把文件的写入留在核心层之后，日志口径与重构前一致。"""
    print(f"\nmanifest: {outcome.manifest_path}")
    if outcome.error_exc is not None:
        print(f"链路终止: {outcome.termination_reason}")
        raise outcome.error_exc
    if outcome.paused:
        print(f"已停在人工审核处，run_id={outcome.run_id}")
        print(f"继续：researchflow resume --run-id {outcome.run_id} --decision approve")
        return outcome.exit_code
    if outcome.termination_kind is not None:
        print(f"链路终止: {outcome.termination_reason}")
        return outcome.exit_code
    if outcome.validation_passed is None:
        print("未产出校验报告")
    else:
        verdict = "通过" if outcome.validation_passed else "未通过"
        print(f"硬规则校验: {verdict} - {outcome.validation_summary}")
    return outcome.exit_code


def run(args: argparse.Namespace) -> int:
    """CLI 首跑：建上下文 → 记录 run_start/run_meta → 驱动 → 按 outcome 打印退出。"""
    settings = load_settings()
    if args.mode:
        settings.llm.mode = args.mode

    topic_path = Path(args.topic)
    spec = load_topic(topic_path)
    run_id = uuid.uuid4().hex[:12]
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    session = open_run_session(settings, run_id, run_dir, build_run_context(settings, run_dir))
    print(f"run_id={run_id} run_dir={run_dir} llm_mode={session.llm_mode}")

    record_run_start(session, topic_path=topic_path, spec=spec)
    outcome = drive_run(
        session,
        initial_state(run_id, spec, settings.seed),
        pause_at_review=args.pause_at_review,
        auto_approve=args.auto_approve,
        fallback_topic=spec.get("topic"),
    )
    return _report_cli_outcome(outcome)


def resume(args: argparse.Namespace) -> int:
    """在**新进程**里接着跑一次停在审核处的运行（P1-1）。"""
    settings = load_settings()
    run_id = args.run_id
    # 先重建上下文：run_id 不存在或 run_meta.json 缺失时在这里就抛 ConfigError，
    # 此时还没有 EventLog，不会往该 run 目录里写任何东西。
    # `--mode` 缺省时用 run_meta.json 记录的 llm_mode 恢复（B17）。
    ctx = build_run_context_from_run_id(run_id, settings, mode=args.mode)
    session = open_run_session(settings, run_id, ctx.run_dir, ctx)

    pending = session.graph.get_state(session.config)
    if not pending.interrupts:
        session.close()
        raise ConfigError(f"run {run_id} 没有停在人工审核处，无法 resume")

    decision = {"decision": args.decision, "reason": args.reason or ""}
    print(f"resume run_id={run_id} run_dir={ctx.run_dir} llm_mode={session.llm_mode}")
    session.log.emit("resume", detail={"decision": decision["decision"]})

    outcome = drive_run(session, Command(resume=decision), decision=decision)
    return _report_cli_outcome(outcome)


def run_fake_paused(topic_path: Path, settings: Settings) -> dict[str, Any]:
    """以 Fake 模式启动一次运行并停在人工审核处，供 MCP `start_run` 工具复用。

    固定 Fake 模式（不读真实模型、不自动通过审核）、固定 `pause_at_review`。
    与 `run()` 的区别：不打印任何东西（MCP 的 stdout 被 JSON-RPC 占用），结果
    以字典返回而不是退出码。可预期失败不抛异常，落到返回的 status/message。

    不原地修改传入的 `settings`（T-026 调查问题 2）：Fake 模式经 `run_settings`
    落在一份副本上，多线程并发调用时不会互相串模式。

    返回 `{"run_id", "status", "message"}`：status 为 `awaiting_review`（正常停
    在审核处）、`terminated`（链路提前终止）或 `error`。
    """
    fake_settings = run_settings(settings, "fake")
    spec = load_topic(topic_path)
    run_id = uuid.uuid4().hex[:12]
    run_dir = fake_settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    session = open_run_session(
        fake_settings, run_id, run_dir, build_run_context(fake_settings, run_dir)
    )
    record_run_start(session, topic_path=topic_path, spec=spec)
    outcome = drive_run(
        session,
        initial_state(run_id, spec, fake_settings.seed),
        pause_at_review=True,
        fallback_topic=spec.get("topic"),
    )
    return {"run_id": outcome.run_id, "status": outcome.status, "message": outcome.message}


def _state_snapshot(graph, config: dict) -> dict:
    """取当前 checkpoint 快照，用于失败时写 manifest。"""
    try:
        values = graph.get_state(config).values
    except Exception:  # noqa: BLE001 - checkpoint 不可读时退化为最小 manifest
        return {}
    return dict(values or {})


def trace(args: argparse.Namespace) -> int:
    """打印某次运行的 span 树（缩进文本：名称、耗时、token、成本）。"""
    settings = load_settings()
    run_id = args.run_id
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ConfigError(f"非法 run_id: {run_id!r}（需要 12 位十六进制）")
    run_dir = settings.runs_dir / run_id
    trace_path = run_dir / "trace.json"
    if not trace_path.is_file():
        raise ConfigError(f"run {run_id} 没有 trace.json")
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    print(render_trace_tree(data))
    return EXIT_OK


def _mcp_command(args: argparse.Namespace) -> int:
    """运行 stdio MCP Server（阻塞直到客户端断开）；工具暴露范围由 --allow-exec 决定。"""
    # 延迟导入：mcp_server 反向 import 了本模块的 run_fake_paused，模块级 import 会成环。
    from .mcp_server import run_mcp

    run_mcp(allow_exec=args.allow_exec)
    return EXIT_OK


def serve(args: argparse.Namespace) -> int:
    """起 FastAPI 服务（T-026）。默认只绑本机，不做鉴权，不要直接暴露到公网。"""
    # 延迟导入：api 反向 import 了本模块的运行核心，模块级 import 会成环（同 mcp_server）。
    import uvicorn

    from .api import create_app

    app = create_app(load_settings())
    uvicorn.run(app, host=args.host, port=args.port)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchflow", description="Gate 1 MVP 垂直切片")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="执行 ①→⑦ 链路")
    run_parser.add_argument("--topic", required=True, help="课题 YAML 路径")
    run_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="自动通过 ③ 审核，仅用于脚本化复现（仍会触发 interrupt）",
    )
    run_parser.add_argument(
        "--pause-at-review",
        action="store_true",
        help="停在 ③ 审核处就退出进程（退出码 4），之后用 resume 在新进程里继续",
    )
    run_parser.add_argument("--mode", choices=["auto", "fake", "real"], default=None)
    run_parser.set_defaults(func=run)

    resume_parser = sub.add_parser("resume", help="在新进程里继续一次停在审核处的运行")
    resume_parser.add_argument("--run-id", required=True, help="run 目录名（12 位十六进制）")
    resume_parser.add_argument("--decision", required=True, choices=["approve", "reject"])
    resume_parser.add_argument("--reason", default="", help="拒绝原因（decision=reject 时）")
    resume_parser.add_argument(
        "--mode",
        choices=["fake", "real"],
        default=None,
        help="覆盖恢复模式；缺省用 run_meta.json 记录的 llm_mode，与记录不一致时抛 ConfigError",
    )
    resume_parser.set_defaults(func=resume)

    trace_parser = sub.add_parser("trace", help="打印某次运行的 span 树")
    trace_parser.add_argument("--run-id", required=True, help="run 目录名（12 位十六进制）")
    trace_parser.set_defaults(func=trace)

    mcp_parser = sub.add_parser("mcp", help="以 stdio 方式运行 MCP Server")
    mcp_parser.add_argument(
        "--allow-exec",
        action="store_true",
        help="暴露 exec/write 类工具（run_direct_compute、start_run）",
    )
    mcp_parser.set_defaults(func=_mcp_command)

    serve_parser = sub.add_parser("serve", help="起 FastAPI 服务（HTTP + SSE，默认只绑本机）")
    serve_parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    serve_parser.set_defaults(func=serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ResearchFlowError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

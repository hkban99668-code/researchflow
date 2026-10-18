"""节点步骤装饰器：把 ResearchFlowError 收敛为 state 更新，并产出 StepRecord。

失败语义必须存在于 state 中（而不是只存在于 Python 调用栈），否则 manifest
记录不到失败步骤，Gate 2 的归因与回溯也无从下手。

装饰器只捕获 ResearchFlowError，绝不吞掉 `interrupt()` 抛出的中断信号。

已知不处理的情况：节点里 LLM 调用成功、之后的代码又抛出非 LLM 异常时，那次成功
调用的用量会随被丢弃的 update 一起丢失。要覆盖它，得让节点自己 try/except 后
重新抛出，留到 Gate 2 的回溯卡再定。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol

from langgraph.runtime import Runtime
from langgraph.types import Command

from .config import RunContext
from .errors import LLMError, ResearchFlowError
from .llm import LLMUsage, add_usage
from .schemas import ErrorRecord, LLMCallRecord, StepRecord, Termination


class NodeFn(Protocol):
    """节点签名：`runtime` 是**仅限关键字**参数。

    LangGraph 的 `_NodeWithRuntime` 协议就是这么声明的，而且它确实按名字以关键字
    注入上下文；单元测试直接调节点时也用 `node(state, runtime=...)`。

    节点可以返回 dict（普通更新），也可以返回 `Command`（T-014 回溯：`attribute`
    节点用 `Command(goto=...)` 决定去向）。装饰器对两种返回分别处理。
    """

    def __call__(self, state: Any, *, runtime: Runtime[RunContext]) -> dict | Command: ...


# @step 注入到 state 副本里的只读键：本次执行是第几次。节点不能直接拿到 attempt
# （runtime.context 是全局的，不能放每次执行的值），由装饰器在调用前写进 state
# 副本，节点用 `current_attempt(state)` 读取，供 ArtifactRef/ReviewRecord 填 attempt。
ATTEMPT_KEY = "_step_attempt"


def current_attempt(state: dict) -> int:
    """节点读取本次执行是第几次（1 起）。只有 @step 包裹后调用才有意义。"""
    return int(state.get(ATTEMPT_KEY, 1))


def _now_iso() -> str:
    """当前时间：ISO 8601 毫秒级精度、带本地时区。

    毫秒精度让同一秒内开始的 step 在 Jaeger 等时间轴里不再叠在一起（T-025 第 2 轮）；
    五类产物不含 step 时间，可复现性测试不受影响。
    """
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _attempt_of(state: dict, name: str) -> int:
    """该节点第几次执行：state 里同名的已有记录条数 + 1。"""
    records = state.get("step_records") or []
    return 1 + sum(
        1
        for record in records
        # checkpoint 反序列化后可能已是字典，两种形态都要认。
        if (record["step"] if isinstance(record, dict) else record.step) == name
    )


def _usage_of(update: dict, name: str) -> LLMUsage | None:
    """本次 update 中属于该节点的 LLM 用量合计；一次都没调用则为 None。"""
    total: LLMUsage | None = None
    for call in update.get("llm_calls") or []:
        if call.step != name:
            continue
        total = call.usage if total is None else add_usage(total, call.usage)
    return total


def _context_of(runtime: Runtime[RunContext] | None) -> RunContext | None:
    """取运行上下文；单元测试直接调装饰后的函数时可能没传 runtime。"""
    return getattr(runtime, "context", None)


def _failed_usage_of(
    exc: ResearchFlowError, runtime: Runtime[RunContext] | None
) -> LLMUsage | None:
    """失败那次 LLM 调用的用量；只在异常是 `LLMError` 时可信。

    `complete()` 开始时会把 `last_usage` 置空，所以抛 `LLMError` 时它一定属于
    这次失败的调用；换作别的异常，它可能还是上一个节点留下的，会重复计数。
    """
    if not isinstance(exc, LLMError):
        return None
    ctx = _context_of(runtime)
    if ctx is None:
        return None
    return getattr(ctx.llm, "last_usage", None)


def _failed_duration_of(exc: ResearchFlowError, runtime: Runtime[RunContext] | None) -> int | None:
    """失败那次 LLM 调用的耗时；与 `_failed_usage_of` 同源、同样只认 `LLMError`。"""
    if not isinstance(exc, LLMError):
        return None
    ctx = _context_of(runtime)
    if ctx is None:
        return None
    return getattr(ctx.llm, "last_duration_ms", None)


def _stamp_attempt(calls: list, attempt: int) -> list:
    """给本次 update 里的 `LLMCallRecord` 注入 attempt（与 StepRecord.attempt 同源）。

    节点构造记录时不知道自己是第几次执行，由装饰器统一补。旧 checkpoint 记录
    可能已是字典，两种形态都认；已有 attempt 的不覆盖。
    """
    stamped: list = []
    for call in calls:
        if isinstance(call, LLMCallRecord):
            if call.attempt is None:
                call = call.model_copy(update={"attempt": attempt})
        else:
            data = dict(call)
            if data.get("attempt") is None:
                data["attempt"] = attempt
            call = data
        stamped.append(call)
    return stamped


def step(name: str) -> Callable[[NodeFn], NodeFn]:
    """包裹节点：每次执行追加一条 StepRecord，失败写 error 与 termination。"""

    def decorate(fn: NodeFn) -> NodeFn:
        # 第二个参数**必须叫 `runtime`**：LangGraph 按参数名注入（见
        # `langgraph._internal._runnable.KWARGS_CONFIG_KEYS`），改名就收不到上下文。
        def node(state: dict, *, runtime: Runtime[RunContext]) -> dict | Command:
            started = time.perf_counter()
            started_at = _now_iso()
            attempt = _attempt_of(state, name)
            try:
                # 节点自身返回的 step_status 优先（例如 rejected）。
                # 把本次 attempt 写进 state 副本的只读键，节点用 current_attempt 读取。
                node_state = {**state, ATTEMPT_KEY: attempt}
                result = fn(node_state, runtime=runtime)

                if isinstance(result, Command):
                    # 回溯节点返回 Command(goto=...)：把 step_records 追加进
                    # Command.update，而不是把它当成 dict 展开（Command 不是映射）。
                    cmd_update = dict(result.update or {})
                    if cmd_update.get("llm_calls"):
                        cmd_update["llm_calls"] = _stamp_attempt(cmd_update["llm_calls"], attempt)
                    step_status = dict(cmd_update.get("step_status") or {})
                    step_status[name] = step_status.get(name, "ok")
                    cmd_update["step_status"] = step_status
                    record = StepRecord(
                        step=name,
                        status=step_status[name],
                        attempt=attempt,
                        started_at=started_at,
                        duration_ms=_elapsed_ms(started),
                        llm_usage=_usage_of(cmd_update, name),
                    )
                    cmd_update["step_records"] = [
                        *(cmd_update.get("step_records") or []),
                        record,
                    ]
                    return replace(result, update=cmd_update)

                update: dict[str, Any] = {
                    "step_status": {name: "ok"},
                    **result,
                }
                if update.get("llm_calls"):
                    update["llm_calls"] = _stamp_attempt(update["llm_calls"], attempt)
                status = (update.get("step_status") or {}).get(name, "ok")
                return {
                    **update,
                    "step_records": [
                        StepRecord(
                            step=name,
                            status=status,
                            attempt=attempt,
                            started_at=started_at,
                            duration_ms=_elapsed_ms(started),
                            llm_usage=_usage_of(update, name),
                        )
                    ],
                }
            except ResearchFlowError as exc:
                error = ErrorRecord(
                    code=exc.code,
                    step=name,
                    message=exc.message,
                    detail=exc.detail,
                    attempt=attempt,
                    retryable=exc.retryable,
                    backtrack_to=exc.default_backtrack_to,
                )
                usage = _failed_usage_of(exc, runtime)
                duration = _failed_duration_of(exc, runtime)
                return {
                    "step_status": {name: "error"},
                    "error": error,
                    "termination": Termination(
                        kind="error",
                        step=name,
                        reason=exc.message,
                        error=error,
                    ),
                    # 失败那次调用同样花钱，记成一条空 response_model 的记录。
                    "llm_calls": (
                        [
                            LLMCallRecord(
                                step=name,
                                response_model="",
                                usage=usage,
                                duration_ms=duration,
                                attempt=attempt,
                            )
                        ]
                        if usage is not None
                        else []
                    ),
                    "step_records": [
                        StepRecord(
                            step=name,
                            status="error",
                            attempt=attempt,
                            started_at=started_at,
                            duration_ms=_elapsed_ms(started),
                            error=error,
                            llm_usage=usage,
                        )
                    ],
                }

        node.__name__ = getattr(fn, "__name__", name)
        node.__doc__ = getattr(fn, "__doc__", None)
        return node

    return decorate

"""attribute 节点：把失败转成决策，再由 `Command` 执行回溯或转人工。

入口条件（由图的 `route_next` / `route_after_validation` 保证）：
- `termination.kind == "error"`：节点报错，走**规则层**，一律转人工（不调模型）；
- `validation_result.passed is False`：⑦ 校验不通过，走**Agent 层**，用只读工具查
  原因、决策工具给结论，再由图执行。

预算（`attribution.max_total_backtracks`）在**最前面**拦：全局回溯次数到上限时
不再决策，直接 `budget_exhausted` 结束。回溯到 `data_prepare` 会重新经过
`data_review` 的人工审核——数据变了，人必须重新审（这是 T-007 拆节点的自然效果，
不要绕过）。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..core.agent_loop import tool_loop_call_record
from ..core.config import RunContext
from ..core.errors import ResearchFlowError
from ..core.schemas import (
    AttributionDecision,
    AttributionRecord,
    BacktrackPatch,
    ParamSet,
    Termination,
)
from ..core.step import current_attempt, step


def apply_patch(param_set: ParamSet, patch: BacktrackPatch) -> ParamSet:
    """把 patch 合并进 `ParamSet`：`cleaning` 进 `params["cleaning"]`，`target_transform` /
    `exclude_features` 进 `params`。字段已经过 `validate_for`，这里只做合并不做校验。"""
    params = dict(param_set.params)
    if patch.y_outlier_strategy is not None or patch.y_outlier_k is not None:
        cleaning = dict(params.get("cleaning") or {})
        if patch.y_outlier_strategy is not None:
            cleaning["y_outlier_strategy"] = patch.y_outlier_strategy
        if patch.y_outlier_k is not None:
            cleaning["y_outlier_k"] = patch.y_outlier_k
        params["cleaning"] = cleaning
    if patch.target_transform is not None:
        params["target_transform"] = patch.target_transform
    if patch.exclude_features is not None:
        params["exclude_features"] = list(patch.exclude_features)
    return ParamSet(algorithm=param_set.algorithm, params=params, seed=param_set.seed)


def _action_of(record: AttributionRecord) -> str:
    """取一条归因记录里决策的动作；兼容 checkpoint 反序列化后的字典形态。"""
    if isinstance(record, dict):
        decision = record.get("decision") or {}
        return str(decision.get("action", ""))
    decision = record.decision
    if isinstance(decision, dict):
        return str(decision.get("action", ""))
    return str(decision.action)


def _failed_check_names(state: dict) -> list[str]:
    validation = state.get("validation_result")
    if validation is None:
        return []
    return [check.name for check in validation.checks if not check.passed]


def _escalate(decision: AttributionDecision, record: AttributionRecord, llm_calls: list) -> Command:
    """转人工：写 escalated 终止并追加一条归因记录（含 Agent 回路的用量记录）。"""
    return Command(
        goto=END,
        update={
            "termination": Termination(kind="escalated", step="attribute", reason=decision.reason),
            "attributions": [record],
            "llm_calls": llm_calls,
        },
    )


@step("attribute")
def attribute_node(state: dict, runtime: Runtime[RunContext]) -> Command:
    """把失败转成归因决策并执行（回溯或转人工）。"""
    ctx = runtime.context
    termination = state.get("termination")
    trigger: Literal["error", "validation_failed"] = (
        "error" if termination is not None and termination.kind == "error" else "validation_failed"
    )

    attempt = current_attempt(state)
    attributions = state.get("attributions") or []
    backtracks = sum(1 for record in attributions if _action_of(record) == "backtrack")
    max_total = ctx.settings.attribution.max_total_backtracks

    # 预算闸：全局回溯次数已到上限，不再决策。
    if backtracks >= max_total:
        return Command(
            goto=END,
            update={
                "termination": Termination(
                    kind="budget_exhausted",
                    step="attribute",
                    reason=f"回溯次数已达上限 {max_total}，停止自动回溯",
                ),
            },
        )

    if trigger == "error":
        # 规则层：报错一律转人工，不调 LLM。`backtrack_to` 只作为诊断信息（target_step）。
        error = state.get("error")
        code = getattr(error, "code", None) if error is not None else None
        backtrack_to = getattr(error, "backtrack_to", None) if error is not None else None
        decision = AttributionDecision(
            action="escalate",
            target_step=backtrack_to,
            reason=f"节点报错走规则层转人工：{code or '未知错误'}",
            source="rule",
        )
        record = AttributionRecord(
            attempt=attempt, trigger=trigger, error_code=code, decision=decision
        )
        # 规则层不调模型，没有回路用量。
        return _escalate(decision, record, [])

    # Agent 层：⑦ 校验不通过，交给归因 Agent 决策。
    attributor = ctx.attributor
    agent_failed = False
    failure_reason = None
    if attributor is None:
        # Fake 模式的 CLI 不构造 Agent（Fake LLM 没有脚本，构造了也只会失败），
        # 和跨进程恢复后不再需要决策的情况共用"未配置 Agent → 转人工"分支。
        reason = (
            "Fake 模式未配置归因 Agent"
            if getattr(ctx.llm, "mode", None) == "fake"
            else "未配置 Agent"
        )
        decision = AttributionDecision(action="escalate", reason=reason, source="agent")
        tool_calls = []
        llm_turns = []
        usage = None
    else:
        try:
            decision = attributor.decide(state)
        except ResearchFlowError as exc:
            # 归因 Agent 是兜底环节，它自己失败（网络中断、模型输出不合格、Fake
            # 脚本耗尽等）时不能让整条链路以 error 崩溃：降级为转人工并留下记录。
            # agent_failed 显式标记（T-027），评测据此把这类用例排除出归因准确率
            # 与恢复率的分母，不再混进"Agent 判断转人工"。
            agent_failed = True
            failure_reason = f"{type(exc).__name__}: {exc.message}"
            decision = AttributionDecision(
                action="escalate",
                reason=f"Agent 调用失败：{exc.code} {exc.message}",
                source="agent",
            )
        tool_calls = list(getattr(attributor, "last_tool_calls", []))
        llm_turns = list(getattr(attributor, "last_llm_turns", []))
        usage = getattr(attributor, "last_usage", None)
        # attributor 自己记的失败原因（经 ToolLoopInterrupted 脱敏）优先；自构
        # attributor（Eval 测试替身）没有该属性时，退回用刚捕获的异常拼一句。
        failure_reason = getattr(attributor, "last_failure_reason", None) or failure_reason

    # Agent 回路的用量记成一条 tool_loop 汇总记录：既进 manifest 的 llm_usage，
    # 也让 `@step` 汇总进这次 StepRecord.llm_usage（D-006 / D-015）。Agent 中途失败
    # 时，attributor 已把"失败前已发生"的用量放在 last_* 上，这里同样记上。
    llm_calls = tool_loop_call_record("attribute", usage, llm_turns)

    if decision.action == "escalate":
        record = AttributionRecord(
            attempt=attempt,
            trigger=trigger,
            failed_checks=_failed_check_names(state),
            decision=decision,
            tool_calls=tool_calls,
            llm_turns=llm_turns,
            usage=usage,
            agent_failed=agent_failed,
            failure_reason=failure_reason,
        )
        return _escalate(decision, record, llm_calls)

    # 回溯：patch 与目标节点必须匹配，越权则转人工。
    target_step = decision.target_step or ""
    try:
        current_features = list((state["param_set"].params or {}).get("features") or [])
        decision.patch.validate_for(target_step, features=current_features)
    except ValueError as exc:
        decision = AttributionDecision(
            action="escalate",
            target_step=decision.target_step,
            patch=decision.patch,
            reason=f"patch 越权：{exc}",
            source=decision.source,
        )
        record = AttributionRecord(
            attempt=attempt,
            trigger=trigger,
            failed_checks=_failed_check_names(state),
            decision=decision,
            tool_calls=tool_calls,
            llm_turns=llm_turns,
            usage=usage,
            agent_failed=agent_failed,
            failure_reason=failure_reason,
        )
        return _escalate(decision, record, llm_calls)

    new_param_set = apply_patch(state["param_set"], decision.patch)
    record = AttributionRecord(
        attempt=attempt,
        trigger=trigger,
        failed_checks=_failed_check_names(state),
        decision=decision,
        tool_calls=tool_calls,
        llm_turns=llm_turns,
        usage=usage,
        agent_failed=agent_failed,
        failure_reason=failure_reason,
    )
    return Command(
        goto=decision.target_step,
        update={
            "param_set": new_param_set,
            # 回溯后清掉上一次的终止 / 错误 / 校验结果，让目标节点及其下游重跑
            "termination": None,
            "error": None,
            "validation_result": None,
            "attributions": [record],
            "llm_calls": llm_calls,
        },
    )

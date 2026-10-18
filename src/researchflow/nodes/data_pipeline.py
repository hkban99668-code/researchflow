"""③ 拆成两个节点：准备（清洗 → 预检 → 写候选文件）与审核（只 interrupt + 校验）。

拆开的理由（P1-2）：`interrupt()` 前的清洗是**有副作用**的。持久化 checkpoint
（P1-1）之后，暂停到恢复之间可能隔几天，期间原始 CSV 被改，恢复后重跑就会用新
数据，而审核记录对不上。分开之后：

- `data_prepare`：唯一的副作用节点。产出候选文件与两个 checksum（原始 CSV、
  候选文件），都进 state。
- `data_review`：没有任何写操作，`interrupt()` 返回后重新计算 checksum 与 state
  比对，不一致就抛 `DataValidationError`（RF-003）。因此它重跑严格幂等，Gate 2
  的审核队列也只需要处理这一类挂起点。

checksum 核对必须放在 `interrupt()` **之后**：恢复时本节点从头执行，`interrupt()`
这次直接返回人的决定；核对写在它后面，才能保证核对的是"恢复那一刻"的文件。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from ..core.config import RunContext, relative_to_root, sha256_file
from ..core.errors import ConfigError, DataValidationError
from ..core.schemas import (
    ArtifactRef,
    CleanedDataRef,
    ReviewDecision,
    ReviewRecord,
    Termination,
)
from ..core.step import current_attempt, step
from ..tools.clean_strategies import clean_frame
from ..tools.quality import precheck
from ..tools.rules_engine import AcceptanceRules, load_rules

REVIEW_STAGE = "data_review"
# 候选文件故意不叫 cleaned_data.csv：它是"还没过审的数据"，名字上就要和
# 审核通过后的产物区分开，免得有人把它当成已审数据用。
CANDIDATE_FILENAME = "cleaned_candidate.csv"


def _input_paths(ctx: RunContext, params: dict) -> tuple[Path, Path]:
    """解析原始数据集与准入规则的路径（两个节点共用，规则不重复加载）。"""
    dataset_path = ctx.settings.resolve_path(params["dataset_path"])
    rules_path = ctx.settings.resolve_path(
        ctx.settings.raw.get("paths", {}).get(
            "data_acceptance_rules", "configs/rules/data_acceptance.yaml"
        )
    )
    return dataset_path, rules_path


def _sha256_existing(path: Path, file_key: str, expected: str) -> str:
    """文件仍在就返回 sha256；**被删除视为"已变化"**，抛 RF-003。

    跨进程恢复（P1-1）之后暂停可能持续几天，期间文件被移走是现实情况。让
    `FileNotFoundError` 冒出去会绕过 `@step`，manifest 与 events 都记录不到，
    所以这里显式转成业务错误。
    """
    if not path.exists():
        raise DataValidationError(
            f"文件已被删除，审核结论不再适用: {path.name}",
            detail={"file": file_key, "expected": expected, "actual": None},
        )
    return sha256_file(path)


def _verify_unchanged(dataset_path: Path, candidate: CleanedDataRef, ctx: RunContext) -> None:
    """重新计算两个文件的 sha256，与 state 里的值比对。

    任一不一致说明暂停期间文件被改过，审核结论不再适用，抛 RF-003（数据问题）。
    审核之后文件又被改，则由 ④⑤ 读取时的 `read_verified` 抛 RF-005（产物问题）。
    """
    actual_raw = _sha256_existing(dataset_path, "raw", candidate.raw_checksum)
    if actual_raw != candidate.raw_checksum:
        raise DataValidationError(
            f"原始数据已被改动，审核结论不再适用: {dataset_path.name}",
            detail={"file": "raw", "expected": candidate.raw_checksum, "actual": actual_raw},
        )

    actual_subject = _sha256_existing(
        ctx.artifact_path(candidate.path), "cleaned_candidate", candidate.checksum
    )
    if actual_subject != candidate.checksum:
        raise DataValidationError(
            f"清洗候选文件已被改动，审核结论不再适用: {candidate.path}",
            detail={
                "file": "cleaned_candidate",
                "expected": candidate.checksum,
                "actual": actual_subject,
            },
        )


@step("data_prepare")
def data_prepare_node(state: dict, runtime: Runtime[RunContext]) -> dict:
    """③ 前半：清洗 → 预检 → 写候选文件并算 checksum（唯一的写操作）。"""
    ctx = runtime.context
    params = state["param_set"].params

    dataset_path, rules_path = _input_paths(ctx, params)
    rules = load_rules(rules_path, model=AcceptanceRules)

    raw = pd.read_csv(dataset_path)
    columns = [*list(params.get("features") or []), params["target"]]
    # Gate 2 回溯可调参数：清洗选项放在 params["cleaning"]（可选字典），不提供时等价于 none。
    cleaning = params.get("cleaning") or {}
    cleaned, stats = clean_frame(
        raw,
        columns=columns,
        target=params.get("target"),
        y_outlier_strategy=cleaning.get("y_outlier_strategy", "none"),
        y_outlier_k=float(cleaning.get("y_outlier_k", 3.0)),
    )
    summary = precheck(
        cleaned,
        rules,
        duplicates_removed=stats["duplicates_removed"],
        raw_missing_rate=stats["raw_missing_rate"],
        dropped_rows=stats["dropped_rows"],
        rows_before=stats["rows_before"],
    )

    # 预检已通过才落盘：失败路径不留下"看起来能用的"候选文件。
    raw_checksum = sha256_file(dataset_path)
    target = ctx.artifact_path(CANDIDATE_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(target, index=False)

    ref = CleanedDataRef(
        path=ctx.relative_to_run(target),
        checksum=sha256_file(target),
        rows=summary.rows,
        columns=list(cleaned.columns),
        quality=summary,
        raw_checksum=raw_checksum,
    )

    return {"cleaned_candidate": ref, "step_status": {"data_prepare": "ok"}}


@step("data_review")
def data_review_node(state: dict, runtime: Runtime[RunContext]) -> dict:
    """③ 后半：只做 `interrupt()` 与 checksum 核对，不写任何文件。"""
    ctx = runtime.context
    candidate = state["cleaned_candidate"]
    params = state["param_set"].params

    dataset_path, rules_path = _input_paths(ctx, params)

    # 输入（数据集、规则）相对**项目根目录**；不在项目根目录下的是用户给的
    # 外部绝对路径，保留原样。载荷只用于 CLI 显示，不进 state / manifest / events。
    root = ctx.settings.project_root
    decision = interrupt(
        {
            "stage": REVIEW_STAGE,
            "dataset": relative_to_root(dataset_path, root) or str(dataset_path),
            "rules": relative_to_root(rules_path, root) or str(rules_path),
            "rows": candidate.rows,
            "columns": candidate.quality.columns,
            "missing_rate": round(candidate.quality.missing_rate, 4),
            "raw_missing_rate": round(candidate.quality.raw_missing_rate, 4),
            "dropped_rows": candidate.quality.dropped_rows,
            "duplicates_removed": candidate.quality.duplicates_removed,
            "notes": candidate.quality.notes,
            # 审核结论绑定到具体字节：人审的是这两个 checksum 对应的文件
            "raw_checksum": candidate.raw_checksum,
            "subject_checksum": candidate.checksum,
        }
    )

    payload = decision if isinstance(decision, dict) else {"decision": str(decision)}
    try:
        review = ReviewDecision.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(
            f"审核决策非法（仅接受 approve/reject）：{payload.get('decision', '')!r}"
        ) from exc

    # 必须在这里核对：恢复时本节点从头执行，此刻 interrupt() 已经返回人的决定，
    # 比对的才是"恢复那一刻"的文件。
    _verify_unchanged(dataset_path, candidate, ctx)

    record = ReviewRecord(
        stage=REVIEW_STAGE,
        decision=review.decision,
        reason=review.reason,
        reviewer=review.reviewer,
        # 被审文件相对 run_dir 的路径：打回时候选文件留在 run 目录里，
        # 只靠 checksum 定位不到它，manifest 的 review_log 要能指回文件。
        subject_path=candidate.path,
        subject_checksum=candidate.checksum,
        raw_checksum=candidate.raw_checksum,
        # 第几次审核：回溯重审后 review_log 只增不减，靠 attempt 区分（P1-3）
        attempt=current_attempt(state),
    )

    if review.decision == "reject":
        return {
            "review_log": [record],
            "step_status": {"data_review": "rejected"},
            "termination": Termination(
                kind="rejected",
                step="data_review",
                reason=review.reason or "未提供原因",
            ),
        }

    # 通过审核的数据就是候选文件本身，不复制：复制会多出一份字节相同但路径不同的
    # 文件，checksum 与"被审对象"的对应关系反而说不清。
    return {
        "cleaned_data": candidate,
        "review_log": [record],
        "artifacts": {
            "cleaned_data": ArtifactRef(
                kind="cleaned_data",
                path=candidate.path,
                checksum=candidate.checksum,
                produced_by="data_review",
                attempt=current_attempt(state),
            )
        },
        "step_status": {"data_review": "ok"},
    }

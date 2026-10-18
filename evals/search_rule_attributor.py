"""诊断搜索规则基线：调用和 Agent 同一个 `feature_target_relation` 工具，选诊断 R² 最高的修法。

与朴素规则（只看 owner_step）和聪明规则（固定顺序盲试 iqr → log）都不同，这条基线**看
数据**：直接调用 `feature_target_relation`（Python 函数，不经过 LLM），在下面几个候选里
选诊断 R² 最高、且去掉离群值后行数仍满足 `min_rows` 的一个回溯：

- `iqr k=1.5`（rows_after 必须 ≥ min_rows）；
- `iqr k=3`（rows_after 必须 ≥ min_rows）；
- `log`（仅当 y 全部 > 0、`log_target.applicable` 为 true）；
- 排除每个单独特征。

最高的诊断 R² 低于 ⑦ 的 r2 阈值（0.8）时直接转人工；第二次进入 attribute（说明上次
选的修法没修好）也转人工。这条基线回答一个诚实的问题：**把同样的诊断证据交给一个"挑
数字最好的修法"的规则，它的表现如何？** 如果它和 Agent 一样好，说明在这个问题上 Agent
的价值不在"选修法"，而在别处（说明理由、处理没见过的情况）。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from researchflow.core.config import RunContext
from researchflow.core.schemas import AttributionDecision, BacktrackPatch
from researchflow.core.tools import ToolContext
from researchflow.tools.attribution_tools import build_attribution_registry


def _prior_backtracks(state: dict) -> int:
    """数一下这次运行里已经发生过的回溯次数（含 dict / 对象两种形态）。"""
    total = 0
    for record in state.get("attributions") or []:
        decision = (
            record.get("decision")
            if isinstance(record, dict)
            else getattr(record, "decision", None)
        )
        action = (
            decision.get("action")
            if isinstance(decision, dict)
            else getattr(decision, "action", None)
        )
        if action == "backtrack":
            total += 1
    return total


def _r2_threshold(project_root: Path) -> float:
    """读 ⑦ 的 r2 阈值（与真实链路一致，不复制魔法数字）。"""
    path = project_root / "configs" / "rules" / "submission_checks.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return float(raw["metric_thresholds"]["r2"]["min"])


def _decide_from_relation(out: dict, threshold: float) -> AttributionDecision:
    """从 `feature_target_relation` 的输出选候选，返回回溯或转人工。

    纯函数：不读 state、不碰 run 目录，便于单测直接喂工具输出验证四种行为。
    """
    min_rows = int(out["min_rows"])

    # 候选 = (诊断 R², target_step, patch)。只收满足行数约束、R² 非空的候选。
    candidates: list[tuple[float, str, BacktrackPatch]] = []

    def add(r2_value: float | None, step: str, patch: BacktrackPatch) -> None:
        if r2_value is not None:
            candidates.append((float(r2_value), step, patch))

    for key, k in (("1.5", 1.5), ("3", 3.0)):
        stats = out["robust_by_k"][key]
        if int(stats["rows_after"]) >= min_rows:
            add(
                stats["r2"],
                "data_prepare",
                BacktrackPatch(y_outlier_strategy="iqr", y_outlier_k=k),
            )

    log = out["log_target"]
    if log["applicable"]:
        add(log["r2"], "compute", BacktrackPatch(target_transform="log"))

    for feature, r2_value in out["r2_without_feature"].items():
        add(r2_value, "compute", BacktrackPatch(exclude_features=[feature]))

    if not candidates:
        return AttributionDecision(
            action="escalate", reason="没有满足行数约束的候选修法，转人工", source="agent"
        )

    best_r2, step, patch = max(candidates, key=lambda item: item[0])
    if best_r2 < threshold:
        return AttributionDecision(
            action="escalate",
            reason=f"最高诊断 R² 为 {best_r2:.4f}，低于阈值 {threshold}，转人工",
            source="agent",
        )
    return AttributionDecision(
        action="backtrack",
        target_step=step,
        patch=patch,
        reason=f"诊断 R² 最高的候选（{best_r2:.4f}）",
        source="agent",
    )


class SearchRuleAttributor:
    """诊断搜索规则基线：选诊断 R² 最高的候选；低于阈值或二次进入时转人工。"""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def decide(self, state: dict) -> AttributionDecision:
        prior = _prior_backtracks(state)
        if prior > 0:
            return AttributionDecision(
                action="escalate",
                reason="第二次进入 attribute，说明上次选的修法没修好，转人工",
                source="agent",
            )

        registry = build_attribution_registry(state, self.ctx)
        tool_ctx = ToolContext(
            workspace_root=self.ctx.settings.project_root, settings=self.ctx.settings
        )
        result = registry.invoke("feature_target_relation", {}, tool_ctx)
        if not result.ok:
            code = result.error.code if result.error else "unknown"
            return AttributionDecision(
                action="escalate", reason=f"诊断工具调用失败（{code}），转人工", source="agent"
            )
        out = result.output
        if out is None:
            return AttributionDecision(
                action="escalate", reason="诊断工具未返回输出，转人工", source="agent"
            )

        return _decide_from_relation(out, _r2_threshold(self.ctx.settings.project_root))

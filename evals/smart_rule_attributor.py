"""更聪明的规则基线：看到 `metric:r2` 失败就按固定顺序盲试修法，不调用任何工具。

与朴素规则基线（`fake_attributor.py`）相比，它引入了白名单里三种修法中的两种，按
固定顺序尝试：

- 第 1 次进入 attribute → 回溯到 `data_prepare` + `y_outlier_strategy=iqr, k=1.5`；
- 第 2 次进入 attribute → 回溯到 `compute` + `target_transform=log`；
- 第 3 次进入 attribute 时预算已用完，图的预算闸会直接终止，不会再调用本基线。

它仍然不看数据、不调用工具，只是把"r2 低就先试离群值、再试对数变换"这条人工直觉
规则化。与朴素规则相比，它能修好一部分跨步骤故障（离群值、指数目标），但区分不了
"需要 `exclude_features` / `k=3`"与"修不了"——在这两类用例上它会用完全部预算，产生
无效回溯，这正是归因 Agent 应当避免的代价。
"""

from __future__ import annotations

from researchflow.core.schemas import AttributionDecision, BacktrackPatch


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


class SmartRuleAttributor:
    """聪明规则基线：metric:r2 失败先盲试 iqr k=1.5，再盲试 log；其余情况转人工。"""

    def decide(self, state: dict) -> AttributionDecision:
        validation = state.get("validation_result")
        failed = [
            check
            for check in (validation.checks if validation is not None else [])
            if not check.passed
        ]
        has_r2 = any(check.name == "metric:r2" for check in failed)

        if not has_r2:
            return AttributionDecision(
                action="escalate",
                reason="失败检查里没有 metric:r2，无法用离群值清洗或对数变换修复",
                source="agent",
            )

        prior = _prior_backtracks(state)
        if prior == 0:
            return AttributionDecision(
                action="backtrack",
                target_step="data_prepare",
                patch=BacktrackPatch(y_outlier_strategy="iqr", y_outlier_k=1.5),
                reason="失败检查含 metric:r2，第 1 次盲试去掉 y 离群值（k=1.5）",
                source="agent",
            )
        if prior == 1:
            return AttributionDecision(
                action="backtrack",
                target_step="compute",
                patch=BacktrackPatch(target_transform="log"),
                reason="失败检查含 metric:r2，第 2 次盲试对数变换",
                source="agent",
            )
        return AttributionDecision(
            action="escalate",
            reason="回溯预算已用完，转人工",
            source="agent",
        )

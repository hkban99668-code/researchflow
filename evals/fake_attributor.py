"""规则版假归因 Agent：只看第一条未通过检查的 `owner_step`，永远回到那一步、不带 patch。

这是 T-011 纯规则基线的"被执行版"：`attribute` 节点对 `validation_failed` 调用
`decide()`，假 Agent 把第一条未通过硬规则检查的 `owner_step` 当作回溯目标。它不带
任何 patch（`patch` 为默认空值），因此回到 `compute` 后 r2 不会变，回溯会被预算
（`attribution.max_total_backtracks`）拦在 2 次，最终以 `budget_exhausted` 结束。

它只处理 `validation_failed`：节点报错走规则层、检索为空不进 `attribute`，都由图
的路由保证，假 Agent 不会遇到这两种情况。
"""

from __future__ import annotations

from researchflow.core.schemas import AttributionDecision


class FakeAttributor:
    """规则版假 Agent：第一条未通过检查归谁，就回到谁。"""

    def decide(self, state: dict) -> AttributionDecision:
        validation = state.get("validation_result")
        failed = [
            check
            for check in (validation.checks if validation is not None else [])
            if not check.passed
        ]
        if not failed:
            return AttributionDecision(
                action="escalate",
                reason="没有未通过的硬规则检查，无法归因",
                source="agent",
            )
        target = failed[0].owner_step
        return AttributionDecision(
            action="backtrack",
            target_step=target,
            reason=f"第一条未通过检查 {failed[0].name} 归 {target}，回到该步",
            source="agent",
        )

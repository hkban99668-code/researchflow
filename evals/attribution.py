"""纯规则归因基线：只看错误码与归属表，不看任何"实际该改哪里"的证据。

规则依次是：

1. `termination.kind == "empty"` → `termination.step`；
2. `error` 不为空 → `error.backtrack_to`，为空时用 `error.step`；
3. 校验未通过 → 第一条未通过检查的 `owner_step`；
4. 其他情况 → `None`。

这个基线**从不**返回 `"human"`。这是刻意的：它是"只看错误码和归属表"的朴素
做法，对照组越朴素，以后接上归因 Agent 后的提升才越可信。跨步骤故障（问题在
上游、症状在下游）与"谁都修不好、应转人工"的用例上，它必然判错——这正是以后
Agent 要赢的地方。
"""

from __future__ import annotations

from typing import Any


def attribute_by_rules(manifest: dict[str, Any]) -> str | None:
    """按四条规则从 manifest 推断"该回到哪个节点"，返回图节点名或 None。"""
    termination = manifest.get("termination")
    if termination and termination.get("kind") == "empty":
        step = termination.get("step")
        return str(step) if step else None

    error = manifest.get("error")
    if error:
        return error.get("backtrack_to") or error.get("step") or None

    validation = manifest.get("validation")
    if validation is not None and not validation.get("passed", False):
        for check in validation.get("checks") or []:
            if not check.get("passed", False):
                owner = check.get("owner_step")
                if owner:
                    return str(owner)

    return None

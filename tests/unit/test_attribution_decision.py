"""T-015 A2：`escalate_to_human(suspected_step=...)` 写进 `AttributionDecision.target_step`。"""

from __future__ import annotations

from researchflow.core.agent_loop import ToolInvocation
from researchflow.core.attribution import _decision_from_terminal_call
from researchflow.core.schemas import ToolCall
from researchflow.core.tools import ToolResult


def _invocation(arguments: str) -> ToolInvocation:
    return ToolInvocation(
        call=ToolCall(id="c1", name="escalate_to_human", arguments=arguments),
        result=ToolResult(tool="escalate_to_human", ok=True, output={"message": "已记录"}),
    )


def test_escalate_suspected_step_writes_target_step():
    """A2：suspected_step 传入时写进 target_step。"""
    decision = _decision_from_terminal_call(
        _invocation('{"reason": "诊断在清洗环节", "suspected_step": "data_prepare"}')
    )
    assert decision.action == "escalate"
    assert decision.target_step == "data_prepare"


def test_escalate_without_suspected_step_is_none():
    """A2：不传 suspected_step 时 target_step 为 None。"""
    decision = _decision_from_terminal_call(_invocation('{"reason": "没诊断出来"}'))
    assert decision.action == "escalate"
    assert decision.target_step is None

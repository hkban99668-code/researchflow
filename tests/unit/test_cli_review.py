"""T-P0-5：HITL 输入必须显式 approve / reject。

第 2 轮达标标准：
1. 空串、no、rejcet 都会被重新询问，最终 reject，input 共被调用 5 次；
2. 连续非法输入达到 max_attempts 后抛 ConfigError（不允许无限循环）；
3. ReviewDecision 只接受 approve / reject（"yes" 之类的旧值必须被拒）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchflow import main
from researchflow.core.errors import ConfigError
from researchflow.core.schemas import ReviewDecision


def test_review_decision_requires_explicit_approve(monkeypatch):
    answers = iter(["", "no", "rejcet", "reject", "缺失率仍偏高"])
    calls: list[str] = []

    def fake_input(*args):
        calls.append(str(args))
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)

    decision = main._ask_decision(False)

    assert decision["decision"] == "reject"
    assert decision["reason"] == "缺失率仍偏高"
    assert len(calls) == 5, "非法输入应重新询问直到给出合法决定，且不得死循环"


def test_review_decision_raises_after_max_attempts(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("builtins.input", lambda *args: calls.append(args) or "maybe")

    with pytest.raises(ConfigError) as exc:
        main._ask_decision(False, max_attempts=3)

    assert "3" in exc.value.message
    assert len(calls) == 3


def test_review_decision_model_rejects_legacy_values():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate({"decision": "yes"})

    assert ReviewDecision.model_validate({"decision": "approve"}).decision == "approve"
    assert ReviewDecision.model_validate({"decision": "reject"}).reason == ""

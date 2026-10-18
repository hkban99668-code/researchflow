"""T-011 A5 与"结果类别推导"的单测：纯规则归因的四条规则各至少一条用例。"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.attribution import attribute_by_rules  # noqa: E402
from evals.run import derive_termination  # noqa: E402

# --- attribute_by_rules：四条规则 ------------------------------------------ #


def test_rule_empty_returns_termination_step():
    """规则 1：termination.kind == empty → termination.step。"""
    manifest = {"termination": {"kind": "empty", "step": "retrieval"}}
    assert attribute_by_rules(manifest) == "retrieval"


def test_rule_error_uses_backtrack_to():
    """规则 2：error 不为空 → error.backtrack_to。"""
    manifest = {
        "termination": {"kind": "error", "step": "data_prepare"},
        "error": {"code": "RF-003", "step": "data_prepare", "backtrack_to": "data_prepare"},
    }
    assert attribute_by_rules(manifest) == "data_prepare"


def test_rule_error_falls_back_to_step():
    """规则 2 兜底：error.backtrack_to 为空 → error.step。"""
    manifest = {
        "error": {"code": "RF-001", "step": "startup", "backtrack_to": None},
    }
    assert attribute_by_rules(manifest) == "startup"


def test_rule_validation_first_failed_owner():
    """规则 3：校验未通过 → 第一条未通过检查的 owner_step。"""
    manifest = {
        "validation": {
            "passed": False,
            "checks": [
                {"name": "metric:r2", "passed": False, "owner_step": "compute"},
                {"name": "section:背景", "passed": False, "owner_step": "writing"},
            ],
        }
    }
    assert attribute_by_rules(manifest) == "compute"


def test_rule_fallback_none():
    """规则 4：其余情况 → None。"""
    assert attribute_by_rules({}) is None
    assert attribute_by_rules({"validation": {"passed": True, "checks": []}}) is None


def test_never_returns_human():
    """基线从不返回 "human"：跨步骤症状与"应转人工"用例上都归到某个节点。"""
    manifests = [
        # 空检索：归到检索节点
        {"termination": {"kind": "empty", "step": "retrieval"}},
        # 数据错误：回溯到 data_prepare
        {"error": {"code": "RF-003", "step": "data_prepare", "backtrack_to": "data_prepare"}},
        # 校验失败：指标归 compute
        {
            "validation": {
                "passed": False,
                "checks": [{"name": "metric:r2", "passed": False, "owner_step": "compute"}],
            }
        },
        # 校验失败：章节归 writing
        {
            "validation": {
                "passed": False,
                "checks": [{"name": "section:背景", "passed": False, "owner_step": "writing"}],
            }
        },
    ]
    for manifest in manifests:
        assert attribute_by_rules(manifest) != "human"


# --- derive_termination ---------------------------------------------------- #


def test_derive_termination_ok_when_validation_passed():
    assert derive_termination({"termination": None, "validation": {"passed": True}}) == "ok"


def test_derive_termination_validation_failed():
    assert derive_termination({"validation": {"passed": False}}) == "validation_failed"


def test_derive_termination_takes_termination_kind():
    assert derive_termination({"termination": {"kind": "error"}}) == "error"
    assert derive_termination({"termination": {"kind": "empty"}}) == "empty"
    assert derive_termination({"termination": {"kind": "rejected"}}) == "rejected"


def test_derive_termination_no_termination_no_validation():
    assert derive_termination({}) == "ok"

"""T-029 引用评测（Fake 层）回归：拦截率 100%、误杀率 0%、逐条判定符合预期。

对应验收标准 A5（报告写入 evals/citation/reports/ 由命令行运行负责，这里只断言
`run_fake()` 的确定性结果）。评测与 .env 隔离由 conftest 与 evals.citation.run 的
模块级代码共同保证，不联网。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `import evals` 可用（与 test_eval_regression 同法）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.citation import run as citation_run  # noqa: E402


def test_citation_eval_fake_summary():
    """A5：拦截率 100%、误杀率 0%、幻觉引用率等于合计 invalid/total。"""
    report = citation_run.run_fake()
    summary = report["summary"]

    assert summary["n_cases"] >= 12, "用例至少 12 条"
    assert summary["blocked"]["n"] > 0
    assert summary["blocked"]["blocked_rate"] == 1.0, "编造/注入用例必须全部被 ⑦ 拦下"
    assert summary["false_kill"]["n"] > 0
    assert summary["false_kill"]["false_kill_rate"] == 0.0, "正常用例不得被误杀"
    hallucination = summary["hallucination"]
    assert hallucination["total"] == sum(r["citation_stats"]["total"] for r in report["cases"])
    assert hallucination["invalid"] == sum(r["citation_stats"]["invalid"] for r in report["cases"])
    assert summary["all_case_judgments_ok"] is True


def test_citation_eval_groups_cover_required_scenarios():
    """用例集覆盖任务卡要求的场景：正常/空列表/编造/重复/claim 空/注入/文献为空。"""
    cases = {case["id"]: case for case in citation_run.load_cases()}

    normal_ids = {c["id"] for c in cases.values() if c["group"] == "normal"}
    assert any("empty_citations" in cid for cid in normal_ids), "缺引用空列表用例"
    assert any("no_literature" in cid for cid in normal_ids), "缺文献为空用例"
    assert any("duplicate" in cid for cid in normal_ids), "缺重复引用用例"
    assert any("injection" in cid for cid in normal_ids), "缺摘要带注入但模型未被骗用例"
    assert any(c["group"] == "fabricated" for c in cases.values()), "缺编造 ref_id 用例"
    assert any(c["group"] == "injection" and "deceived" in c["id"] for c in cases.values()), (
        "缺被注入骗到的用例"
    )
    assert any(c["group"] == "edge" for c in cases.values()), "缺 claim 为空用例"


def test_citation_eval_case_judgments_individually():
    """逐条断言：拦截组全失败、正常组全通过、edge 组按 expected 判。"""
    report = citation_run.run_fake()
    by_group: dict[str, list[dict]] = {}
    for record in report["cases"]:
        by_group.setdefault(record["group"], []).append(record)

    for record in by_group.get("fabricated", []) + by_group.get("injection", []):
        assert record["actual_passed"] is False, record["id"]
        assert record["actual_invalid"] == record["expected_invalid"], record["id"]
    for record in by_group.get("normal", []):
        assert record["actual_passed"] is True, record["id"]
        assert record["actual_invalid"] == 0, record["id"]


def test_citation_eval_real_layer_preview_gates(capsys):
    """A7：`--layer real` 不带 --confirm-real-calls 只预演（退出码 2），不发请求。"""
    exit_code = citation_run.main(["--layer", "real"])

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "真实请求次数: 0" in out

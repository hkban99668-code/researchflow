"""T-011 A4/A6/A7/A8（更新至 T-018）：Fake 层回归测试。

跑一遍 Fake 层（朴素规则 + 聪明规则两个基线，各 58 条完整链路），把确定性字段与
`evals/baseline_fake.json` 比较，任何不同即失败。耗时和时间戳不比较。整个过程让
`httpx.post/get` 一调用就抛异常，证明评测不联网。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import httpx
import pytest

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import run as evals_run  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "evals" / "baseline_fake.json"


def _boom(*args, **kwargs):
    raise AssertionError("评测不应发出任何网络请求")


def _judgment_one(cases: list[dict], summary: dict) -> dict:
    """一个基线的确定性判定字段。"""
    return {
        "cases": [
            {
                "id": c["id"],
                "termination_ok": c["termination_ok"],
                "agent_target": c["agent_target"],
                "attribution_ok": c["attribution_ok"],
                "recovered": c["recovered"],
                "backtracks": c["backtracks"],
            }
            for c in cases
        ],
        "summary": {
            "termination_ok": summary["termination_ok"],
            "attribution_ok": summary["attribution_ok"],
            "recovered": summary["recovered"],
            "avg_backtracks": summary["avg_backtracks"],
            "invalid_backtrack_rate": summary["invalid_backtrack_rate"],
            "fix_ok_rate": summary["fix_ok_rate"],
        },
    }


def _judgment_subset(report: dict) -> dict:
    """三个基线（朴素 + 聪明 + 搜索）的确定性判定字段，与基线比较只看这些。"""
    return {
        "naive": _judgment_one(report["cases"], report["summary"]),
        "smart": _judgment_one(report["smart"]["cases"], report["smart"]["summary"]),
        "search": _judgment_one(report["search"]["cases"], report["search"]["summary"]),
    }


def _strip_non_deterministic(report: dict) -> dict:
    """去掉 generated_at 与耗时字段，剩下的应逐字节相等（A4）。"""
    out = copy.deepcopy(report)
    out.pop("generated_at", None)
    out["summary"].pop("latency_ms", None)
    out["smart"]["summary"].pop("latency_ms", None)
    out["search"]["summary"].pop("latency_ms", None)
    for case in out["cases"] + out["smart"]["cases"] + out["search"]["cases"]:
        case.pop("latency_ms", None)
    return out


@pytest.fixture(scope="module")
def eval_report():
    """跑一遍 Fake 层（带 httpx 守卫），供各测试复用，避免重复跑 116 条链路。"""
    original_post = httpx.post
    original_get = httpx.get
    httpx.post = _boom
    httpx.get = _boom
    try:
        return evals_run.run_fake()
    finally:
        httpx.post = original_post
        httpx.get = original_get


def test_regression_matches_baseline(eval_report):
    """A4/A6：两个基线的确定性字段与 baseline_fake.json 一致。"""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert _judgment_subset(eval_report) == _judgment_subset(baseline)


def test_eval_makes_no_network_calls(eval_report):
    """A8：整个评测过程不联网（httpx 一调用就抛，上面 fixture 里已挂守卫）。"""
    assert eval_report["summary"]["n_cases"] >= 36


def test_cross_step_attribution_below_single_step(eval_report):
    """A7：朴素基线的 cross_step 归因准确率明显低于 single_step。"""
    attribution = eval_report["summary"]["attribution_ok"]
    assert attribution["single_step"] > 0.8
    assert attribution["cross_step"] < attribution["single_step"]


def test_smart_baseline_recovers_cross_step_only(eval_report):
    """A4：聪明基线能修好大部分跨步骤故障（离群值/指数目标），但 human 仍恢复不了。

    k_choice_trap（需要 k=3）会让聪明基线掉进陷阱，因此 cross_step 的 recovered
    不再是 100%（这是 T-019 故意制造的区分度）。
    """
    smart = eval_report["smart"]["summary"]
    assert smart["recovered"]["cross_step"] >= 0.8
    assert smart["recovered"]["cross_step"] < 1.0
    assert smart["recovered"]["human"] == 0.0


def test_invalid_backtrack_rate_computed(eval_report):
    """A4：两个基线的无效回溯率都算得出来，且聪明基线 < 朴素基线（它只浪费 human 组）。"""
    naive_rate = eval_report["summary"]["invalid_backtrack_rate"]
    smart_rate = eval_report["smart"]["summary"]["invalid_backtrack_rate"]
    assert naive_rate > 0.0
    assert smart_rate > 0.0
    assert smart_rate < naive_rate


def test_deterministic_across_runs(monkeypatch):
    """A4：连跑两次，除 generated_at 与耗时字段外完全相同。"""
    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr(httpx, "get", _boom)
    first = evals_run.run_fake()
    second = evals_run.run_fake()
    assert _strip_non_deterministic(first) == _strip_non_deterministic(second)

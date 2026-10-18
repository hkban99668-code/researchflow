"""T-020 A3/A5：诊断搜索规则基线。

搜索规则用和 Agent 同一个 `feature_target_relation` 工具（不经过 LLM），在候选修法里
选诊断 R² 最高、且行数满足 min_rows 的一个回溯；最高诊断 R² 低于 ⑦ 阈值或第二次进入
attribute 时转人工。用 `evals/faults.py` 的真实故障数据做集成验证，用受控的工具输出
对 `_decide_from_relation` 做单元验证。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import faults  # noqa: E402
from evals.search_rule_attributor import (  # noqa: E402
    SearchRuleAttributor,
    _decide_from_relation,
)

from researchflow.core.config import RunContext, load_settings, sha256_file  # noqa: E402
from researchflow.core.schemas import (  # noqa: E402
    AttributionDecision,
    AttributionRecord,
    CleanedDataRef,
    ParamSet,
)
from researchflow.core.tools import ToolContext  # noqa: E402
from researchflow.tools.attribution_tools import build_attribution_registry  # noqa: E402

FEATURES = ["x1", "x2", "x3"]
THRESHOLD = 0.8


def _ctx_and_state(tmp_path, project_root, frame, prior_backtracks: int = 0):
    """构造搜索规则所需的 ctx 与 state：清洗数据落盘 + param_set +（可选）已回溯记录。"""
    settings = load_settings(project_root=project_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "cleaned_candidate.csv"
    frame.to_csv(csv_path, index=False)
    ref = CleanedDataRef(
        path="cleaned_candidate.csv",
        checksum=sha256_file(csv_path),
        rows=len(frame),
        columns=list(frame.columns),
    )
    attributions = [
        AttributionRecord(
            attempt=i + 1,
            trigger="validation_failed",
            decision=AttributionDecision(
                action="backtrack", target_step="data_prepare", reason="上次回溯", source="agent"
            ),
        )
        for i in range(prior_backtracks)
    ]
    state = {
        "cleaned_data": ref,
        "param_set": ParamSet(
            algorithm="linear_regression",
            params={"target": "y", "features": list(FEATURES)},
            seed=42,
        ),
        "attributions": attributions,
    }
    ctx = RunContext(settings=settings, run_dir=run_dir)
    return SearchRuleAttributor(ctx), state


def _relation_out(
    *,
    min_rows: int = 30,
    k15_rows: int = 30,
    k15_r2: float = 0.5,
    k3_rows: int = 30,
    k3_r2: float = 0.5,
    log_applicable: bool = False,
    log_r2: float | None = None,
    without: dict | None = None,
) -> dict:
    """构造 `feature_target_relation` 输出形状的受控 dict。"""
    return {
        "min_rows": min_rows,
        "robust_by_k": {
            "1.5": {"rows_after": k15_rows, "r2": k15_r2},
            "3": {"rows_after": k3_rows, "r2": k3_r2},
        },
        "log_target": {"applicable": log_applicable, "r2": log_r2, "note": ""},
        "r2_without_feature": without or {},
    }


# --- 集成验证（真实故障数据，走 decide() → 工具 → 选候选） ----------------------


def test_picks_log_for_exp_target(tmp_path, project_root):
    """A5：exp_target 上 log 的诊断 R² 最高，搜索规则选 compute + log。"""
    mutated = faults.apply_fault("exp_target", faults.load_sample(), 700, {"noise_scale": 0.05})
    attributor, state = _ctx_and_state(tmp_path, project_root, mutated)

    decision = attributor.decide(state)

    assert decision.action == "backtrack"
    assert decision.target_step == "compute"
    assert decision.patch.target_transform == "log"


def test_k_choice_trap_escalates_not_picks_k15(tmp_path, project_root):
    """A5：k_choice_trap 上 k=1.5 因 rows_after < min_rows 被排除；k=3 的诊断 R² 又低于
    0.8 阈值，于是转人工——绝不会错误地回 data_prepare 用 k=1.5 删过头。"""
    mutated = faults.apply_fault(
        "k_choice_trap", faults.load_sample(), 43, {"rows": 36, "n_extreme": 4, "n_mild": 3}
    )
    attributor, state = _ctx_and_state(tmp_path, project_root, mutated)

    decision = attributor.decide(state)

    assert decision.action == "escalate"
    assert not (
        decision.action == "backtrack"
        and decision.patch.y_outlier_strategy == "iqr"
        and decision.patch.y_outlier_k == 1.5
    )


def test_escalates_when_best_r2_below_threshold(tmp_path, project_root):
    """A5：y 打乱后所有候选诊断 R² 都低于阈值，搜索规则转人工。"""
    mutated = faults.apply_fault("y_shuffled", faults.load_sample(), 301, {})
    attributor, state = _ctx_and_state(tmp_path, project_root, mutated)

    decision = attributor.decide(state)

    assert decision.action == "escalate"


def test_escalates_on_second_entry(tmp_path, project_root):
    """A5：第二次进入 attribute（说明上次没修好）转人工，不再选修法。"""
    mutated = faults.apply_fault("exp_target", faults.load_sample(), 700, {"noise_scale": 0.05})
    attributor, state = _ctx_and_state(tmp_path, project_root, mutated, prior_backtracks=1)

    decision = attributor.decide(state)

    assert decision.action == "escalate"


# --- A3：k_choice_trap 的 rows_after 证据（工具输出） ---------------------------


def test_k_choice_trap_rows_after_bridges_min_rows(tmp_path, project_root):
    """A3：k_choice_trap 上 robust_by_k["1.5"].rows_after < min_rows <= robust_by_k["3"].rows_after。"""
    mutated = faults.apply_fault(
        "k_choice_trap", faults.load_sample(), 43, {"rows": 36, "n_extreme": 4, "n_mild": 3}
    )
    attributor, state = _ctx_and_state(tmp_path, project_root, mutated)
    registry = build_attribution_registry(state, attributor.ctx)
    tool_ctx = ToolContext(
        workspace_root=attributor.ctx.settings.project_root, settings=attributor.ctx.settings
    )

    result = registry.invoke("feature_target_relation", {}, tool_ctx)

    assert result.ok
    out = result.output
    min_rows = out["min_rows"]
    assert out["robust_by_k"]["1.5"]["rows_after"] < min_rows
    assert min_rows <= out["robust_by_k"]["3"]["rows_after"]


# --- 单元验证（受控工具输出，喂给纯函数 _decide_from_relation） -----------------


def test_picks_highest_r2_candidate():
    """A5：选诊断 R² 最高的候选（log 最高 → compute + log）。"""
    out = _relation_out(
        k15_r2=0.6,
        k3_r2=0.7,
        log_applicable=True,
        log_r2=0.95,
        without={"x1": 0.5, "x2": 0.55, "x3": 0.52},
    )

    decision = _decide_from_relation(out, THRESHOLD)

    assert decision.action == "backtrack"
    assert decision.target_step == "compute"
    assert decision.patch.target_transform == "log"


def test_excludes_insufficient_rows_candidate():
    """A5：k=1.5 诊断 R² 最高但 rows_after < min_rows，被排除；选 k=3。"""
    out = _relation_out(
        k15_rows=29,
        k15_r2=0.99,
        k3_rows=30,
        k3_r2=0.85,
        log_applicable=False,
        without={"x1": 0.2, "x2": 0.25, "x3": 0.3},
    )

    decision = _decide_from_relation(out, THRESHOLD)

    assert decision.action == "backtrack"
    assert decision.target_step == "data_prepare"
    assert decision.patch.y_outlier_strategy == "iqr"
    assert decision.patch.y_outlier_k == 3.0


def test_escalates_when_no_candidate_above_threshold():
    """A5：最高诊断 R² 低于阈值 → 转人工。"""
    out = _relation_out(
        k15_r2=0.4,
        k3_r2=0.5,
        log_applicable=False,
        without={"x1": 0.3, "x2": 0.35, "x3": 0.32},
    )

    decision = _decide_from_relation(out, THRESHOLD)

    assert decision.action == "escalate"


def test_escalates_when_no_valid_candidate():
    """A5：所有 iqr 候选行数都不足、log 不可用、无特征可排除 → 转人工。"""
    out = _relation_out(k15_rows=29, k3_rows=29, log_applicable=False, without={})

    decision = _decide_from_relation(out, THRESHOLD)

    assert decision.action == "escalate"

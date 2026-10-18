"""Eval 批量运行与报告：在 Fake 层批量跑完整链路并统计。

用法：

    uv run python -m evals.run --layer fake

流程：读 `evals/cases.yaml` → 逐条生成变异 CSV 与课题 YAML → 用
`researchflow.main.main([...])`（`--mode fake --auto-approve`）跑一次完整链路，
`runs_dir` 指向临时目录 → 读 manifest → 判定 → 汇总 → 写 JSON 与 Markdown 报告。

本运行器**不调用任何真实模型、不联网**。Fake 层的数字是确定的，进 CI 做回归。

归因口径（T-014）：Fake 层的 `attributor` 用一个规则版假 Agent（`fake_attributor.py`，
只看第一条未通过检查的 `owner_step`、永远回到那一步、不带 patch），替换 T-011 的
离线纯规则基线，现在真正在图里被执行。归因分开看"诊断"和"动作"两件事：

- `agent_target`：归因机制给出的目标（从 manifest 的 `attributions` 取最后一次决策的
  `target_step`；没有归因记录——如检索为空——时为 None）。
- `recovered`：动作是否真的修好（最终 ⑦ 是否通过）。
- `attribution_ok`：期望 `target_step == "human"` 时，决策是 `escalate` 即算对；否则
  看决策的 `target_step`（不论动作是回溯还是转人工）是否等于期望。这样单步故障
  （诊断对、但规则层没有可调参数所以转人工）仍然算归因正确，不会被误判。
- **检索为空不计入归因准确率**：期望 `target_step == "retrieval"` 的用例本来就不进
  `attribute`、不归这个 Agent 管，其 `attribution_ok` 记为 `n/a`（None），并从
  `summary.attribution_ok` 的分子分母里排除。
"""

from __future__ import annotations

from . import _env

# 运行器与 .env 隔离（T-031 起统一到 evals._env）：必须在 import researchflow
# 之前调用——保存 shell 里显式给的 RF_LLM_*/RF_EMBED_*（真实层再写回，命令行
# 优先于 .env），设 RF_LOAD_DOTENV=0，再清掉这些变量（开发者 shell 里 export
# 的 Key 也不影响评测的 Fake 层）。
_env.isolate_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

from researchflow import main as main_mod  # noqa: E402
from researchflow.core.agent_loop import LoopBudget  # noqa: E402
from researchflow.core.attribution import AgentAttributor  # noqa: E402
from researchflow.core.config import Settings, load_settings  # noqa: E402
from researchflow.core.llm import add_usage, build_llm_client  # noqa: E402
from researchflow.core.schemas import LLMUsage  # noqa: E402

from . import (  # noqa: E402
    fake_attributor,
    faults,
    label_check,
    search_rule_attributor,
    smart_rule_attributor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals" / "cases.yaml"
REPORTS_DIR = PROJECT_ROOT / "evals" / "reports"
DEMO_TOPIC = PROJECT_ROOT / "examples" / "demo_topic.yaml"

GROUPS = ("clean", "single_step", "cross_step", "human")
ATTRIBUTION_GROUPS = ("single_step", "cross_step", "human")


def load_cases() -> list[dict[str, Any]]:
    """读取并校验用例清单。"""
    raw = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError("cases.yaml 顶层必须是列表")
    return raw


def cases_sha256() -> str:
    """`cases.yaml` 的 sha256，进报告用于锁定用例集。"""
    return hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()


def git_commit() -> str | None:
    """当前 HEAD；不在 git 仓库里或 git 不可用时返回 None。"""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def derive_termination(manifest: dict[str, Any]) -> str:
    """从 manifest 推出"结果类别"，用于与 `expected.termination` 比较。

    不是 state 里的 `termination.kind`：有 `termination` 时取它的 `kind`；
    没有 `termination`、但校验报告未通过时为 `validation_failed`；都没有时为 `ok`。
    """
    termination = manifest.get("termination")
    if termination:
        kind = termination.get("kind")
        return str(kind) if kind else "error"
    validation = manifest.get("validation")
    if validation is not None and not validation.get("passed", False):
        return "validation_failed"
    return "ok"


def _settings_with_runs_dir(runs_dir: Path) -> Any:
    """构造一次运行的 Settings：`runs_dir` 指向临时目录，其余沿用真实配置。

    `main.run` 用模块级 `load_settings()`（无参），只能靠猴子补丁把 `runs_dir`
    指到临时目录。`settings.yaml` 与 `.env` 里都没有 `runs_dir` 的环境变量可覆盖。
    """
    settings = load_settings()
    settings.runs_dir = runs_dir
    return settings


def _single_run_dir(runs_dir: Path) -> Path:
    """取出本次运行生成的唯一 run 子目录。"""
    subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(f"预期 runs_dir 下恰好一个 run 目录，实际 {len(subdirs)} 个")
    return subdirs[0]


def _materialize(case: dict[str, Any], case_dir: Path) -> Path:
    """按用例生成变异 CSV 与课题 YAML，返回课题文件路径。"""
    base_topic = yaml.safe_load(DEMO_TOPIC.read_text(encoding="utf-8")) or {}
    fault = case["fault"]
    params = dict(case.get("params") or {})

    if fault in faults.DATA_FAULTS:
        frame = faults.load_sample()
        mutated = faults.apply_fault(fault, frame, int(case["seed"]), params)
        csv_path = case_dir / "dataset.csv"
        mutated.to_csv(csv_path, index=False)
        topic = deepcopy(base_topic)
        topic["dataset"]["path"] = str(csv_path)
    elif fault in faults.TOPIC_FAULTS:
        topic = faults.apply_topic_fault(fault, base_topic, params)
        topic["dataset"]["path"] = str(faults.SAMPLE_CSV)
    else:
        raise KeyError(f"未登记的故障: {fault!r}")

    topic_path = case_dir / "topic.yaml"
    topic_path.write_text(
        yaml.safe_dump(topic, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return topic_path


def _termination_ok(
    expected: dict[str, Any], actual_termination: str, manifest: dict[str, Any]
) -> bool:
    """实际终止结果与期望一致（含错误码）。"""
    if actual_termination != expected["termination"]:
        return False
    expected_code = expected.get("error_code")
    if expected_code is not None:
        error = manifest.get("error") or {}
        if error.get("code") != expected_code:
            return False
    return True


def _last_decision(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """取 manifest 里最后一次归因决策；没有归因记录（如检索为空）时为 None。"""
    attributions = manifest.get("attributions") or []
    if not attributions:
        return None
    return attributions[-1].get("decision") or None


def _agent_failed_of(manifest: dict[str, Any]) -> bool:
    """本次运行是否发生过"Agent 调用失败降级"（T-027）。

    看 `AttributionRecord.agent_failed`——attribute 节点在 attributor.decide 抛错
    降级为转人工时显式标记；评测据此把这类用例单列并排除出归因准确率与恢复率的
    分母，不与"Agent 判断转人工"混在一起。
    """
    return any(record.get("agent_failed") for record in manifest.get("attributions") or [])


def _attribution_ok(expected_target: str | None, decision: dict[str, Any] | None) -> bool:
    """归因是否判对：human → escalate；否则看 target_step 是否等于期望（见模块 docstring）。"""
    if decision is None:
        return False
    if expected_target == "human":
        return decision.get("action") == "escalate"
    return decision.get("target_step") == expected_target


def _count_backtracks(manifest: dict[str, Any]) -> int:
    """数 manifest 里回溯动作的次数（供无效回溯率用）。"""
    return sum(1 for record in _all_decisions(manifest) if record.get("action") == "backtrack")


def _all_decisions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """取 manifest 里**全部**归因决策（不只最后一次），供报告记录与修法正确率用。"""
    attributions = manifest.get("attributions") or []
    decisions: list[dict[str, Any]] = []
    for record in attributions:
        decision = record.get("decision") or {}
        decisions.append(
            {
                "action": decision.get("action"),
                "target_step": decision.get("target_step"),
                "patch": decision.get("patch") or {},
                "source": decision.get("source"),
            }
        )
    return decisions


def _feasible_ops(case: dict[str, Any]) -> set[str]:
    """该用例在 label_check 里**最短**可行组合中出现的修法名集合（供修法正确率用）。

    只取最短（最少修法次数）的可行组合：把"某修法出现在某个更长的组合里"也算作
    正确，会让修法正确率失真（例如 exp_target 上 iqr k=1.5 单独并不修，只是碰巧
    出现在 [iqr, log] 这个 2 次组合里）。
    """
    result = label_check.check_fixable(case)
    if not result["shortest"]:
        return set()
    min_len = len(result["shortest"])
    ops: set[str] = set()
    for combo in result["feasible"]:
        if len(combo) == min_len:
            ops.update(combo)
    return ops


def _patch_summary(patch: dict[str, Any]) -> str:
    """把 patch 压成一句简短描述，用于报告里逐条决策的展示。"""
    parts: list[str] = []
    if patch.get("y_outlier_strategy"):
        parts.append(f"iqr_k={patch.get('y_outlier_k')}")
    if patch.get("target_transform"):
        parts.append("log")
    if patch.get("exclude_features"):
        parts.append("exclude:" + ",".join(patch["exclude_features"]))
    return " ".join(parts)


def _run_one(
    case: dict[str, Any], tmp_root: Path, attributor_factory: Callable[[Any], Any]
) -> dict[str, Any]:
    """跑一条用例，返回判定记录。`attributor_factory(ctx)` 构造本基线用的归因器。"""
    case_dir = Path(tempfile.mkdtemp(prefix=f"{case['id']}-", dir=tmp_root))
    topic_path = _materialize(case, case_dir)
    runs_dir = case_dir / "runs"
    settings = _settings_with_runs_dir(runs_dir)

    original = main_mod.load_settings
    original_attributor = main_mod._build_attributor
    main_mod.load_settings = lambda *a, **k: settings
    # Fake 层用规则版假 Agent，替换 main.py 默认构造的 AgentAttributor（不调真实模型）。
    main_mod._build_attributor = attributor_factory  # type: ignore[assignment]
    try:
        with redirect_stdout(io.StringIO()):
            started = time.perf_counter()
            main_mod.main(["run", "--topic", str(topic_path), "--mode", "fake", "--auto-approve"])
            latency_ms = int((time.perf_counter() - started) * 1000)
    finally:
        main_mod.load_settings = original
        main_mod._build_attributor = original_attributor

    run_dir = _single_run_dir(runs_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    expected = case["expected"]
    actual_termination = derive_termination(manifest)
    decision = _last_decision(manifest)
    agent_target = decision.get("target_step") if decision else None
    # 检索为空的用例不进 attribute，不归归因机制管，attribution_ok 记为 n/a（None）。
    attribution_ok = (
        None
        if expected["target_step"] == "retrieval"
        else _attribution_ok(expected["target_step"], decision)
    )
    return {
        "id": case["id"],
        "fault": case["fault"],
        "group": case["group"],
        "expected": {
            "termination": expected["termination"],
            "error_code": expected.get("error_code"),
            "target_step": expected["target_step"],
        },
        "actual_termination": actual_termination,
        "termination_ok": _termination_ok(expected, actual_termination, manifest),
        "agent_target": agent_target,
        "attribution_ok": attribution_ok,
        "agent_failed": _agent_failed_of(manifest),
        "recovered": actual_termination == "ok",
        "backtracks": _count_backtracks(manifest),
        "decisions": _all_decisions(manifest),
        "tokens": manifest["llm_usage"]["total"]["total_tokens"],
        "latency_ms": latency_ms,
    }


def _ratio(records: list[dict[str, Any]], field: str) -> float:
    if not records:
        return 0.0
    return round(sum(1 for r in records if r[field]) / len(records), 4)


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    frac = rank - low
    return int(values[low] + (values[high] - values[low]) * frac)


def _attribution_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """归因准确率统计的行：排除 clean 组、期望 target_step 为 retrieval、以及 Agent
    调用失败的用例（T-027）。

    检索为空的用例不进 `attribute`（gu 确认的设计点），本就不该由归因机制判，
    算作"判错"会让人误以为规则基线在单步故障上出错，所以排除出分子分母。
    Agent 调用失败降级成转人工的用例同理：它衡量的是"Agent 坏了"而不是"Agent
    判断转人工"，单列计数（见 `_summarize` 的 `agent_failed`），不进分母。
    """
    return [
        r
        for r in records
        if r["group"] != "clean"
        and r["expected"]["target_step"] != "retrieval"
        and not r.get("agent_failed")
    ]


def _agent_failed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """发生过 Agent 调用失败降级的用例（T-027），报告单列。"""
    return [r for r in records if r.get("agent_failed")]


def _summarize(
    records: list[dict[str, Any]], feasible_by_id: dict[str, set[str]] | None = None
) -> dict[str, Any]:
    termination_ok: dict[str, float] = {"overall": _ratio(records, "termination_ok")}
    for group in GROUPS:
        termination_ok[group] = _ratio(
            [r for r in records if r["group"] == group], "termination_ok"
        )

    attr_rows = _attribution_rows(records)
    attribution_ok: dict[str, float] = {"overall": _ratio(attr_rows, "attribution_ok")}
    for group in ATTRIBUTION_GROUPS:
        attribution_ok[group] = _ratio(
            [r for r in attr_rows if r["group"] == group], "attribution_ok"
        )

    # 恢复率的分母同样排除 Agent 调用失败的用例（T-027）：失败降级的转人工不是
    # "动作没修好"，不应拉低恢复率。
    recovered_records = [r for r in records if not r.get("agent_failed")]
    recovered: dict[str, float] = {"overall": _ratio(recovered_records, "recovered")}
    for group in GROUPS:
        recovered[group] = _ratio(
            [r for r in recovered_records if r["group"] == group], "recovered"
        )

    agent_failed = _agent_failed_records(records)
    avg_backtracks: dict[str, float] = {"overall": _avg(records, "backtracks")}
    for group in GROUPS:
        avg_backtracks[group] = _avg([r for r in records if r["group"] == group], "backtracks")

    latencies = sorted(r["latency_ms"] for r in records)
    tokens = [r["tokens"] for r in records]
    return {
        "n_cases": len(records),
        "agent_failed": {
            "n": len(agent_failed),
            "ids": [r["id"] for r in agent_failed],
        },
        "termination_ok": termination_ok,
        "attribution_ok": attribution_ok,
        "recovered": recovered,
        "avg_backtracks": avg_backtracks,
        "invalid_backtrack_rate": _invalid_backtrack_rate(records),
        "fix_ok_rate": _fix_ok_rate(records, feasible_by_id),
        "tokens": {"avg": round(sum(tokens) / len(tokens), 1) if tokens else 0.0},
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
    }


def _avg(records: list[dict[str, Any]], field: str) -> float:
    """某字段的平均值；空列表返回 0.0。"""
    if not records:
        return 0.0
    return round(sum(r[field] for r in records) / len(records), 4)


def _invalid_backtrack_rate(records: list[dict[str, Any]]) -> float:
    """无效回溯率：回溯了但没有修好的次数 / 回溯总次数。"""
    total = sum(r["backtracks"] for r in records)
    if total == 0:
        return 0.0
    invalid = sum(r["backtracks"] for r in records if not r["recovered"])
    return round(invalid / total, 4)


def _fix_ok_rate(
    records: list[dict[str, Any]], feasible_by_id: dict[str, set[str]] | None = None
) -> float:
    """修法正确率：回溯时用的修法落在 label_check 可行组合里的次数 / 回溯总次数。

    回溯但没带任何可映射修法（如朴素规则的空 patch）算"不正确"。`feasible_by_id`
    为 None 时不计算（返回 0.0）。
    """
    if not feasible_by_id:
        return 0.0
    total = 0
    correct = 0
    for record in records:
        feasible = feasible_by_id.get(record["id"], set())
        for decision in record["decisions"]:
            if decision.get("action") != "backtrack":
                continue
            op = label_check.expected_fix_op(decision)
            total += 1
            if op is not None and op in feasible:
                correct += 1
    return round(correct / total, 4) if total else 0.0


def run_fake(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """跑一遍 Fake 层全部用例（朴素规则 + 聪明规则两个基线），返回报告字典（不写文件）。"""
    cases = cases if cases is not None else load_cases()
    # 修法正确率用：每条用例的可行修法集合（label_check 机器核验一次）。
    feasible_by_id = {
        case["id"]: _feasible_ops(case)
        for case in cases
        if case["group"] in ("cross_step", "human")
    }
    tmp_root = Path(tempfile.mkdtemp(prefix="rf-eval-"))
    try:
        naive_records = [
            _run_one(case, tmp_root, lambda ctx: fake_attributor.FakeAttributor()) for case in cases
        ]
        smart_records = [
            _run_one(case, tmp_root, lambda ctx: smart_rule_attributor.SmartRuleAttributor())
            for case in cases
        ]
        search_records = [
            _run_one(case, tmp_root, lambda ctx: search_rule_attributor.SearchRuleAttributor(ctx))
            for case in cases
        ]
    finally:
        # 判定的字段已经从 manifest 抽出，运行目录可以整体清掉，不留临时文件。
        shutil.rmtree(tmp_root, ignore_errors=True)
    return {
        "layer": "fake",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "cases_sha256": cases_sha256(),
        "cases": naive_records,
        "summary": _summarize(naive_records, feasible_by_id),
        "smart": {
            "cases": smart_records,
            "summary": _summarize(smart_records, feasible_by_id),
        },
        "search": {
            "cases": search_records,
            "summary": _summarize(search_records, feasible_by_id),
        },
    }


def _filter_by_split(cases: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """按 split 过滤用例；`all` 返回全部。"""
    if split == "all":
        return list(cases)
    return [case for case in cases if case.get("split") == split]


def _dump_invocation(invocation: Any) -> dict[str, Any]:
    """把一次工具调用转成报告字段：工具名、参数、ok、错误码。"""
    call = invocation.call
    result = invocation.result
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "name": call.name,
        "arguments": arguments,
        "ok": result.ok,
        "error_code": result.error.code if result.error else None,
    }


class _CapturingAgentAttributor(AgentAttributor):
    """包一层 AgentAttributor：把每次 decide() 的工具调用、用量、耗时累计进 capture。"""

    def __init__(self, llm: Any, ctx: Any, budget: Any, system_prompt: str, capture: dict) -> None:
        super().__init__(llm=llm, ctx=ctx, budget=budget, system_prompt=system_prompt)
        self._capture = capture

    def decide(self, state: dict) -> Any:
        started = time.perf_counter()
        decision = super().decide(state)
        self._capture["latency_ms"] += int((time.perf_counter() - started) * 1000)
        usage = self.last_usage or LLMUsage(attempts=0)
        self._capture["usage"] = add_usage(self._capture["usage"], usage)
        for invocation in self.last_invocations:
            self._capture["tool_calls"].append(_dump_invocation(invocation))
        return decision


def _run_one_real(
    case: dict[str, Any],
    tmp_root: Path,
    real_client: Any,
    model: str,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    """真实层跑一条用例：流水线用 Fake，只有归因 Agent 用真实客户端。"""
    case_dir = Path(tempfile.mkdtemp(prefix=f"{case['id']}-", dir=tmp_root))
    topic_path = _materialize(case, case_dir)
    runs_dir = case_dir / "runs"
    settings = _settings_with_runs_dir(runs_dir)
    settings.llm.mode = "fake"
    settings.llm.api_key = None  # ②⑥ 用 Fake；Key 只属于 real_client

    capture: dict[str, Any] = {"tool_calls": [], "usage": LLMUsage(attempts=0), "latency_ms": 0}

    original = main_mod.load_settings
    original_attributor = main_mod._build_attributor
    main_mod.load_settings = lambda *a, **k: settings

    def build_attributor(ctx: Any) -> AgentAttributor:
        prompt_path = ctx.settings.resolve_path(
            ctx.settings.raw.get("paths", {}).get(
                "attribution_prompt", "configs/prompts/attribution_system.md"
            )
        )
        return _CapturingAgentAttributor(
            llm=real_client,
            ctx=ctx,
            budget=LoopBudget(),
            system_prompt=prompt_path.read_text(encoding="utf-8"),
            capture=capture,
        )

    main_mod._build_attributor = build_attributor
    try:
        with redirect_stdout(io.StringIO()):
            main_mod.main(["run", "--topic", str(topic_path), "--mode", "fake", "--auto-approve"])
    finally:
        main_mod.load_settings = original
        main_mod._build_attributor = original_attributor

    run_dir = _single_run_dir(runs_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    expected = case["expected"]
    actual_termination = derive_termination(manifest)
    decision = _last_decision(manifest)
    attribution_ok = (
        None
        if expected["target_step"] == "retrieval"
        else _attribution_ok(expected["target_step"], decision)
    )
    return {
        "id": case["id"],
        "fault": case["fault"],
        "group": case["group"],
        "split": case["split"],
        "expected": {
            "termination": expected["termination"],
            "error_code": expected.get("error_code"),
            "target_step": expected["target_step"],
        },
        "actual_termination": actual_termination,
        "termination_ok": _termination_ok(expected, actual_termination, manifest),
        "decision": decision,
        "decisions": _all_decisions(manifest),
        "backtracks": _count_backtracks(manifest),
        "agent_failed": _agent_failed_of(manifest),
        "tool_calls": capture["tool_calls"],
        "agent_tokens": capture["usage"].total_tokens,
        "agent_latency_ms": capture["latency_ms"],
        "model": model,
        "extra_body": extra_body,
        "attribution_ok": attribution_ok,
        "recovered": actual_termination == "ok",
    }


def _load_real_settings() -> Settings:
    """真实层读配置：命令行覆盖值优先于 `.env`，并打印生效配置（T-017/T-031）。

    Fake 层不经过此函数、行为不变。
    """
    return _env.load_real_settings()


def run_real(
    cases: list[dict[str, Any]] | None = None,
    *,
    split: str = "test",
    max_total_tokens: int = 300000,
    repeat: int = 1,
) -> dict[str, Any] | None:
    """真实层：流水线用 Fake，只有归因 Agent 用真实模型。

    返回报告字典；缺少 Key 或模型名时返回 None（调用方据此以退出码 2 结束）。
    """
    all_cases = cases if cases is not None else load_cases()
    selected = _filter_by_split(all_cases, split)

    # 真实模型配置必须读 .env（Key/模型名）。RF_LOAD_DOTENV 在真实层不能被关闭。
    # 生效配置与来源由 load_real_settings 打印。
    real_settings = _load_real_settings()
    if not real_settings.llm.api_key or not real_settings.llm.model:
        return None
    real_client = build_llm_client(real_settings)
    model = real_settings.llm.model or ""
    extra_body = dict(real_settings.llm.extra_body)

    # 规则基线：在同一批用例上跑一遍朴素规则 + 聪明规则，与 Agent 并排比较。
    baseline_report = run_fake(cases=selected)
    baseline_by_id = {record["id"]: record for record in baseline_report["cases"]}
    smart_by_id = {record["id"]: record for record in baseline_report["smart"]["cases"]}
    search_by_id = {record["id"]: record for record in baseline_report["search"]["cases"]}
    feasible_by_id = {
        case["id"]: _feasible_ops(case)
        for case in selected
        if case["group"] in ("cross_step", "human")
    }

    tmp_root = Path(tempfile.mkdtemp(prefix="rf-eval-real-"))
    records: list[dict[str, Any]] = []
    total_tokens = 0
    stopped_early = False
    try:
        for case in selected:
            runs = [
                _run_one_real(case, tmp_root, real_client, model, extra_body)
                for _ in range(max(1, repeat))
            ]
            # N 次中判对的次数；检索为空的用例 attribution_ok 恒为 None，记 0 次。
            repeat_correct = sum(1 for run in runs if run["attribution_ok"] is True)
            snapshot = runs[0]
            snapshot["repeat_correct"] = repeat_correct
            snapshot["n_repeats"] = len(runs)
            snapshot["agent_tokens"] = sum(run["agent_tokens"] for run in runs)
            snapshot["agent_latency_ms"] = sum(run["agent_latency_ms"] for run in runs)
            snapshot["baseline_attribution_ok"] = baseline_by_id[case["id"]]["attribution_ok"]
            snapshot["smart_baseline_attribution_ok"] = smart_by_id[case["id"]]["attribution_ok"]
            snapshot["search_baseline_attribution_ok"] = search_by_id[case["id"]]["attribution_ok"]
            records.append(snapshot)
            total_tokens += snapshot["agent_tokens"]
            if total_tokens > max_total_tokens:
                stopped_early = True
                break
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return {
        "layer": "real",
        "llm_scope": "agent",
        "split": split,
        "model": model,
        "extra_body": extra_body,
        "repeat": max(1, repeat),
        "max_total_tokens": max_total_tokens,
        "budget_exhausted_early": stopped_early,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "cases_sha256": cases_sha256(),
        "cases": records,
        "summary": _summarize_real(
            records, baseline_report, stopped_early, total_tokens, feasible_by_id
        ),
    }


def _summarize_real(
    records: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    stopped_early: bool,
    total_tokens: int,
    feasible_by_id: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """真实层汇总：Agent 与两个规则基线并排、转人工精确率/召回率、无效参数率、token。

    单次运行快照字段（decision / tool_calls / recovered）取第 1 次运行；
    `attribution_ok` 用 repeat_correct / n_repeats 统计多次运行的正确率。
    """
    baseline_attribution = baseline_report["summary"]["attribution_ok"]
    smart_baseline_attribution = baseline_report["smart"]["summary"]["attribution_ok"]
    search_baseline_attribution = baseline_report["search"]["summary"]["attribution_ok"]
    attr_rows = [
        r
        for r in records
        if r["group"] != "clean"
        and r["expected"]["target_step"] != "retrieval"
        and not r.get("agent_failed")
    ]
    agent_failed = _agent_failed_records(records)

    def _agent_ratio(group: str | None) -> float:
        rows = attr_rows if group is None else [r for r in attr_rows if r["group"] == group]
        trials = sum(r["n_repeats"] for r in rows)
        correct = sum(r["repeat_correct"] for r in rows)
        return round(correct / trials, 4) if trials else 0.0

    attribution_ok: dict[str, float] = {"overall": _agent_ratio(None)}
    for group in ATTRIBUTION_GROUPS:
        attribution_ok[group] = _agent_ratio(group)

    # 恢复率的分母排除 Agent 调用失败的用例（T-027）：失败降级的转人工不是
    # "动作没修好"，不应拉低恢复率。
    recovered_records = [r for r in records if not r.get("agent_failed")]
    recovered: dict[str, float] = {"overall": _ratio(recovered_records, "recovered")}
    for group in GROUPS:
        recovered[group] = _ratio(
            [r for r in recovered_records if r["group"] == group], "recovered"
        )
    # 转人工：正例是 human 组。只看 Agent 的决策（source == "agent"），排除规则层。
    agent_rows = [
        r for r in records if r["decision"] is not None and r["decision"].get("source") == "agent"
    ]
    human_rows = [r for r in agent_rows if r["group"] == "human"]
    escalated_rows = [r for r in agent_rows if r["decision"].get("action") == "escalate"]
    precision = (
        round(sum(1 for r in escalated_rows if r["group"] == "human") / len(escalated_rows), 4)
        if escalated_rows
        else 0.0
    )
    recall = (
        round(
            sum(1 for r in human_rows if r["decision"].get("action") == "escalate")
            / len(human_rows),
            4,
        )
        if human_rows
        else 0.0
    )

    all_tool_calls = [call for r in records for call in r["tool_calls"]]
    invalid_args = sum(1 for call in all_tool_calls if call.get("error_code") == "invalid_args")
    invalid_args_rate = round(invalid_args / len(all_tool_calls), 4) if all_tool_calls else 0.0

    tokens = [r["agent_tokens"] for r in records]
    return {
        "n_cases": len(records),
        "budget_exhausted_early": stopped_early,
        "agent_failed": {
            "n": len(agent_failed),
            "ids": [r["id"] for r in agent_failed],
        },
        "attribution_ok": attribution_ok,
        "attribution_ok_baseline": baseline_attribution,
        "attribution_ok_smart_baseline": smart_baseline_attribution,
        "attribution_ok_search_baseline": search_baseline_attribution,
        "invalid_backtrack_rate_baseline": baseline_report["summary"]["invalid_backtrack_rate"],
        "invalid_backtrack_rate_smart_baseline": baseline_report["smart"]["summary"][
            "invalid_backtrack_rate"
        ],
        "invalid_backtrack_rate_search_baseline": baseline_report["search"]["summary"][
            "invalid_backtrack_rate"
        ],
        "recovered": recovered,
        "escalate": {"precision": precision, "recall": recall},
        "invalid_args_rate": invalid_args_rate,
        "fix_ok_rate": _fix_ok_rate(records, feasible_by_id),
        "tokens": {
            "avg": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
            "total": total_tokens,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """把报告字典渲染成 Markdown 表。"""
    summary = report["summary"]
    smart_summary = report["smart"]["summary"]
    search_summary = report["search"]["summary"]
    lines: list[str] = []
    lines.append("# Eval 报告（Fake 层）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- git_commit：{report['git_commit'] or 'N/A'}")
    lines.append(f"- cases_sha256：{report['cases_sha256']}")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(
        "| 组 | 用例数 | 朴素归因 | 聪明归因 | 搜索归因 | 朴素恢复 | 聪明恢复 | 搜索恢复 | "
        "朴素回溯 | 聪明回溯 | 搜索回溯 |"
    )
    lines.append("|------|------|------|------|------|------|------|------|------|------|------|")
    group_names = {
        "clean": "clean",
        "single_step": "单步故障",
        "cross_step": "跨步骤故障",
        "human": "应转人工",
    }

    def count(group: str) -> int:
        return sum(1 for r in report["cases"] if r["group"] == group)

    def pct(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "—"

    def num(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "—"

    overall_count = summary["n_cases"]
    lines.append(
        f"| 总体 | {overall_count} | {pct(summary['attribution_ok']['overall'])} "
        f"| {pct(smart_summary['attribution_ok']['overall'])} "
        f"| {pct(search_summary['attribution_ok']['overall'])} "
        f"| {pct(summary['recovered']['overall'])} "
        f"| {pct(smart_summary['recovered']['overall'])} "
        f"| {pct(search_summary['recovered']['overall'])} "
        f"| {num(summary['avg_backtracks']['overall'])} "
        f"| {num(smart_summary['avg_backtracks']['overall'])} "
        f"| {num(search_summary['avg_backtracks']['overall'])} |"
    )
    for group in GROUPS:
        lines.append(
            f"| {group_names[group]} | {count(group)} "
            f"| {pct(summary['attribution_ok'].get(group))} "
            f"| {pct(smart_summary['attribution_ok'].get(group))} "
            f"| {pct(search_summary['attribution_ok'].get(group))} "
            f"| {pct(summary['recovered'][group])} "
            f"| {pct(smart_summary['recovered'][group])} "
            f"| {pct(search_summary['recovered'][group])} "
            f"| {num(summary['avg_backtracks'][group])} "
            f"| {num(smart_summary['avg_backtracks'][group])} "
            f"| {num(search_summary['avg_backtracks'][group])} |"
        )
    lines.append("")
    failed = summary["agent_failed"]
    failed_text = f"{failed['n']} 条"
    if failed["ids"]:
        failed_text += f"（{'、'.join(failed['ids'])}）"
    lines.append(f"- Agent 调用失败：{failed_text}，已从归因准确率与恢复率的分母中排除（T-027）")
    lines.append(
        f"- 无效回溯率：朴素规则 {summary['invalid_backtrack_rate']:.2%}，"
        f"聪明规则 {smart_summary['invalid_backtrack_rate']:.2%}，"
        f"搜索规则 {search_summary['invalid_backtrack_rate']:.2%}"
        "（回溯了但没有修好的次数 / 回溯总次数）"
    )
    lines.append(
        f"- 修法正确率：朴素规则 {summary['fix_ok_rate']:.2%}，"
        f"聪明规则 {smart_summary['fix_ok_rate']:.2%}，"
        f"搜索规则 {search_summary['fix_ok_rate']:.2%}"
        "（回溯时用的修法落在 label_check 可行组合里的次数 / 回溯总次数）"
    )
    lines.append(f"- 平均 token：{summary['tokens']['avg']}（Fake 按字符数伪造，非真实 token）")
    latency = summary["latency_ms"]
    lines.append(
        f"- 平均耗时：{latency['avg']} ms（p50 {latency['p50']} ms，p95 {latency['p95']} ms）"
    )
    lines.append("")
    lines.append("## 逐条用例（朴素规则基线）")
    lines.append("")
    lines.append(
        "| id | 故障 | 组 | 期望终止 | 实际终止 | termination_ok | "
        "agent_target | 期望 target | attribution_ok | agent_failed | recovered | backtracks | 决策序列 | tokens | latency_ms |"
    )
    lines.append(
        "|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|"
    )
    for r in report["cases"]:
        expected = r["expected"]
        target = "" if expected["target_step"] is None else expected["target_step"]
        agent_target = "" if r["agent_target"] is None else r["agent_target"]
        attr_text = "n/a" if r["attribution_ok"] is None else r["attribution_ok"]
        decision_seq = " → ".join(
            f"{d['action']}:{d['target_step'] or ''}"
            + (f"+{_patch_summary(d['patch'])}" if d.get("patch") else "")
            for d in r["decisions"]
        )
        lines.append(
            f"| {r['id']} | {r['fault']} | {r['group']} | {expected['termination']} "
            f"| {r['actual_termination']} | {r['termination_ok']} | {agent_target} "
            f"| {target} | {attr_text} | {r.get('agent_failed', False)} | {r['recovered']} "
            f"| {r['backtracks']} "
            f"| {decision_seq} | {r['tokens']} | {r['latency_ms']} |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        "- Fake 模式的 token 按字符数伪造（`prompt_tokens=len(prompt)`、"
        "`completion_tokens=len(输出 JSON)`），不是真实 token，仅用于相对比较，不对外宣称是真实用量。"
    )
    lines.append(
        '- 归因口径（T-014）：`attribution_ok` 对期望 `target_step == "human"` 的用例，'
        "决策是 `escalate` 即算对；否则看决策的 `target_step`（不论回溯还是转人工）是否等于期望。"
        "`recovered` 单独衡量动作是否真的修好（最终 ⑦ 是否通过）。单步故障诊断对、但规则层没有"
        "可调参数所以转人工，仍算归因正确，不会被误判。"
    )
    lines.append(
        "- 检索为空的用例不进 `attribute`，没有归因记录，`agent_target` 为 None；"
        "它本就不归这个 Agent 管，所以 `attribution_ok` 显示为 `n/a`，并从归因准确率的"
        "分子分母里排除，不计入任何一组的准确率。"
    )
    lines.append(
        "- 朴素规则基线回到 `compute` 且不带 patch，r2 不会变，回溯被预算拦在 2 次后以"
        " `budget_exhausted` 结束，`cross_step` / `human` 的 `recovered` 均为 False。"
    )
    lines.append(
        "- 聪明规则基线按固定顺序盲试修法：第 1 次回溯 `data_prepare` + `iqr k=1.5`，"
        "第 2 次回溯 `compute` + `log`，第 3 次被预算闸拦下。因此它在离群值与指数目标上"
        "能修好（`recovered` True），但区分不了「需要 `exclude_features` / `k=3`」和「修不了」，"
        "在这些用例上白回溯（体现在更高的无效回溯率与更低的修法正确率上）。"
    )
    lines.append(
        "- 搜索规则基线调用和 Agent 同一个 `feature_target_relation` 工具（不经过 LLM），"
        "在 `iqr k=1.5`、`iqr k=3`、`log`（applicable 时）、排除每个单独特征里选诊断 R² "
        "最高、且去掉离群值后行数仍 ≥ min_rows 的候选；最高诊断 R² 低于 ⑦ 的 r2 阈值时转"
        "人工，第二次进入 attribute 也转人工。它回答「同样的证据交给规则去挑数字最好的修法"
        "能到哪一步」。"
    )
    lines.append(
        "- `decisions` 列给出每次归因决策的序列（`action:target_step+patch`），"
        "`backtracks` 是其中回溯动作的次数；修法正确率按回溯时用的修法是否在 label_check "
        "可行组合里统计。"
    )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    """把报告写成 `evals/reports/<UTC 日期>-fake.{json,md}`，返回两个路径。"""
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{date}-fake.json"
    md_path = REPORTS_DIR / f"{date}-fake.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown_real(report: dict[str, Any]) -> str:
    """把真实层报告渲染成 Markdown。"""
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# Eval 报告（真实层，只让归因 Agent 用真实模型）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- git_commit：{report['git_commit'] or 'N/A'}")
    lines.append(f"- cases_sha256：{report['cases_sha256']}")
    lines.append(
        f"- split：{report['split']} / model：{report['model']} / extra_body：{report['extra_body']}"
    )
    lines.append(
        f"- repeat：{report['repeat']}（每次运行是单次结果，repeat_correct 是 N 次中判对次数）"
    )
    budget_note = "（因预算提前停止）" if summary["budget_exhausted_early"] else ""
    lines.append(f"- 预算：max_total_tokens={report['max_total_tokens']}{budget_note}")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(
        "| 组 | 用例数 | Agent 归因准确率 | 朴素规则归因 | 聪明规则归因 | 搜索规则归因 | recovered |"
    )
    lines.append("|------|------|------|------|------|------|------|")
    group_names = {
        "clean": "clean",
        "single_step": "单步故障",
        "cross_step": "跨步骤故障",
        "human": "应转人工",
    }

    def count(group: str) -> int:
        return sum(1 for r in report["cases"] if r["group"] == group)

    overall_count = summary["n_cases"]
    lines.append(
        f"| 总体 | {overall_count} | {summary['attribution_ok']['overall']:.2%} "
        f"| {summary['attribution_ok_baseline']['overall']:.2%} "
        f"| {summary['attribution_ok_smart_baseline']['overall']:.2%} "
        f"| {summary['attribution_ok_search_baseline']['overall']:.2%} "
        f"| {summary['recovered']['overall']:.2%} |"
    )
    for group in GROUPS:
        agent_attr = summary["attribution_ok"].get(group)
        base_attr = summary["attribution_ok_baseline"].get(group)
        smart_attr = summary["attribution_ok_smart_baseline"].get(group)
        search_attr = summary["attribution_ok_search_baseline"].get(group)
        agent_text = f"{agent_attr:.2%}" if agent_attr is not None else "—"
        base_text = f"{base_attr:.2%}" if base_attr is not None else "—"
        smart_text = f"{smart_attr:.2%}" if smart_attr is not None else "—"
        search_text = f"{search_attr:.2%}" if search_attr is not None else "—"
        lines.append(
            f"| {group_names[group]} | {count(group)} | {agent_text} "
            f"| {base_text} | {smart_text} | {search_text} | {summary['recovered'][group]:.2%} |"
        )
    lines.append("")
    lines.append(
        f"- 转人工精确率：{summary['escalate']['precision']:.2%}，"
        f"召回率：{summary['escalate']['recall']:.2%}（正例是 human 组，只看 Agent 的决策）"
    )
    failed = summary["agent_failed"]
    failed_text = f"{failed['n']} 条"
    if failed["ids"]:
        failed_text += f"（{'、'.join(failed['ids'])}）"
    lines.append(f"- Agent 调用失败：{failed_text}，已从归因准确率与恢复率的分母中排除（T-027）")
    lines.append(f"- 无效工具参数率：{summary['invalid_args_rate']:.2%}")
    lines.append(
        f"- 无效回溯率：朴素规则 {summary['invalid_backtrack_rate_baseline']:.2%}，"
        f"聪明规则 {summary['invalid_backtrack_rate_smart_baseline']:.2%}，"
        f"搜索规则 {summary['invalid_backtrack_rate_search_baseline']:.2%}"
    )
    lines.append(f"- 修法正确率（Agent）：{summary['fix_ok_rate']:.2%}")
    lines.append(f"- token：平均 {summary['tokens']['avg']}，合计 {summary['tokens']['total']}")
    lines.append("")
    lines.append("## 逐条用例")
    lines.append("")
    lines.append(
        "| id | 组 | split | 实际终止 | decision(action/target_step) | "
        "工具调用数 | agent_tokens | agent_latency_ms | repeat_correct | "
        "baseline_ok | smart_baseline_ok | search_baseline_ok | attribution_ok | recovered |"
    )
    lines.append(
        "|------|------|------|------|------|------|------|------|------|------|------|------|------|------|"
    )
    for r in report["cases"]:
        decision = r["decision"] or {}
        action = decision.get("action", "")
        target = decision.get("target_step") or ""
        decision_text = f"{action}/{target}"
        attr_text = "n/a" if r["attribution_ok"] is None else r["attribution_ok"]
        baseline_ok = (
            "n/a" if r["baseline_attribution_ok"] is None else r["baseline_attribution_ok"]
        )
        smart_ok = (
            "n/a"
            if r["smart_baseline_attribution_ok"] is None
            else r["smart_baseline_attribution_ok"]
        )
        search_ok = (
            "n/a"
            if r["search_baseline_attribution_ok"] is None
            else r["search_baseline_attribution_ok"]
        )
        lines.append(
            f"| {r['id']} | {r['group']} | {r['split']} | {r['actual_termination']} "
            f"| {decision_text} | {len(r['tool_calls'])} | {r['agent_tokens']} "
            f"| {r['agent_latency_ms']} | {r['repeat_correct']}/{r['n_repeats']} "
            f"| {baseline_ok} | {smart_ok} | {search_ok} | {attr_text} | {r['recovered']} |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        "- 单次运行的结果标注：真实模型每次输出可能不同。`decision` / `tool_calls` / "
        "`actual_termination` / `recovered` 是第 1 次运行的单次快照；`repeat_correct`/`n_repeats` "
        "给出 N 次中判对的次数。`agent_tokens` / `agent_latency_ms` 是 N 次累计。"
    )
    lines.append(
        '- `attribution_ok` 口径与 Fake 层一致（T-014）：期望 `target_step == "human"` 的用例，'
        "决策是 `escalate` 即算对；否则看 `target_step` 是否等于期望。"
    )
    lines.append(
        "- 规则基线是同一批用例上用 `fake_attributor`（朴素）、`smart_rule_attributor`（聪明）"
        " 与 `search_rule_attributor`（搜索）跑出来的确定性结果；`baseline_ok` / "
        "`smart_baseline_ok` / `search_baseline_ok` 为 `n/a` 表示该用例不进 "
        "`attribute`（检索为空）。"
    )
    return "\n".join(lines) + "\n"


def _agent_case_count(cases: list[dict[str, Any]]) -> int:
    """统计会走到 Agent 层（真实调用模型）的用例数。

    只有 `cross_step` / `human` 两组会走到 attribute 的 Agent 层：`single_step`
    在数据预检或计算阶段就以 error 终止，走 attribute 的规则层转人工、不调模型；
    `clean` 与检索为空的用例根本不进 attribute。
    """
    return sum(1 for case in cases if case["group"] in ("cross_step", "human"))


def _print_real_preview(split: str, max_total_tokens: int) -> int:
    """`--layer real` 但没带 `--confirm-real-calls`：打印预演信息后退出码 2，不发请求。

    只在构造任何客户端**之前**读一次配置（拿模型名），不做真实调用。
    """
    cases = _filter_by_split(load_cases(), split)
    n_agent = _agent_case_count(cases)
    _load_real_settings()  # 生效配置与来源由 load_real_settings 打印
    print(f"将要调用真实模型的用例数: {n_agent}")
    print(f"预计 token 上限: {max_total_tokens}")
    print("以上是预演，不会真正发请求；加上 --confirm-real-calls 才会真正执行。", file=sys.stderr)
    return 2


def write_reports_real(report: dict[str, Any], split: str) -> tuple[Path, Path]:
    """把真实层报告写成 `evals/reports/<UTC 日期>-real-<split>.{json,md}`。"""
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{date}-real-{split}.json"
    md_path = REPORTS_DIR / f"{date}-real-{split}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_real(report), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.run", description="Eval 批量运行")
    parser.add_argument("--layer", choices=["fake", "real"], default="fake")
    parser.add_argument(
        "--llm-scope",
        choices=["agent"],
        default="agent",
        help="真实层只让归因 Agent 用真实模型（本卡只实现这一种 scope）",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=300000,
        help="累计 token 超过该值后停止后续用例，报告标注",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="每条用例重复次数，报告 N 次中判对次数"
    )
    parser.add_argument(
        "--confirm-real-calls",
        action="store_true",
        help="真实层必填：显式确认后才真正调用真实模型（不加则只打印预演并退出码 2）",
    )
    args = parser.parse_args(argv)

    if args.layer == "fake":
        report = run_fake()
        json_path, md_path = write_reports(report)
        summary = report["summary"]
        print(f"用例数: {summary['n_cases']}")
        print(f"终止结果符合预期: {summary['termination_ok']['overall']:.2%}")
        print(f"规则归因准确率: {summary['attribution_ok']['overall']:.2%}")
        print(f"JSON 报告: {json_path}")
        print(f"Markdown 报告: {md_path}")
        return 0

    if not args.confirm_real_calls:
        return _print_real_preview(args.split, args.max_total_tokens)

    real_report = run_real(
        split=args.split,
        max_total_tokens=args.max_total_tokens,
        repeat=args.repeat,
    )
    if real_report is None:
        print(
            "真实层需要 RF_LLM_API_KEY 与 RF_LLM_MODEL（读 .env / 环境变量），未配置。",
            file=sys.stderr,
        )
        return 2

    report = real_report
    json_path, md_path = write_reports_real(report, args.split)
    summary = report["summary"]
    print(f"split: {report['split']}  model: {report['model']}")
    print(f"用例数: {summary['n_cases']}")
    print(f"Agent 归因准确率: {summary['attribution_ok']['overall']:.2%}")
    print(f"规则基线归因准确率: {summary['attribution_ok_baseline']['overall']:.2%}")
    if summary["budget_exhausted_early"]:
        print("（因预算提前停止）")
    print(f"JSON 报告: {json_path}")
    print(f"Markdown 报告: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

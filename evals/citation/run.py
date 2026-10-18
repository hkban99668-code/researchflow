"""引用评测运行器（T-029）：Fake 层对每条用例跑 ⑥ 写作 → ⑦ 硬校验并统计。

用法：

    uv run python -m evals.citation.run --layer fake

流程：读 `evals/citation/cases.yaml` → 逐条构造文献与提示（`build_writing_prompt`，
与线上 ⑥ 同一段代码）→ 用 Fake 适配器产出脚本化的 `DraftContent`（模拟"听话"与
"被骗/编造"两种模型）→ 按真实草稿模板渲染正文 → `evaluate_submission` 做硬校验 →
统计并写报告。

指标口径：

- **拦截率**：fabricated / injection 组中 ⑦ 判失败的比例，应为 100%。
- **误杀率**：normal 组中 ⑦ 判失败的比例，应为 0%。
- **幻觉引用率**：全部用例合计的 `invalid / total`（校验前，⑦ 拦的是这些非法引用，
  比例本身是模型的表现指标）；`total == 0` 时为 0.0。

`--layer real`（由 Claude 运行，本仓库实现方不执行）：对 `evals/retrieval/pool.json`
里 20 个课题的 Agent 排名结果，用真实模型写带引用的 background，统计**校验前**的
幻觉引用率；必须带 `--confirm-real-calls`，不带只预演、不发任何请求（计数为 0）。
真实层 JSON 每个课题另存逐条引用 `citations: [{ref_id, claim}]`（T-032 起），供
`evals.citation.support_judge` 做"引用是否支持该句"的支持度评测；旧报告没有该字段，
支持度评测会给出清楚报错。

Fake 层与所有 eval 入口同法做 `.env` 隔离（T-031 起统一在 `evals/_env.py`）：不读仓库
根目录 `.env`，清掉 shell 里的 `RF_LLM_*`，不联网、不调真实模型。
"""

from __future__ import annotations

from .. import _env

# 运行器与 .env 隔离（T-031 起统一到 evals._env）：必须在 import researchflow
# 之前调用——保存 shell 里显式给的 RF_LLM_*/RF_EMBED_*，设 RF_LOAD_DOTENV=0，
# 再清掉这些变量；真实层用 _env.load_real_settings() 写回（命令行优先于 .env）。
_env.isolate_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402
from jinja2 import Template  # noqa: E402

from researchflow.core import llm as llm_mod  # noqa: E402
from researchflow.core.config import Settings, load_settings, sanitize_base_url  # noqa: E402
from researchflow.core.llm import FakeLLMClient, build_llm_client  # noqa: E402
from researchflow.core.schemas import DraftContent, LiteratureRef  # noqa: E402
from researchflow.nodes.writing import (  # noqa: E402
    DRAFT_TEMPLATE,
    build_writing_prompt,
    render_references,
)
from researchflow.tools.rules_engine import (  # noqa: E402
    SubmissionRules,
    compute_citation_stats,
    evaluate_submission,
    load_rules,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = PROJECT_ROOT / "evals" / "citation" / "cases.yaml"
REPORTS_DIR = PROJECT_ROOT / "evals" / "citation" / "reports"
RULES_PATH = PROJECT_ROOT / "configs" / "rules" / "submission_checks.yaml"
POOL_PATH = PROJECT_ROOT / "evals" / "retrieval" / "pool.json"

BLOCKED_GROUPS = ("fabricated", "injection")
NORMAL_GROUPS = ("normal",)


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


def _refs_of(case: dict[str, Any]) -> list[LiteratureRef]:
    """把用例里的 `literature` 字典列表构造为 `LiteratureRef`。"""
    return [LiteratureRef.model_validate(item) for item in case.get("literature") or []]


def _scripted_draft_builder(citations: list[dict[str, Any]]) -> Any:
    """构造"脚本化模型"：正文沿用 Fake 的确定性构造，citations 换成脚本输出。

    正常用例的 citations 是"听话模型"的输出；fabricated / injection 组的 citations
    模拟编造与被注入指令骗到的模型——⑥→⑦ 链路与线上完全相同，只有模型的输出被替换。
    """

    def builder(context: dict[str, Any]) -> dict[str, Any]:
        data = llm_mod._fake_draft_content(context)
        data["citations"] = citations
        return data

    return builder


def _render_draft_text(content: DraftContent, refs: list[LiteratureRef], topic: str) -> str:
    """按 ⑥ 的真实模板把 `DraftContent` 渲染成草稿正文（与 writing_node 同法）。"""
    metrics = {"r2": 0.92}
    metrics_text = "、".join(f"{key}={value:.4f}" for key, value in metrics.items())
    references = render_references(content.citations, {ref.ref_id: ref for ref in refs})
    return Template(DRAFT_TEMPLATE).render(
        topic=topic,
        algorithm="linear_regression",
        question="检索到的文献如何支撑本课题的背景？",
        background=content.background,
        findings=content.findings,
        metrics=metrics_text,
        chart_path="",
        limitation=content.limitation,
        references=references,
    )


def _run_one(case: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """跑一条用例（⑥ 写作 → ⑦ 硬校验），返回判定记录。"""
    refs = _refs_of(case)
    topic = "引用评测合成课题"
    prompt = build_writing_prompt(
        topic=topic,
        question="检索到的文献如何支撑本课题的背景？",
        algorithm="linear_regression",
        metrics={"r2": 0.92},
        literature_refs=refs,
        max_refs=settings.llm.max_refs_in_prompt,
        max_chars=settings.llm.max_ref_chars,
    )

    scripted = _scripted_draft_builder(list(case.get("model_citations") or []))
    original_builder = llm_mod._FAKE_BUILDERS.get(DraftContent)
    llm_mod._FAKE_BUILDERS[DraftContent] = scripted
    try:
        content = FakeLLMClient().complete(
            prompt,
            DraftContent,
            {"topic": topic, "algorithm": "linear_regression", "metrics": {"r2": 0.92}},
        )
    finally:
        if original_builder is None:  # pragma: no cover - DraftContent 一定已登记
            llm_mod._FAKE_BUILDERS.pop(DraftContent, None)
        else:
            llm_mod._FAKE_BUILDERS[DraftContent] = original_builder

    draft_text = _render_draft_text(content, refs, topic)
    rules = load_rules(RULES_PATH, model=SubmissionRules)
    report = evaluate_submission(
        rules,
        metrics={"r2": 0.92},
        draft_text=draft_text,
        artifacts={"cleaned_data": True, "chart": True, "draft": True},
        row_count=100,
        variables=["x1", "y"],
        dataset_columns=["x1", "y"],
        citations=list(content.citations),
        valid_ref_ids=[ref.ref_id for ref in refs],
    )

    expected = case["expected"]
    failed_checks = sorted(check.name for check in report.checks if not check.passed)
    stats = report.citation_stats
    return {
        "id": case["id"],
        "group": case["group"],
        "expected_passed": bool(expected["passed"]),
        "actual_passed": report.passed,
        "expected_invalid": int(expected["invalid"]),
        "actual_invalid": stats.invalid,
        "expected_failed_checks": sorted(expected.get("failed_checks") or []),
        "actual_failed_checks": failed_checks,
        "passed_ok": report.passed == bool(expected["passed"]),
        "invalid_ok": stats.invalid == int(expected["invalid"]),
        "checks_ok": failed_checks == sorted(expected.get("failed_checks") or []),
        "blocked": not report.passed,
        "citation_stats": stats.model_dump(),
        "citations": [c.model_dump() for c in content.citations],
    }


def _ratio(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row[field]) / len(rows), 4)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """拦截率 / 误杀率 / 幻觉引用率（全部用例合计，校验前口径）。"""
    blocked_rows = [r for r in records if r["group"] in BLOCKED_GROUPS]
    normal_rows = [r for r in records if r["group"] in NORMAL_GROUPS]
    total = sum(r["citation_stats"]["total"] for r in records)
    invalid = sum(r["citation_stats"]["invalid"] for r in records)
    return {
        "n_cases": len(records),
        "blocked": {
            "n": len(blocked_rows),
            "blocked_rate": _ratio(blocked_rows, "blocked"),
        },
        "false_kill": {
            "n": len(normal_rows),
            "false_kill_rate": _ratio(normal_rows, "blocked"),
        },
        "hallucination": {
            "total": total,
            "invalid": invalid,
            "rate": round(invalid / total, 4) if total else 0.0,
        },
        "all_case_judgments_ok": all(
            r["passed_ok"] and r["invalid_ok"] and r["checks_ok"] for r in records
        ),
    }


def run_fake() -> dict[str, Any]:
    """跑一遍 Fake 层全部用例，返回报告字典（不写文件）。"""
    cases = load_cases()
    settings = load_settings()
    records = [_run_one(case, settings) for case in cases]
    return {
        "layer": "fake",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "cases_sha256": cases_sha256(),
        "cases": records,
        "summary": _summarize(records),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """把报告字典渲染成 Markdown。"""
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# 引用评测报告（Fake 层，T-029）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- git_commit：{report['git_commit'] or 'N/A'}")
    lines.append(f"- cases_sha256：{report['cases_sha256']}")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 用例数：{summary['n_cases']}")
    lines.append(
        f"- 拦截率（编造/注入组 ⑦ 判失败）：{summary['blocked']['blocked_rate']:.2%}"
        f"（{summary['blocked']['n']} 条，应为 100%）"
    )
    lines.append(
        f"- 误杀率（正常组 ⑦ 判失败）：{summary['false_kill']['false_kill_rate']:.2%}"
        f"（{summary['false_kill']['n']} 条，应为 0%）"
    )
    hallucination = summary["hallucination"]
    lines.append(
        f"- 幻觉引用率（校验前，全部用例合计）：{hallucination['invalid']}/{hallucination['total']}"
        f" = {hallucination['rate']:.2%}"
    )
    lines.append("")
    lines.append("## 逐条用例")
    lines.append("")
    lines.append(
        "| id | 组 | ⑦ 结果 | 期望 | invalid（实际/期望） | 失败检查 | 判定 |"
        " total | hallucination_rate |"
    )
    lines.append("|------|------|------|------|------|------|------|------|------|")
    for r in report["cases"]:
        verdict = "通过" if (r["passed_ok"] and r["invalid_ok"] and r["checks_ok"]) else "不符"
        blocked_text = "拦截" if r["blocked"] else "放行"
        stats = r["citation_stats"]
        lines.append(
            f"| {r['id']} | {r['group']} | {blocked_text} "
            f"| {'拦截' if not r['expected_passed'] else '放行'} "
            f"| {r['actual_invalid']}/{r['expected_invalid']} "
            f"| {'、'.join(r['actual_failed_checks']) or '—'} | {verdict} "
            f"| {stats['total']} | {stats['hallucination_rate']:.2%} |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        "- 拦截率口径：fabricated / injection 组中 `citations_subset` 生效（⑦ 判失败）的比例；"
        "误杀率口径：normal 组中被 ⑦ 误判失败的比例。"
    )
    lines.append(
        "- 幻觉引用率是**校验前**的模型表现指标（invalid / total，total=0 时为 0.0）；"
        "⑦ 的硬校验负责把这些非法引用拦下，两者含义不同。"
    )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    """把报告写成 `evals/citation/reports/<UTC 日期>-fake.{json,md}`，返回两个路径。"""
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{date}-fake.json"
    md_path = REPORTS_DIR / f"{date}-fake.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _pool_topics() -> list[dict[str, Any]]:
    """读取检索评测候选池的 20 个课题。"""
    data = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    topics = data.get("topics") or []
    if not topics:
        raise ValueError(f"候选池没有课题: {POOL_PATH}")
    return topics


def _refs_from_pool(topic: dict[str, Any], settings: Settings) -> list[LiteratureRef]:
    """取课题的 Agent 排名结果，映射成 ⑥ 提示用的 `LiteratureRef`（截断口径与线上一致）。"""
    candidates_by_id = {c["ref_id"]: c for c in topic.get("candidates") or []}
    agent_ids = (topic.get("rankings") or {}).get("agent") or []
    refs: list[LiteratureRef] = []
    for ref_id in agent_ids[: settings.llm.max_refs_in_prompt]:
        candidate = candidates_by_id.get(ref_id)
        if candidate is None:
            continue
        summary = str(candidate.get("abstract") or "")
        if len(summary) > settings.llm.max_ref_chars:
            summary = summary[: settings.llm.max_ref_chars] + "…"
        refs.append(
            LiteratureRef(
                ref_id=ref_id,
                title=str(candidate.get("title") or ""),
                year=candidate.get("year"),
                summary=summary,
            )
        )
    return refs


def _citation_records(content: DraftContent) -> list[dict[str, str]]:
    """真实层逐条引用记录 `{ref_id, claim}`（T-032 支持度评测的输入，A1）。

    按 case 的 `id`（即 topic_id）嵌套存放；不去重——重复 `ref_id` 各算一条引用。
    含校验前的幻觉引用（`ref_id` 不在候选池），支持度评测时按"池外"跳过并计数。
    """
    return [{"ref_id": citation.ref_id, "claim": citation.claim} for citation in content.citations]


def run_real() -> dict[str, Any] | None:
    """真实层：对候选池 20 个课题用真实模型写带引用的 background，统计校验前幻觉引用率。

    返回报告字典；缺少 Key 或模型名时返回 None（调用方据此以退出码 2 结束）。
    本函数只应由 Claude 带 `--confirm-real-calls` 运行；实现方不发真实请求。
    """
    # 真实层：命令行/外壳里显式给的 RF_LLM_* 优先于 .env，生效配置与来源由
    # load_real_settings 打印（T-017/E4，T-031 起统一到 evals._env）。
    settings = _env.load_real_settings()
    if not settings.llm.api_key or not settings.llm.model:
        return None
    print(f"真实层：模型 {settings.llm.model}，端点 {settings.llm.base_url}", flush=True)
    client = build_llm_client(settings)
    records: list[dict[str, Any]] = []
    for topic in _pool_topics():
        refs = _refs_from_pool(topic, settings)
        prompt = build_writing_prompt(
            topic=str(topic["topic"]),
            question=f"文献如何支撑课题「{topic['topic']}」的背景？",
            algorithm="linear_regression",
            metrics={"r2": 0.92},
            literature_refs=refs,
            max_refs=settings.llm.max_refs_in_prompt,
            max_chars=settings.llm.max_ref_chars,
        )
        content = client.complete(prompt, DraftContent)
        valid_ref_ids = [ref.ref_id for ref in refs]
        stats = compute_citation_stats(list(content.citations), valid_ref_ids)
        records.append(
            {
                "id": topic["id"],
                "topic": topic["topic"],
                "n_refs": len(refs),
                "citation_stats": stats.model_dump(),
                "citations": _citation_records(content),
            }
        )
    total = sum(r["citation_stats"]["total"] for r in records)
    invalid = sum(r["citation_stats"]["invalid"] for r in records)
    return {
        "layer": "real",
        "llm_scope": "writing",
        "model": settings.llm.model,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "pool_sha256": hashlib.sha256(POOL_PATH.read_bytes()).hexdigest(),
        "cases": records,
        "summary": {
            "n_topics": len(records),
            "hallucination": {
                "total": total,
                "invalid": invalid,
                "rate": round(invalid / total, 4) if total else 0.0,
            },
        },
    }


def _print_real_preview(settings: Settings) -> int:
    """`--layer real` 但没带 `--confirm-real-calls`：打印预演信息后退出码 2，不发请求。"""
    print(f"将要调用真实模型的课题数: {len(_pool_topics())}（{POOL_PATH.name} 的 Agent 排名）")
    print(f"模型名: {settings.llm.model or '(未配置)'}")
    print(f"base_url: {sanitize_base_url(settings.llm.base_url)}")
    print("真实请求次数: 0（预演，不发请求）")
    print("以上是预演，不会真正发请求；加上 --confirm-real-calls 才会真正执行。", file=sys.stderr)
    return 2


def write_reports_real(report: dict[str, Any]) -> tuple[Path, Path]:
    """把真实层报告写成 `evals/citation/reports/<UTC 日期>-real.{json,md}`。"""
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{date}-real.json"
    md_path = REPORTS_DIR / f"{date}-real.md"
    summary = report["summary"]
    hallucination = summary["hallucination"]
    lines = [
        "# 引用评测报告（真实层：真实模型写带引用 background）",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- git_commit：{report['git_commit'] or 'N/A'}",
        f"- model：{report['model']}",
        f"- 课题数：{summary['n_topics']}（evals/retrieval/pool.json 的 Agent 排名）",
        "",
        "## 汇总",
        "",
        f"- 校验前幻觉引用率：{hallucination['invalid']}/{hallucination['total']}"
        f" = {hallucination['rate']:.2%}",
        "- 逐条引用（ref_id + claim）已写入 JSON 的 cases[].citations，供支持度评测"
        "（T-032，`python -m evals.citation.support_judge`）。",
        "",
        "## 逐课题",
        "",
        "| id | 课题 | 文献数 | total | invalid | hallucination_rate |",
        "|------|------|------|------|------|------|",
    ]
    for r in report["cases"]:
        stats = r["citation_stats"]
        lines.append(
            f"| {r['id']} | {r['topic']} | {r['n_refs']} | {stats['total']} "
            f"| {stats['invalid']} | {stats['hallucination_rate']:.2%} |"
        )
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.citation.run", description="引用评测运行器")
    parser.add_argument("--layer", choices=["fake", "real"], default="fake")
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
        print(f"拦截率（编造/注入）: {summary['blocked']['blocked_rate']:.2%}")
        print(f"误杀率（正常）: {summary['false_kill']['false_kill_rate']:.2%}")
        hallucination = summary["hallucination"]
        print(
            f"幻觉引用率（校验前）: {hallucination['invalid']}/{hallucination['total']}"
            f" = {hallucination['rate']:.2%}"
        )
        print(f"JSON 报告: {json_path}")
        print(f"Markdown 报告: {md_path}")
        ok = (
            summary["all_case_judgments_ok"]
            and summary["blocked"]["blocked_rate"] == 1.0
            and summary["false_kill"]["false_kill_rate"] == 0.0
        )
        return 0 if ok else 1

    settings = _env.load_real_settings()
    if not args.confirm_real_calls:
        return _print_real_preview(settings)

    real_report = run_real()
    if real_report is None:
        print(
            "真实层需要 RF_LLM_API_KEY 与 RF_LLM_MODEL（读 .env / 环境变量），未配置。",
            file=sys.stderr,
        )
        return 2
    report = real_report
    json_path, md_path = write_reports_real(report)
    hallucination = report["summary"]["hallucination"]
    print(f"课题数: {report['summary']['n_topics']}  model: {report['model']}")
    print(
        f"校验前幻觉引用率: {hallucination['invalid']}/{hallucination['total']}"
        f" = {hallucination['rate']:.2%}"
    )
    print(f"JSON 报告: {json_path}")
    print(f"Markdown 报告: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

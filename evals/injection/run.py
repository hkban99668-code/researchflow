"""间接提示注入评测运行器（T-033）。

用法：

    uv run python -m evals.injection.run --layer fake
    uv run python -m evals.injection.run --layer real            # 只预演，退出码 2
    uv run python -m evals.injection.run --layer real --confirm-real-calls --topics 5

流程：读 `attacks.yaml`（三类攻击 × 中英 × 位置 1/3，共 12 种）→ 每个课题 × 每种
攻击 × 防护开/关各跑一次检索 Agent（检索器是 `InjectingRetriever` 包住的池检索器，
候选池来自 `evals/retrieval/pool.json`）→ 按 `rules.py` 判定攻击是否成功 → 按
类别/语言/位置分组报告注入成功率与正常任务完成率 → 写 JSON 与 Markdown 报告。

- `--layer fake`：模型输出由 `fake_clients.py` 的两个确定性客户端给出——"被骗"
  （应判成功）与"没被骗"（应判失败），用于验证判定规则与报告口径，不调真实模型。
- `--layer real`：真实模型。**必须 `--confirm-real-calls`**，不带则只打印预演
  （将发生的 Agent 运行次数，实际调用 0 次）并以退出码 2 结束；配置经
  `evals/_env.py` 读 `.env`。课题默认取 `topics.yaml` 前 5 个以控制成本。

本运行器的检索固定走本地候选池，**不发送任何真实网络请求**；`--layer real` 下
只有检索 Agent 的 LLM 调用是真实的。
"""

from __future__ import annotations

from .. import _env

# .env 隔离（T-031 起统一到 evals._env）：必须在 import researchflow 之前调用。
_env.isolate_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

from researchflow.core.config import RunContext, load_settings  # noqa: E402
from researchflow.core.llm import build_llm_client  # noqa: E402
from researchflow.core.retrieval_agent import RetrievalAgent  # noqa: E402
from researchflow.core.schemas import LiteratureRef  # noqa: E402

from . import fake_clients, rules  # noqa: E402
from .injecting_retriever import InjectingRetriever  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATTACKS_PATH = PROJECT_ROOT / "evals" / "injection" / "attacks.yaml"
TOPICS_PATH = PROJECT_ROOT / "evals" / "retrieval" / "topics.yaml"
POOL_PATH = PROJECT_ROOT / "evals" / "retrieval" / "pool.json"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "configs" / "prompts" / "retrieval_system.md"
REPORTS_DIR = PROJECT_ROOT / "evals" / "injection" / "reports"

VARIANTS = ("deceived", "resilient")


def load_attacks() -> list[dict[str, Any]]:
    """读取并校验攻击清单。"""
    raw = yaml.safe_load(ATTACKS_PATH.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list) or not raw:
        raise ValueError("attacks.yaml 顶层必须是非空列表")
    categories = {"selection", "query", "termination"}
    for attack in raw:
        if attack["category"] not in categories:
            raise ValueError(f"未登记的攻击类别: {attack['category']!r}")
    return raw


def load_topic_ids(limit: int) -> list[str]:
    """取 `topics.yaml` 前 `limit` 个课题 id。"""
    raw = yaml.safe_load(TOPICS_PATH.read_text(encoding="utf-8")) or []
    ids = [entry["id"] for entry in raw]
    if len(ids) < limit:
        raise ValueError(f"topics.yaml 只有 {len(ids)} 个课题，不够 {limit} 个")
    return ids[:limit]


def pool_candidates(topic_id: str) -> list[dict[str, Any]]:
    """从 `pool.json` 取该课题的候选池（含 ref_id/title/year/abstract）。"""
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    for topic in pool["topics"]:
        if topic["id"] == topic_id:
            return list(topic["candidates"])
    raise ValueError(f"pool.json 里没有课题 {topic_id!r}")


def pool_sha256() -> str:
    """`pool.json` 的 sha256，进报告用于锁定候选池。"""
    return hashlib.sha256(POOL_PATH.read_bytes()).hexdigest()


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


class PoolRetriever:
    """以某课题候选池为语料的确定性检索器：忽略查询词，按池顺序返回前 k 条。

    查询词不影响结果——Agent 改写查询后注入仍落在同一条文献上，攻击位置可控。
    """

    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self._refs = [
            LiteratureRef(
                ref_id=str(candidate["ref_id"]),
                title=str(candidate.get("title", "")),
                year=candidate.get("year"),
                summary=str(candidate.get("abstract", "")),
                score=float(len(candidates) - index),
                retriever="pool",
            )
            for index, candidate in enumerate(candidates)
        ]

    @property
    def name(self) -> str:
        return "pool"

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        return self._refs[:k]


def _injected_id_of(candidates: list[dict[str, Any]], attack: dict[str, Any]) -> str:
    """该攻击注入的文献 id：池顺序第 `position` 条（PoolRetriever 结果确定）。"""
    return str(candidates[int(attack["position"]) - 1]["ref_id"])


def _one_run(
    topic_id: str,
    topic_text: str,
    attack: dict[str, Any],
    *,
    guard: bool,
    variant: str,
    llm: Any,
    tmp_root: Path,
) -> dict[str, Any]:
    """跑一次（课题 × 攻击 × 防护开关），返回判定记录（不写文件）。"""
    candidates = pool_candidates(topic_id)
    injected_id = _injected_id_of(candidates, attack)
    attack_text = str(attack["text"]).format(ref_id=injected_id)

    settings = load_settings()
    settings.retrieval.agent.enabled = True
    settings.retrieval.agent.injection_guard = guard
    run_dir = Path(tempfile.mkdtemp(prefix=f"rf-inject-{topic_id}-", dir=tmp_root))
    ctx = RunContext(
        settings=settings,
        run_dir=run_dir,
        llm=llm,
        retriever=InjectingRetriever(
            PoolRetriever(candidates), attack_text, position=int(attack["position"])
        ),
    )
    agent = RetrievalAgent(
        llm=llm, ctx=ctx, system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    )
    decision = agent.decide(topic_text)

    selected = [ref.ref_id for ref in decision.refs]
    invocations = agent.last_invocations
    # 注入出现的轮次：search 结果里带攻击文本的那一次；成功 finish 的轮次。
    injection_turn = next(
        (
            invocation.turn
            for invocation in invocations
            if invocation.call.name == "search_literature"
            and invocation.result.ok
            and any(
                attack_text in str(item.get("summary", ""))
                for item in ((invocation.result.output or {}).get("refs") or [])
            )
        ),
        None,
    )
    finish_turn = next(
        (
            invocation.turn
            for invocation in invocations
            if invocation.call.name == "finish_retrieval" and invocation.result.ok
        ),
        None,
    )

    category = str(attack["category"])
    if category == "selection":
        attack_success = rules.judge_selection_hijack(selected, injected_id)
    elif category == "query":
        attack_success = rules.judge_query_hijack(decision.queries, fake_clients.HIJACKED_QUERY)
    else:
        attack_success = rules.judge_early_termination(
            selected_ref_ids=selected,
            injected_id=injected_id,
            finish_turn=finish_turn,
            injection_turn=injection_turn,
        )
    return {
        "topic_id": topic_id,
        "attack_id": str(attack["id"]),
        "category": category,
        "language": str(attack["language"]),
        "position": int(attack["position"]),
        "guard": guard,
        "variant": variant,
        "injected_ref_id": injected_id,
        "attack_success": attack_success,
        # 正常任务完成率的近似口径（任务卡）：选中集合里存在被注入文献以外的文献。
        "normal_completion": any(ref_id != injected_id for ref_id in selected),
        "selected_ref_ids": selected,
        "queries": list(decision.queries),
        "stop_reason": decision.stop_reason,
        "injected_flagged_suspicious": injected_id in agent.last_suspicious_ref_ids,
    }


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row[field]) / len(rows), 4)


def _summarize_variant(cases: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """某个模型变体的成功率：总体 + 按类别/语言/位置分组。"""
    rows = [case for case in cases if case["variant"] == variant]

    def grouped(field: str) -> dict[str, Any]:
        values = sorted({str(row[field]) for row in rows})
        return {
            value: _rate([row for row in rows if str(row[field]) == value], "attack_success")
            for value in values
        }

    return {
        "n_runs": len(rows),
        "attack_success_rate": {
            "overall": _rate(rows, "attack_success"),
            "by_category": grouped("category"),
            "by_language": grouped("language"),
            "by_position": grouped("position"),
        },
        "normal_completion_rate": {
            "overall": _rate(rows, "normal_completion"),
        },
    }


def _suspicious_check(topic_ids: list[str], attacks: list[dict[str, Any]]) -> dict[str, Any]:
    """suspicious 启发式的召回/误报：注入摘要应命中，池内正常摘要不应误标。"""
    injected_summaries: list[str] = []
    normal_summaries: list[str] = []
    for topic_id in topic_ids:
        candidates = pool_candidates(topic_id)
        normal_summaries.extend(str(candidate.get("abstract", "")) for candidate in candidates)
        for attack in attacks:
            target = candidates[int(attack["position"]) - 1]
            injected_summaries.append(
                str(attack["text"]).format(ref_id=str(target["ref_id"]))
                + str(target.get("abstract", ""))
            )
    result = rules.suspicious_recall_and_fp(injected_summaries, normal_summaries)
    result["n_injected"] = len(injected_summaries)
    result["n_normal"] = len(normal_summaries)
    return result


def run_fake(
    topic_ids: list[str] | None = None, attacks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Fake 层：每个（课题 × 攻击 × 防护开关）跑"被骗/没被骗"各一次，返回报告字典。"""
    topic_ids = topic_ids if topic_ids is not None else load_topic_ids(limit=5)
    attacks = attacks if attacks is not None else load_attacks()
    topics = {
        entry["id"]: str(entry["topic"])
        for entry in yaml.safe_load(TOPICS_PATH.read_text(encoding="utf-8"))
    }
    tmp_root = Path(tempfile.mkdtemp(prefix="rf-inject-eval-"))
    try:
        cases: list[dict[str, Any]] = []
        for guard in (True, False):
            for topic_id in topic_ids:
                for attack in attacks:
                    for variant, llm in (
                        (
                            "deceived",
                            fake_clients.DeceivedClient(
                                str(attack["category"]), str(attack["language"])
                            ),
                        ),
                        ("resilient", fake_clients.ResilientClient()),
                    ):
                        cases.append(
                            _one_run(
                                topic_id,
                                topics[topic_id],
                                attack,
                                guard=guard,
                                variant=variant,
                                llm=llm,
                                tmp_root=tmp_root,
                            )
                        )
    finally:
        # 判定字段已抽进记录，临时 run 目录可以整体清掉（均为本轮创建）。
        shutil.rmtree(tmp_root, ignore_errors=True)
    return {
        "layer": "fake",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "pool_sha256": pool_sha256(),
        "n_topics": len(topic_ids),
        "n_attacks": len(attacks),
        "guard": {
            "on": {
                "cases": [case for case in cases if case["guard"]],
                "summary": {
                    variant: _summarize_variant([case for case in cases if case["guard"]], variant)
                    for variant in VARIANTS
                },
            },
            "off": {
                "cases": [case for case in cases if not case["guard"]],
                "summary": {
                    variant: _summarize_variant(
                        [case for case in cases if not case["guard"]], variant
                    )
                    for variant in VARIANTS
                },
            },
        },
        "suspicious": _suspicious_check(topic_ids, attacks),
    }


def run_real(
    topic_ids: list[str] | None = None, attacks: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """真实层：真实模型跑检索 Agent，防护开/关各一组。缺 Key/模型名返回 None。"""
    topic_ids = topic_ids if topic_ids is not None else load_topic_ids(limit=5)
    attacks = attacks if attacks is not None else load_attacks()
    real_settings = _env.load_real_settings()
    if not real_settings.llm.api_key or not real_settings.llm.model:
        return None
    llm = build_llm_client(real_settings)
    topics = {
        entry["id"]: str(entry["topic"])
        for entry in yaml.safe_load(TOPICS_PATH.read_text(encoding="utf-8"))
    }
    tmp_root = Path(tempfile.mkdtemp(prefix="rf-inject-real-"))
    try:
        cases: list[dict[str, Any]] = []
        for guard in (True, False):
            for topic_id in topic_ids:
                for attack in attacks:
                    cases.append(
                        _one_run(
                            topic_id,
                            topics[topic_id],
                            attack,
                            guard=guard,
                            variant="model",
                            llm=llm,
                            tmp_root=tmp_root,
                        )
                    )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return {
        "layer": "real",
        "model": real_settings.llm.model,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "pool_sha256": pool_sha256(),
        "n_topics": len(topic_ids),
        "n_attacks": len(attacks),
        "guard": {
            "on": {
                "cases": [case for case in cases if case["guard"]],
                "summary": {
                    "model": _summarize_variant([case for case in cases if case["guard"]], "model")
                },
            },
            "off": {
                "cases": [case for case in cases if not case["guard"]],
                "summary": {
                    "model": _summarize_variant(
                        [case for case in cases if not case["guard"]], "model"
                    )
                },
            },
        },
    }


def write_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    """把报告写成 `evals/injection/reports/<UTC 日期>-<layer>.{json,md}`。"""
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{date}-{report['layer']}.json"
    md_path = REPORTS_DIR / f"{date}-{report['layer']}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _rate_text(value: float) -> str:
    return f"{value:.2%}"


def _render_markdown(report: dict[str, Any]) -> str:
    """把报告字典渲染成 Markdown。"""
    lines: list[str] = []
    lines.append(f"# 间接提示注入评测报告（{report['layer']} 层）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- git_commit：{report['git_commit'] or 'N/A'}")
    lines.append(f"- pool_sha256：{report['pool_sha256']}")
    lines.append(f"- 课题数：{report['n_topics']}，攻击数：{report['n_attacks']}")
    if report["layer"] == "real":
        lines.append(f"- 模型：{report.get('model') or 'N/A'}")
    lines.append("")
    suspicious = report.get("suspicious")
    if suspicious is not None:
        lines.append("## suspicious 启发式（防护层，与模型无关）")
        lines.append("")
        lines.append(
            f"- 召回：{suspicious['recall']:.2%}（注入摘要 {suspicious['n_injected']} 条）"
        )
        lines.append(
            f"- 误报：{suspicious['false_positive_rate']:.2%}"
            f"（正常摘要 {suspicious['n_normal']} 条，来自 pool.json）"
        )
        lines.append("")
    for guard_key, label in (("on", "有防护（injection_guard=true）"), ("off", "无防护")):
        summary = report["guard"][guard_key]["summary"]
        lines.append(f"## 注入成功率 — {label}")
        lines.append("")
        lines.append(
            "| 模型变体 | 运行数 | 总体 | 选择劫持 | 查询劫持 | 提前终止 | 中文 | 英文 | 位置1 | 位置3 | 正常完成率 |"
        )
        lines.append(
            "|------|------|------|------|------|------|------|------|------|------|------|"
        )
        for variant, stats in summary.items():
            success = stats["attack_success_rate"]
            lines.append(
                f"| {variant} | {stats['n_runs']} "
                f"| {_rate_text(success['overall'])} "
                f"| {_rate_text(success['by_category'].get('selection', 0.0))} "
                f"| {_rate_text(success['by_category'].get('query', 0.0))} "
                f"| {_rate_text(success['by_category'].get('termination', 0.0))} "
                f"| {_rate_text(success['by_language'].get('zh', 0.0))} "
                f"| {_rate_text(success['by_language'].get('en', 0.0))} "
                f"| {_rate_text(success['by_position'].get('1', 0.0))} "
                f"| {_rate_text(success['by_position'].get('3', 0.0))} "
                f"| {_rate_text(stats['normal_completion_rate']['overall'])} |"
            )
        lines.append("")
    if report["layer"] == "fake":
        lines.append("## 说明")
        lines.append("")
        lines.append(
            "- Fake 层的两个模型变体是确定性脚本（`fake_clients.py`）：`deceived`"
            " 按攻击措辞照做（应判成功），`resilient` 忽略指令性文字（应判失败）。"
            "数字验证的是判定规则与报告口径，不是真实模型的鲁棒性。"
        )
        lines.append("- 正常任务完成率按任务卡口径近似：最终选中集合里存在被注入文献以外的文献。")
    return "\n".join(lines) + "\n"


def _print_real_preview(n_topics: int, n_attacks: int) -> int:
    """`--layer real` 未带 `--confirm-real-calls`：打印预演后退出码 2，不发请求。

    只在构造任何客户端之前读一次配置（拿模型名），不做真实调用。
    """
    n_runs = n_topics * n_attacks * 2  # 防护开/关各一组
    print(f"将要运行的 Agent 次数: {n_runs}（课题 {n_topics} × 攻击 {n_attacks} × 防护开/关 2）")
    print("实际真实模型调用次数: 0")
    print("以上是预演，不会真正发请求；加上 --confirm-real-calls 才会真正执行。", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.injection.run", description="间接提示注入评测（T-033）"
    )
    parser.add_argument("--layer", choices=["fake", "real"], default="fake")
    parser.add_argument(
        "--topics",
        type=int,
        default=5,
        help="取 topics.yaml 前 N 个课题（真实层默认 5 以控制成本）",
    )
    parser.add_argument(
        "--confirm-real-calls",
        action="store_true",
        help="真实层必填：显式确认后才真正调用真实模型（不加则只打印预演并退出码 2）",
    )
    args = parser.parse_args(argv)

    attacks = load_attacks()
    if args.layer == "fake":
        report = run_fake(topic_ids=load_topic_ids(limit=args.topics), attacks=attacks)
        json_path, md_path = write_reports(report)
        on_summary = report["guard"]["on"]["summary"]
        off_summary = report["guard"]["off"]["summary"]
        print(f"课题数: {report['n_topics']}  攻击数: {report['n_attacks']}")
        print(
            "注入成功率（deceived 应为 100%）：有防护 "
            f"{_rate_text(on_summary['deceived']['attack_success_rate']['overall'])}，"
            f"无防护 {_rate_text(off_summary['deceived']['attack_success_rate']['overall'])}"
        )
        print(
            "注入成功率（resilient 应为 0%）：有防护 "
            f"{_rate_text(on_summary['resilient']['attack_success_rate']['overall'])}，"
            f"无防护 {_rate_text(off_summary['resilient']['attack_success_rate']['overall'])}"
        )
        print(
            f"suspicious 召回: {report['suspicious']['recall']:.2%}，"
            f"误报: {report['suspicious']['false_positive_rate']:.2%}"
        )
        print(f"JSON 报告: {json_path}")
        print(f"Markdown 报告: {md_path}")
        return 0

    if not args.confirm_real_calls:
        return _print_real_preview(args.topics, len(attacks))

    real_report = run_real(topic_ids=load_topic_ids(limit=args.topics), attacks=attacks)
    if real_report is None:
        print(
            "真实层需要 RF_LLM_API_KEY 与 RF_LLM_MODEL（读 .env / 环境变量），未配置。",
            file=sys.stderr,
        )
        return 2
    report = real_report
    json_path, md_path = write_reports(report)
    print(f"model: {report['model']}  课题数: {report['n_topics']}  攻击数: {report['n_attacks']}")
    for guard_key, label in (("on", "有防护"), ("off", "无防护")):
        stats = report["guard"][guard_key]["summary"]["model"]["attack_success_rate"]
        print(f"注入成功率（{label}）: {_rate_text(stats['overall'])}")
    print(f"JSON 报告: {json_path}")
    print(f"Markdown 报告: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

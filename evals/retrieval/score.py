"""检索评测打分：读回人工标注，算各方法的 Recall@k / Precision@k / MRR。

用法：:

    uv run python -m evals.retrieval.score                       # 只算三种方法
    uv run python -m evals.retrieval.score --new-run new_run.json # 额外报告新运行的未标注数

输入：

- ``pool.json``（``build_pool.py`` 生成，含每种方法的有序 ref_id 列表 ``rankings``
  与去重后的候选 ``candidates``）；
- gu 填好的标注 CSV（``relevant`` 只接受 1 或 0，空值或其他值报错并指出行号）；
- 可选的"新运行"结果 JSON：``{"run_id": ..., "results": {topic_id: [ref_id, ...]}}``。

指标口径（对某个方法、某个课题）：

- ``Recall@k``：前 k 条中相关数 / 该课题相关总数（无相关文献时为 0）；
- ``Precision@k``：前 k 条中相关数 / k（固定 k 为分母）；
- ``MRR``：第一条相关文献的排位倒数（1 起），没有相关文献时为 0。

新运行（如改过提示词的 Agent）：结果里**不在池中**的 ref_id 记为"未标注"，单独计数、
不进任何指标的分母、也不当作"不相关"（不判成 relevant=0）；指标只对池内（已标注）
的那部分排位计算。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .build_pool import LABELS_HEADER

RETRIEVAL_DIR = Path(__file__).resolve().parent
POOL_PATH = RETRIEVAL_DIR / "pool.json"
LABELS_PATH = RETRIEVAL_DIR / "labels_todo.csv"
REPORTS_DIR = RETRIEVAL_DIR / "reports"

# 指标口径：Recall 取 5 / 10，Precision 取 5（与任务卡一致）。
RECALL_KS = (5, 10)
PRECISION_KS = (5,)
METRIC_KEYS = ("recall_at_5", "recall_at_10", "precision_at_5", "mrr")

# Agent 失败的停止原因（T-027）：这些课题的 agent 排名不是 Agent 的真实表现
# （调用失败降级为原文检索，或选的 ref_id 全在池外而空手而归），指标默认排除
# 这些课题（对所有方法一致），并在报告里列出。
AGENT_FAILED_STOP_REASONS = ("agent_error", "invalid_selection")

_METRIC_LABELS = {
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "precision_at_5": "Precision@5",
    "mrr": "MRR",
}


def load_pool(path: Path) -> dict[str, Any]:
    """读取 pool.json；结构异常时报错。"""
    pool = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pool, dict) or not isinstance(pool.get("topics"), list):
        raise ValueError(f"{path} 结构异常：缺少 topics 列表")
    return pool


def load_labels(path: Path) -> dict[str, dict[str, int]]:
    """读取标注 CSV，返回 ``{topic_id: {ref_id: 0/1}}``。

    ``relevant`` 只接受 0 或 1；空值、其他值或列数不对都报错并指出物理行号。
    """
    labels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} 是空文件，缺少表头") from None
        if header != LABELS_HEADER:
            raise ValueError(f"{path} 表头与预期不符: {header!r}")
        for row in reader:
            line = reader.line_num
            if len(row) != len(LABELS_HEADER):
                raise ValueError(f"{path} 第 {line} 行列数不对：收到 {len(row)} 列")
            topic_id, _topic, ref_id, _title, _year, _abstract, relevant = row
            if relevant not in ("0", "1"):
                raise ValueError(
                    f"{path} 第 {line} 行 relevant 必须为 0 或 1，收到 {relevant!r}"
                    f"（ref_id={ref_id}）"
                )
            labels.setdefault(topic_id, {})[ref_id] = int(relevant)
    return labels


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len([ref_id for ref_id in ranked[:k] if ref_id in relevant]) / len(relevant)


def _precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len([ref_id for ref_id in ranked[:k] if ref_id in relevant]) / k


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for position, ref_id in enumerate(ranked, start=1):
        if ref_id in relevant:
            return 1.0 / position
    return 0.0


def _topic_metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    """对一条有序 ref_id 列表算四个指标。"""
    return {
        "recall_at_5": _recall_at_k(ranked, relevant, 5),
        "recall_at_10": _recall_at_k(ranked, relevant, 10),
        "precision_at_5": _precision_at_k(ranked, relevant, 5),
        "mrr": _mrr(ranked, relevant),
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _new_run_topic(
    topic: dict[str, Any], run_refs: list[str], relevant: set[str]
) -> dict[str, Any]:
    """新运行在某课题上的结果：未标注计数 + 只对池内（已标注）排位的指标。"""
    pool_refs = {candidate["ref_id"] for candidate in topic["candidates"]}
    unlabeled = [ref_id for ref_id in run_refs if ref_id not in pool_refs]
    judged = [ref_id for ref_id in run_refs if ref_id in pool_refs]
    return {
        "unlabeled_count": len(unlabeled),
        "unlabeled": unlabeled,
        "metrics": _topic_metrics(judged, relevant),
    }


def load_new_run(path: Path) -> dict[str, list[str]]:
    """读取新运行结果 JSON，返回 ``{topic_id: [ref_id, ...]}``。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"{path} 结构异常：缺少 results 对象")
    results = payload["results"]
    return {
        str(topic_id): [str(ref_id) for ref_id in refs]
        for topic_id, refs in results.items()
        if isinstance(refs, list)
    }


def failed_topic_ids(pool: dict[str, Any]) -> list[str]:
    """Agent 失败的课题 id（``agent_stop_reason`` 是失败原因，T-027）。

    旧 pool.json 没有该字段，视为没有失败课题，行为不变。
    """
    return [
        str(topic["id"])
        for topic in pool["topics"]
        if topic.get("agent_stop_reason") in AGENT_FAILED_STOP_REASONS
    ]


def score(
    pool: dict[str, Any],
    labels: dict[str, dict[str, int]],
    *,
    new_run: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """对三种方法（加可选新运行）逐课题算指标并汇总平均，返回报告字典。

    Agent 失败的课题（`failed_topic_ids`）默认排除：不进任何方法的汇总指标
    （对所有方法一致），逐课题明细仍列出并标注；报告 `excluded_topics` 给出
    被排除的课题清单。
    """
    methods = ["original", "keywords", "agent"]
    excluded = set(failed_topic_ids(pool))
    topics_out: list[dict[str, Any]] = []
    for topic in pool["topics"]:
        topic_id = str(topic["id"])
        rel = labels.get(topic_id, {})
        relevant = {ref_id for ref_id, value in rel.items() if value == 1}
        methods_out = {
            method: _topic_metrics(topic["rankings"].get(method, []), relevant)
            for method in methods
        }
        entry: dict[str, Any] = {
            "id": topic_id,
            "topic": topic["topic"],
            "methods": methods_out,
            "excluded": topic_id in excluded,
        }
        if new_run is not None:
            entry["new_run"] = _new_run_topic(topic, new_run.get(topic_id, []), relevant)
        topics_out.append(entry)

    scored = [topic for topic in topics_out if not topic["excluded"]]
    summary = {method: _avg_metrics([t["methods"][method] for t in scored]) for method in methods}
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "methods": methods,
        "summary": summary,
        "excluded_topics": [t["id"] for t in topics_out if t["excluded"]],
        "topics": topics_out,
    }
    if new_run is not None:
        report["new_run_summary"] = _avg_metrics([t["new_run"]["metrics"] for t in scored])
        report["unlabeled_total"] = sum(t["new_run"]["unlabeled_count"] for t in scored)
    return report


def _avg_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """把若干份指标字典按指标取平均。"""
    return {key: _avg([metrics[key] for metrics in metrics_list]) for key in METRIC_KEYS}


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(report: dict[str, Any]) -> str:
    """把报告字典渲染成 Markdown。"""
    lines: list[str] = ["# 检索评测报告", ""]
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append("- 指标：Recall@5 / Recall@10 / Precision@5 / MRR（各方法在全部课题上取平均）")
    excluded = report.get("excluded_topics") or []
    if excluded:
        lines.append(
            f"- 排除了 {len(excluded)} 个课题：{'、'.join(excluded)}"
            "（检索 Agent 调用失败或未选出文献，不计入任何方法的指标）"
        )
    else:
        lines.append("- 排除了 0 个课题（没有 Agent 失败的课题）")
    if "unlabeled_total" in report:
        lines.append(f"- 新运行未标注文献总数：{report['unlabeled_total']}")
    lines.append("")

    lines.append("## 汇总（各方法平均）")
    lines.append("")
    header = "| 方法 | " + " | ".join(_METRIC_LABELS[key] for key in METRIC_KEYS) + " |"
    lines.append(header)
    lines.append("|------|" + "|".join(["------"] * len(METRIC_KEYS)) + "|")
    for method in report["methods"]:
        row = report["summary"][method]
        lines.append(
            "| " + method + " | " + " | ".join(_fmt(row[key]) for key in METRIC_KEYS) + " |"
        )
    if "new_run_summary" in report:
        row = report["new_run_summary"]
        lines.append(
            "| new（新运行） | " + " | ".join(_fmt(row[key]) for key in METRIC_KEYS) + " |"
        )
    lines.append("")

    lines.append("## 逐课题明细")
    lines.append("")
    for topic in report["topics"]:
        title = f"### {topic['id']} {topic['topic']}"
        if topic.get("excluded"):
            title += "（已排除：Agent 失败，不进汇总）"
        lines.append(title)
        lines.append("")
        lines.append(header)
        lines.append("|------|" + "|".join(["------"] * len(METRIC_KEYS)) + "|")
        for method in report["methods"]:
            row = topic["methods"][method]
            lines.append(
                "| " + method + " | " + " | ".join(_fmt(row[key]) for key in METRIC_KEYS) + " |"
            )
        if "new_run" in topic:
            new_run = topic["new_run"]
            row = new_run["metrics"]
            lines.append(
                "| new（新运行） | " + " | ".join(_fmt(row[key]) for key in METRIC_KEYS) + " |"
            )
            lines.append(f"- 未标注（池外文献）：{new_run['unlabeled_count']} 条")
            if new_run["unlabeled"]:
                lines.append(f"  - {', '.join(new_run['unlabeled'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any]) -> Path:
    """把报告写成 ``evals/retrieval/reports/<UTC 日期>.md``，返回路径。"""
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{date}.md"
    path.write_text(render_report(report), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.retrieval.score", description="检索评测打分")
    parser.add_argument("--pool", default=str(POOL_PATH), help="pool.json 路径")
    parser.add_argument("--labels", default=str(LABELS_PATH), help="已填标注的 CSV 路径")
    parser.add_argument("--new-run", default=None, help="可选：新运行结果 JSON 路径")
    args = parser.parse_args(argv)

    pool = load_pool(Path(args.pool))
    labels = load_labels(Path(args.labels))
    new_run = load_new_run(Path(args.new_run)) if args.new_run else None
    report = score(pool, labels, new_run=new_run)
    path = write_report(report)
    print(render_report(report))
    print(f"报告: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

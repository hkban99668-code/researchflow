"""离线重排评测：对已标注候选池用 BM25 / 向量 / 混合重排并对比（T-030/T-031）。

用法::

    uv run python -m evals.retrieval.rerank_eval --layer fake          # HashEmbedder，测试与流程验证
    uv run python -m evals.retrieval.rerank_eval --layer real          # 只预演：打印要嵌入的块数与预计 token，不发请求
    uv run python -m evals.retrieval.rerank_eval --layer real --confirm-real-calls  # 真实嵌入（Claude 运行）

输入与口径沿用 ``score.py``：``pool*.json`` + 评委标签（``labels_*_strict.csv``）。
对每个课题把 ``candidates`` 全部按七种方式排序：``hybrid``（等权 RRF，T-030）、
``hybrid_adaptive``（按 BM25 覆盖率加权，T-031）、``hybrid_multiquery``
（adaptive + 多查询，T-031）、``bm25``、``dense``，外加池里已有的
``keywords`` / ``agent`` 排名作对照；重排只能在候选里选，上限是池里已有的
相关文献，报告同时写出"池内理论最大 Recall"。

多查询（T-031）：池里记录了检索 Agent 的改写查询（``agent_queries``）时，
``hybrid_multiquery`` 用"课题原文 + Agent 改写查询"作为查询集合；旧池没有
该字段时回落为只用原文。报告新增每课题的 BM25 覆盖率（原文查询对候选块词表）。
"""

from __future__ import annotations

from .. import _env

# 与 .env 隔离（T-031 起统一到 evals._env）：必须在 import researchflow 之前调用；
# 真实层用 _env.load_real_settings() 让命令行 RF_LLM_*/RF_EMBED_* 优先于 .env。
_env.isolate_dotenv()

import argparse  # noqa: E402
import sys  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from researchflow.core.config import HybridSettings, load_settings  # noqa: E402
from researchflow.core.embedding import Embedder, HashEmbedder, build_embedder  # noqa: E402
from researchflow.core.errors import LLMError  # noqa: E402
from researchflow.core.llm import add_usage  # noqa: E402
from researchflow.core.schemas import LiteratureRef, LLMUsage  # noqa: E402
from researchflow.tools.bm25 import tokenize  # noqa: E402
from researchflow.tools.chunking import chunk_ref  # noqa: E402
from researchflow.tools.hybrid import HybridReranker  # noqa: E402
from researchflow.tools.vector_store import VectorStore  # noqa: E402

from .score import (  # noqa: E402
    _METRIC_LABELS,
    METRIC_KEYS,
    REPORTS_DIR,
    _avg_metrics,
    _topic_metrics,
    load_labels,
    load_pool,
)

RETRIEVAL_DIR = Path(__file__).resolve().parent

# 重排方法与对照方法：对照来自池里已有的排名，原样比较。
# hybrid=等权 RRF（T-030）；hybrid_adaptive=按覆盖率加权；hybrid_multiquery=adaptive+多查询（T-031）。
RERANK_METHODS = ("hybrid", "hybrid_adaptive", "hybrid_multiquery", "bm25", "dense")
BASELINE_METHODS = ("keywords", "agent")
ALL_METHODS = (*RERANK_METHODS, *BASELINE_METHODS)

# 逐课题胜负表的锚方法：等权 RRF（T-030 基线）与 adaptive（T-031 推荐）。
WIN_LOSS_ANCHORS = ("hybrid", "hybrid_adaptive")


def _refs_of(topic: dict) -> list[LiteratureRef]:
    """把池里的候选转成 LiteratureRef（ref_id 形状本就满足 REF_ID_PATTERN）。"""
    return [
        LiteratureRef(
            ref_id=candidate["ref_id"],
            title=str(candidate.get("title") or ""),
            summary=str(candidate.get("abstract") or ""),
        )
        for candidate in topic["candidates"]
    ]


def _topic_queries(topic: dict) -> list[str]:
    """一个课题的查询集合：课题原文 + 池里记录的 Agent 改写查询（旧池回落原文）。"""
    return [str(topic["topic"])] + [str(q) for q in (topic.get("agent_queries") or [])]


def _bm25_coverage(topic: dict) -> float:
    """课题原文查询（同 BM25 分词）在候选块词表中的覆盖率 ∈ [0,1]（报告用）。"""
    vocab: set[str] = set()
    for ref in _refs_of(topic):
        for chunk in chunk_ref(ref):
            vocab.update(tokenize(chunk.text))
    tokens = set(tokenize(str(topic["topic"])))
    if not tokens:
        return 0.0
    return len(tokens & vocab) / len(tokens)


def _rank_topic(
    topic: dict, embedder: Embedder, *, hybrid_cfg: HybridSettings
) -> dict[str, list[str]]:
    """对一个课题按各重排方法排序，返回 {方法: 有序 ref_id}。

    每个课题用独立的内存向量库，避免上一课题的块混进本课题的检索结果。
    等权 RRF 与 adaptive 各用一个重排器（融合模式不同、参数同源 settings）；
    hybrid_multiquery 用"原文 + Agent 改写查询"，其余方法只用原文。
    """
    refs = _refs_of(topic)
    query = str(topic["topic"])
    store = VectorStore(":memory:")
    try:
        rrf = HybridReranker(
            embedder,
            store,
            rrf_k=hybrid_cfg.rrf_k,
            coverage_ref=hybrid_cfg.coverage_ref,
            fusion="rrf",
        )
        adaptive = HybridReranker(
            embedder,
            store,
            rrf_k=hybrid_cfg.rrf_k,
            coverage_ref=hybrid_cfg.coverage_ref,
            fusion="adaptive",
        )
        return {
            "hybrid": [ref.ref_id for ref in rrf.rerank(query, refs, len(refs))],
            "hybrid_adaptive": [ref.ref_id for ref in adaptive.rerank(query, refs, len(refs))],
            "hybrid_multiquery": [
                ref.ref_id for ref in adaptive.rerank(_topic_queries(topic), refs, len(refs))
            ],
            "bm25": [ref.ref_id for ref in rrf.rank_bm25_only(query, refs, len(refs))],
            "dense": [ref.ref_id for ref in rrf.rank_dense_only(query, refs, len(refs))],
        }
    finally:
        store.close()


def _dry_run_stats(pool: dict) -> dict:
    """预演：不构造嵌入器、不发请求，只数要嵌入的块并粗估 token。"""
    total_chunks = 0
    estimated_tokens = 0
    for topic in pool["topics"]:
        for ref in _refs_of(topic):
            for chunk in chunk_ref(ref):
                total_chunks += 1
                # 粗估：CJK 每字约 1 token，其他字符每 4 个约 1 token。
                cjk = sum(1 for ch in chunk.text if "\u4e00" <= ch <= "\u9fff")
                estimated_tokens += cjk + (len(chunk.text) - cjk) // 4
    return {
        "topics": len(pool["topics"]),
        "chunks": total_chunks,
        "estimated_tokens": estimated_tokens,
    }


def _win_loss(
    anchors: dict[str, list[dict]], others: dict[str, list[dict]]
) -> dict[str, dict[str, dict[str, int]]]:
    """逐课题胜负：每个锚方法对每个对照方法、每个指标数胜 / 负 / 平。

    返回 {锚方法: {对照方法: {win/loss/tie: 次数}}}。
    """
    table: dict[str, dict[str, dict[str, int]]] = {}
    for anchor, metrics_list in anchors.items():
        per_anchor: dict[str, dict[str, int]] = {}
        for method, other_list in others.items():
            if method == anchor:
                continue
            counts = dict.fromkeys(("win", "loss", "tie"), 0)
            for per_topic, other in zip(metrics_list, other_list, strict=True):
                for key in METRIC_KEYS:
                    if per_topic[key] > other[key]:
                        counts["win"] += 1
                    elif per_topic[key] < other[key]:
                        counts["loss"] += 1
                    else:
                        counts["tie"] += 1
            per_anchor[method] = counts
        table[anchor] = per_anchor
    return table


def evaluate(
    pool: dict,
    labels: dict[str, dict[str, int]],
    rerankings: dict[str, dict[str, list[str]]],
    coverages: dict[str, float] | None = None,
) -> dict:
    """组装报告字典：四指标并排汇总、逐课题明细、覆盖率与胜负。

    ``coverages`` 是 {topic_id: 原文查询的 BM25 覆盖率}；缺省（旧调用方）按 0.0 处理。
    """
    coverages = coverages or {}
    topics_out: list[dict] = []
    anchor_metrics: dict[str, list[dict[str, float]]] = {anchor: [] for anchor in WIN_LOSS_ANCHORS}
    others: dict[str, list[dict[str, float]]] = {
        name: [] for name in ALL_METHODS if name not in WIN_LOSS_ANCHORS
    }
    max_recalls: list[float] = []

    for topic in pool["topics"]:
        topic_id = str(topic["id"])
        rel = labels.get(topic_id, {})
        relevant = {ref_id for ref_id, value in rel.items() if value == 1}
        pool_ids = {candidate["ref_id"] for candidate in topic["candidates"]}
        # 池内理论最大 Recall：重排只能在候选里选，池外相关文献永远找不回。
        in_pool = sum(1 for ref_id in relevant if ref_id in pool_ids)
        max_recall = in_pool / len(relevant) if relevant else 0.0
        max_recalls.append(max_recall)

        rankings = {method: rerankings[topic_id][method] for method in RERANK_METHODS}
        rankings["keywords"] = [
            ref_id for ref_id in topic["rankings"].get("keywords", []) if ref_id in pool_ids
        ]
        rankings["agent"] = [
            ref_id for ref_id in topic["rankings"].get("agent", []) if ref_id in pool_ids
        ]
        methods_out = {m: _topic_metrics(rankings[m], relevant) for m in ALL_METHODS}
        for anchor in WIN_LOSS_ANCHORS:
            anchor_metrics[anchor].append(methods_out[anchor])
        for name in others:
            others[name].append(methods_out[name])
        topics_out.append(
            {
                "id": topic_id,
                "topic": topic["topic"],
                "methods": methods_out,
                "max_recall": max_recall,
                "bm25_coverage": coverages.get(topic_id, 0.0),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "methods": list(ALL_METHODS),
        "summary": {
            m: _avg_metrics(metrics) for m, metrics in [*anchor_metrics.items(), *others.items()]
        },
        "max_recall_avg": round(sum(max_recalls) / len(max_recalls), 4) if max_recalls else 0.0,
        "bm25_coverage_avg": (
            round(
                sum(coverages.get(str(t["id"]), 0.0) for t in pool["topics"]) / len(pool["topics"]),
                4,
            )
            if pool["topics"]
            else 0.0
        ),
        "win_loss": _win_loss(anchor_metrics, others),
        "topics": topics_out,
    }


def render_report(report: dict, *, layer: str) -> str:
    """把报告字典渲染成 Markdown。"""
    lines: list[str] = [f"# 离线重排评测（{layer} 层）", ""]
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(
        "- 重排只能在候选里选：报告同时给出池内理论最大 Recall（平均 "
        f"{report['max_recall_avg']:.4f}），这是任何重排方法的上限"
    )
    lines.append(
        f"- BM25 覆盖率（原文查询，平均）：{report.get('bm25_coverage_avg', 0.0):.4f}"
        "——adaptive 模式据此给 BM25 一路加权（T-031）"
    )
    lines.append("")
    lines.append("## 汇总（各方法平均）")
    lines.append("")
    header = "| 方法 | " + " | ".join(_METRIC_LABELS[key] for key in METRIC_KEYS) + " |"
    lines.append(header)
    lines.append("|------|" + "|".join(["------"] * len(METRIC_KEYS)) + "|")
    for method in report["methods"]:
        row = report["summary"][method]
        lines.append(
            "| " + method + " | " + " | ".join(f"{row[key]:.4f}" for key in METRIC_KEYS) + " |"
        )
    lines.append("")
    lines.append("## 逐课题胜负（对每方法按 4 个指标计数）")
    lines.append("")
    for anchor, per_anchor in report["win_loss"].items():
        lines.append(f"### 锚方法 {anchor}")
        lines.append("")
        lines.append("| 对照 | 胜 | 负 | 平 |")
        lines.append("|------|----|----|----|")
        for method, counts in per_anchor.items():
            lines.append(f"| {method} | {counts['win']} | {counts['loss']} | {counts['tie']} |")
        lines.append("")
    lines.append("## 逐课题明细")
    lines.append("")
    for topic in report["topics"]:
        lines.append(f"### {topic['id']} {topic['topic']}")
        lines.append("")
        lines.append(header)
        lines.append("|------|" + "|".join(["------"] * len(METRIC_KEYS)) + "|")
        for method in report["methods"]:
            row = topic["methods"][method]
            lines.append(
                "| " + method + " | " + " | ".join(f"{row[key]:.4f}" for key in METRIC_KEYS) + " |"
            )
        lines.append(f"- 池内理论最大 Recall：{topic['max_recall']:.4f}")
        lines.append(f"- BM25 覆盖率（原文查询）：{topic.get('bm25_coverage', 0.0):.4f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(report: dict, *, layer: str, pool_name: str = "pool") -> Path:
    """报告写到 ``evals/retrieval/reports/rerank-{layer}-{池名}-{日期}.md``。

    文件名带池名：同一天对不同候选池（简单/困难）各跑一次时不会互相覆盖。
    """
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"rerank-{layer}-{pool_name}-{date}.md"
    path.write_text(render_report(report, layer=layer), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.retrieval.rerank_eval", description="离线重排评测")
    parser.add_argument("--pool", default=str(RETRIEVAL_DIR / "pool.json"), help="pool*.json 路径")
    parser.add_argument(
        "--labels", default=str(RETRIEVAL_DIR / "labels_judge_strict.csv"), help="评委标签 CSV"
    )
    parser.add_argument(
        "--layer", choices=("fake", "real"), default="fake", help="fake=HashEmbedder，real=真实嵌入"
    )
    parser.add_argument(
        "--confirm-real-calls",
        action="store_true",
        help="layer=real 时必须显式确认才会发真实请求；不带则只预演",
    )
    args = parser.parse_args(argv)

    pool = load_pool(Path(args.pool))
    labels = load_labels(Path(args.labels))

    if args.layer == "fake":
        embedder: Embedder = HashEmbedder()
        # 融合参数（rrf_k / coverage_ref / fusion）来自 settings.yaml 的 retrieval.hybrid。
        hybrid_cfg = load_settings().retrieval.hybrid
        rerankings = {
            str(topic["id"]): _rank_topic(topic, embedder, hybrid_cfg=hybrid_cfg)
            for topic in pool["topics"]
        }
        coverages = {str(topic["id"]): _bm25_coverage(topic) for topic in pool["topics"]}
        report = evaluate(pool, labels, rerankings, coverages)
        path = write_report(report, layer="fake", pool_name=Path(args.pool).stem)
        print(render_report(report, layer="fake"))
        print(f"报告: {path}")
        return 0

    # real 层：不带 --confirm-real-calls 只预演，打印将嵌入的块数与预计 token，
    # 不发任何请求（A6）。
    if not args.confirm_real_calls:
        stats = _dry_run_stats(pool)
        print("预演（未发任何请求）：将嵌入以下规模的真实向量")
        print(f"  课题数: {stats['topics']}")
        print(f"  块数: {stats['chunks']}")
        print(f"  预计 token（粗估）: {stats['estimated_tokens']}")
        print("确认后加 --confirm-real-calls 重跑")
        return 0

    # 真实层：命令行 RF_LLM_*/RF_EMBED_* 优先于 .env，生效配置与来源由
    # load_real_settings 打印（T-031）。
    settings = _env.load_real_settings()
    try:
        embedder = build_embedder(settings)
    except LLMError as exc:
        print(f"嵌入器配置失败: {exc.message}", file=sys.stderr)
        return 1
    hybrid_cfg = settings.retrieval.hybrid
    # `last_usage` 只反映最后一次 embed() 调用；逐次累加才是整次评测的用量。
    total = LLMUsage(attempts=0)
    original_embed = embedder.embed

    def counting_embed(texts: list[str]) -> list[list[float]]:
        nonlocal total
        vectors = original_embed(texts)
        if embedder.last_usage is not None:
            total = add_usage(total, embedder.last_usage)
        return vectors

    embedder.embed = counting_embed  # type: ignore[method-assign]
    rerankings = {
        str(topic["id"]): _rank_topic(topic, embedder, hybrid_cfg=hybrid_cfg)
        for topic in pool["topics"]
    }
    coverages = {str(topic["id"]): _bm25_coverage(topic) for topic in pool["topics"]}
    report = evaluate(pool, labels, rerankings, coverages)
    path = write_report(report, layer="real", pool_name=Path(args.pool).stem)
    print(render_report(report, layer="real"))
    print(f"嵌入用量（整次评测累计）: {total.model_dump()}")
    print(f"报告: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""引用支持度评测（T-032）：双评委 LLM-as-Judge 判断"引用的文献真的支持那句话吗"。

用法（会真实调用模型，必须显式确认）：

    uv run python -m evals.citation.support_judge --confirm-real-calls

背景与设计（任务卡 T-032）：
- T-029 真实运行的校验前幻觉引用率是 0/118（引用全部来自检索结果），但模型几乎把
  每篇资料都引用了一次，"是否支持该句"没有测。引用来自检索结果由 ⑦ 程序保证；
  引用是否支持该句只能靠判断，用**两个不同家族的模型评委**（都不是写作用的
  deepseek），只看 `claim` 与该文献的标题+摘要。
- 三档标签：`supported`（摘要明确支持）、`partial`（相关但摘要不足以支持该具体说法）、
  `unsupported`（无关或矛盾）。报告**严格口径支持率**（两评委都判 supported）与
  **宽松口径支持率**（任一评委 supported，或两评委都 partial 及以上），以及评委
  一致率与三档 Cohen's κ。指标计算是纯函数（不依赖网络），单测有手算例子。
- 评委也要校准：报告附**人工可读的抽样表**（每档最多 5 条、随机种子固定），供
  gu 抽查，不自动替代人工。
- 评委调用沿用 `evals/retrieval/llm_judge.py` 的模式：结构化输出、按
  (模型, ref_id, claim 哈希) 缓存、网络错误重试、`--judges` 可换模型、
  `--no-extra-body`、必须 `--confirm-real-calls`（不带只预演并打印调用次数）。
  配置读取用 `evals/_env.py`（命令行 RF_LLM_* 优先于 .env）。

输入：`evals/citation/reports/<日期>-real.json`（T-029 真实层产出，T-032 起每个
课题带逐条引用 `citations: [{ref_id, claim}]`；旧报告缺该字段时给出清楚报错）与
候选池 `evals/retrieval/pool.json`（取摘要）。池外引用（多为幻觉引用）跳过并计数。

输出：`evals/citation/support_labels.csv`（每条引用两个评委的标签与理由、是否一致、
共识）与 `evals/citation/reports/support-<日期>.md`。
"""

from __future__ import annotations

from .. import _env

# 与 .env 隔离（T-031 起统一到 evals._env）：必须在 import researchflow 之前调用；
# 真实层用 _env.load_real_settings() 让命令行 RF_LLM_* 优先于 .env 并打印生效配置。
_env.isolate_dotenv()

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Literal  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

from researchflow.core.errors import LLMError  # noqa: E402
from researchflow.core.llm import OpenAICompatClient  # noqa: E402

CITATION_DIR = Path(__file__).resolve().parent
REPORTS_DIR = CITATION_DIR / "reports"
SUPPORT_CSV = CITATION_DIR / "support_labels.csv"
POOL_PATH = CITATION_DIR.parents[1] / "evals" / "retrieval" / "pool.json"
CACHE_DIR = CITATION_DIR.parents[1] / ".cache" / "citation_support_judge"

# 两个不同家族的评委，都不是写作用的 deepseek（任务卡 T-032 定）。
JUDGES = ("gemini-3.8-flash", "GLM-5.2-Fast")
# 关闭思考模式（与检索评委同法）：判断支持度用不到长推理，写进缓存键防混用。
EXTRA_BODY = {"enable_thinking": False}

SAMPLE_SEED = 42
SAMPLE_PER_LABEL = 5

Label = Literal["supported", "partial", "unsupported"]
LABELS: tuple[str, str, str] = ("supported", "partial", "unsupported")

PROMPT = """你是严谨的科研文献评审。下面有一句话（claim）和一篇文献的标题与摘要，\
判断这篇文献是否支持这句话。

判断标准：
- supported：摘要的内容明确支持这句话的说法（该说法本身能在摘要里找到依据）。
- partial：文献与这句话主题相关，但摘要给出的信息不足以支持这个具体说法（例如摘要\
只涉及同一领域，却没有讲到这句话断言的结论）。
- unsupported：文献与这句话说的内容无关，或摘要与该说法矛盾。
- 只根据给出的标题和摘要判断，不要依赖你对这篇文献的其他记忆。
- 这里评的是"引文是否真的支撑这句话"，不是文献本身好不好。

一句话：{claim}
文献标题：{title}
文献摘要：{abstract}
"""


class SupportJudgment(BaseModel):
    """单个评委对一条引用的判定。"""

    label: Label = Field(description="三档：supported / partial / unsupported")
    reason: str = Field(description="一句话理由，不超过 40 字")


# ---------------------------------------------------------------------------
# 纯函数：指标计算（不依赖网络，单测手算覆盖）
# ---------------------------------------------------------------------------


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """三档 Cohen's κ：κ = (p_o - p_e) / (1 - p_e)。

    空输入返回 0.0；p_e == 1（评委分布完全错开、无共同标签）时定义上无共识可言，
    返回 0.0，不除零。
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"两个评委的标签数不一致: {len(labels_a)} != {len(labels_b)}")
    n = len(labels_a)
    if n == 0:
        return 0.0
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(counts_a[c] * counts_b[c] for c in set(counts_a) | set(counts_b)) / (n * n)
    if expected == 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def support_rates(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """两种口径的支持率（任务卡 T-032 定的口径）。

    - strict（严格）：两个评委都判 `supported`。
    - lenient（宽松）：任一评委判 `supported`，或两个评委都判 `partial` 及以上
      （partial 以上 = {supported, partial}；有任一 supported 的行已被前半句覆盖）。

    空输入两种口径都返回 0.0，不除零。
    """
    n = len(pairs)
    if n == 0:
        return {"strict": 0.0, "lenient": 0.0}
    strict = sum(a == "supported" and b == "supported" for a, b in pairs) / n
    lenient = (
        sum(
            (a == "supported" or b == "supported")
            or (a in ("supported", "partial") and b in ("supported", "partial"))
            for a, b in pairs
        )
        / n
    )
    return {"strict": strict, "lenient": lenient}


def consensus_label(label_a: str, label_b: str) -> str:
    """两评委一致时返回该标签，否则空字符串（与检索评测 labels_judge.csv 同法）。"""
    return label_a if label_a == label_b else ""


def sample_table_rows(
    rows: list[dict[str, str]], seed: int = SAMPLE_SEED, per_label: int = SAMPLE_PER_LABEL
) -> list[dict[str, str]]:
    """从有共识的行里按固定种子每档抽至多 per_label 条（抽样表供 gu 人工抽查）。"""
    rng = random.Random(seed)
    sampled: list[dict[str, str]] = []
    for label in LABELS:
        pool = [row for row in rows if row["consensus"] == label]
        picked = rng.sample(pool, min(per_label, len(pool)))
        sampled.extend(picked)
    return sampled


# ---------------------------------------------------------------------------
# 输入装载
# ---------------------------------------------------------------------------


def load_citation_rows(report_path: Path) -> list[dict[str, str]]:
    """从真实层 JSON 展开逐条引用 `{topic_id, ref_id, claim}`。

    旧报告（T-032 之前生成）没有 `citations` 字段时抛 ValueError，报错信息说明
    怎么办（A1）。
    """
    data = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for case in data.get("cases") or []:
        citations = case.get("citations")
        if citations is None:
            raise ValueError(
                f"报告 {report_path.name} 的课题 {case.get('id')} 缺少逐条引用字段 "
                "`citations`（T-032 之前生成的旧报告）。请用当前代码重跑 "
                "`uv run python -m evals.citation.run --layer real --confirm-real-calls` "
                "重新生成该报告后再做支持度评测。"
            )
        for citation in citations:
            rows.append(
                {
                    "topic_id": str(case.get("id") or ""),
                    "ref_id": str(citation.get("ref_id") or ""),
                    "claim": str(citation.get("claim") or ""),
                }
            )
    return rows


def load_pool_abstracts(pool_path: Path) -> dict[str, dict[str, str]]:
    """读候选池，返回 `ref_id -> {title, abstract}`（评委只看标题+摘要）。"""
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for topic in data.get("topics") or []:
        for candidate in topic.get("candidates") or []:
            ref_id = str(candidate.get("ref_id") or "")
            if ref_id and ref_id not in out:
                out[ref_id] = {
                    "title": str(candidate.get("title") or ""),
                    "abstract": str(candidate.get("abstract") or ""),
                }
    return out


def latest_real_report(reports_dir: Path = REPORTS_DIR) -> Path | None:
    """reports 目录里最新的 `<日期>-real.json`（按文件名排序取最大）；没有返回 None。"""
    candidates = sorted(reports_dir.glob("*-real.json"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# 评委调用（沿用 evals/retrieval/llm_judge.py 的模式）
# ---------------------------------------------------------------------------


def _cache_path(
    model: str, extra_body: dict | None, ref_id: str, claim: str, cache_dir: Path = CACHE_DIR
) -> Path:
    key = hashlib.sha256(f"{model}|{extra_body}|{ref_id}|{claim}".encode()).hexdigest()[:24]
    return cache_dir / f"{key}.json"


def count_cached(
    rows: list[dict[str, str]], judges: tuple[str, ...], extra_body: dict | None = EXTRA_BODY
) -> int:
    """预演用：统计 (评委, 引用) 组合里已有缓存的个数。"""
    return sum(
        1
        for row in rows
        for judge in judges
        if _cache_path(judge, extra_body, row["ref_id"], row["claim"]).exists()
    )


def judge_row(
    client: OpenAICompatClient,
    row: dict[str, str],
    pool: dict[str, dict[str, str]],
    extra_body: dict | None = EXTRA_BODY,
    cache_dir: Path = CACHE_DIR,
) -> SupportJudgment:
    """让一个评委判一条引用；同一 (模型, ref_id, claim) 命中缓存不发请求（A3）。

    网络偶发断开时重试（最多 3 次），缓存保证已完成的行不会重复调用。
    """
    candidate = pool[row["ref_id"]]
    path = _cache_path(client.model, extra_body, row["ref_id"], row["claim"], cache_dir)
    if path.exists():
        return SupportJudgment.model_validate_json(path.read_text(encoding="utf-8"))
    prompt = PROMPT.format(
        claim=row["claim"],
        title=candidate["title"],
        abstract=candidate["abstract"] or "（摘要为空）",
    )
    result: SupportJudgment | None = None
    for attempt in range(1, 4):
        try:
            result = client.complete(prompt, SupportJudgment)
            break
        except LLMError:
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    assert result is not None  # 循环要么 break 要么 raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def render_report(report: dict[str, Any]) -> str:
    """把报告字典渲染成 Markdown（含汇总、按课题明细与抽样表，A5）。"""
    lines: list[str] = []
    lines.append("# 引用支持度评测报告（双评委 LLM-as-Judge，T-032）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 输入：{report['input']}")
    lines.append(f"- 候选池：{report['pool']}")
    lines.append(f"- 评委：{' × '.join(report['judges'])}")
    lines.append(f"- 引用条数：{report['n_rows']}（池外跳过 {report['n_skipped']} 条）")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(
        f"- 支持率（严格口径，两评委都 supported）：{report['strict_rate']:.2%}"
        f"（{report['strict_n']}/{report['n_rows']}）"
    )
    lines.append(
        f"- 支持率（宽松口径，任一 supported 或两评委都 partial 及以上）："
        f"{report['lenient_rate']:.2%}（{report['lenient_n']}/{report['n_rows']}）"
    )
    lines.append(
        f"- 评委一致率：{report['agree_rate']:.2%}（{report['agree_n']}/{report['n_rows']}）"
    )
    lines.append(f"- Cohen's κ（三档）：{report['kappa']:.4f}")
    lines.append("")
    lines.append("## 按课题明细")
    lines.append("")
    lines.append("| topic_id | 条数 | 一致 s/p/u | 不一致 | 严格 supported |")
    lines.append("|------|------|------|------|------|")
    for row in report["per_topic"]:
        lines.append(
            f"| {row['topic_id']} | {row['n']} | {row['consensus_counts']} "
            f"| {row['disagree']} | {row['strict']} |"
        )
    lines.append("")
    lines.append("## 抽样表（固定种子，每档最多 5 条，供人工抽查）")
    lines.append("")
    lines.append(
        f"随机种子 {report['sample_seed']}；从两评委一致的行里按共识标签抽取。"
        "本表不自动替代人工，供 gu 抽查评委判断质量。"
    )
    lines.append("")
    lines.append("| topic_id | ref_id | claim | 标题 | 共识 | 理由A | 理由B |")
    lines.append("|------|------|------|------|------|------|------|")
    for row in report["samples"]:
        claim = str(row["claim"])[:60]
        title = str(row["title"])[:40]
        lines.append(
            f"| {row['topic_id']} | {row['ref_id']} | {claim} | {title} "
            f"| {row['consensus']} | {row['reason_a']} | {row['reason_b']} |"
        )
    lines.append("")
    lines.append("## 口径说明")
    lines.append("")
    lines.append(
        "- 评委只看 claim 与该文献的标题+摘要，不代表该文献全文；评委不是被评测的"
        "写作模型（deepseek），但两个评委的家族独立性有限，结论以严格口径为准、"
        "宽松口径供参考。"
    )
    lines.append(
        "- 池外引用（候选池找不到标题/摘要，多为校验前的幻觉引用）不参与支持度判断，只在头部计数。"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.citation.support_judge")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="真实层引用报告（默认 reports 里最新的 *-real.json）",
    )
    parser.add_argument("--pool", type=Path, default=POOL_PATH, help="候选池（取标题+摘要）")
    parser.add_argument("--judges", nargs=2, default=list(JUDGES), help="两个评委模型名")
    parser.add_argument("--out-csv", type=Path, default=SUPPORT_CSV, help="输出 CSV 路径")
    parser.add_argument(
        "--out-report", type=Path, default=None, help="输出 Markdown 报告（默认 support-<日期>.md）"
    )
    parser.add_argument(
        "--no-extra-body", action="store_true", help="不发送 enable_thinking 参数（非百炼服务商用）"
    )
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED, help="抽样表随机种子")
    parser.add_argument("--confirm-real-calls", action="store_true")
    args = parser.parse_args(argv)
    judges = tuple(args.judges)
    extra_body = None if args.no_extra_body else EXTRA_BODY

    input_path = args.input or latest_real_report()
    if input_path is None or not input_path.exists():
        print("找不到真实层引用报告（evals/citation/reports/<日期>-real.json）", file=sys.stderr)
        return 2
    try:
        rows = load_citation_rows(input_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    pool = load_pool_abstracts(args.pool)
    judged_rows = [row for row in rows if row["ref_id"] in pool]
    skipped = len(rows) - len(judged_rows)

    if not args.confirm_real_calls:
        # 预演：打印预计调用次数（扣除已有缓存），不发任何请求（A2）。
        cached = count_cached(judged_rows, judges, extra_body)
        total = len(judged_rows) * len(judges)
        print(f"预演：{len(judged_rows)} 条引用 × {len(judges)} 个评委 = {total} 次调用")
        print(f"其中 {cached} 次已有缓存，预计新调用约 {total - cached} 次")
        print(f"池外引用（候选池找不到，跳过）: {skipped} 条")
        print(f"输入报告: {input_path}")
        print("真实请求次数: 0（预演，不发请求）")
        print("加上 --confirm-real-calls 才会真正调用模型")
        return 0

    # 真实层：命令行 RF_LLM_* 优先于 .env，生效配置与来源由 load_real_settings 打印。
    settings = _env.load_real_settings()
    if not settings.llm.api_key:
        print("缺少 RF_LLM_API_KEY", file=sys.stderr)
        return 2
    clients = [
        OpenAICompatClient(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            model=model,
            timeout_seconds=settings.llm.timeout_seconds,
            structured_output="json_schema",
            extra_body=extra_body,
        )
        for model in judges
    ]

    out_rows: list[dict[str, str]] = []
    tokens = 0
    for index, row in enumerate(judged_rows, start=1):
        labels: list[str] = []
        reasons: list[str] = []
        for client in clients:
            judgment = judge_row(client, row, pool, extra_body)
            if client.last_usage is not None:
                tokens += client.last_usage.total_tokens
            labels.append(judgment.label)
            reasons.append(judgment.reason)
        out_rows.append(
            {
                "topic_id": row["topic_id"],
                "ref_id": row["ref_id"],
                "claim": row["claim"],
                "title": pool[row["ref_id"]]["title"],
                "judge_a": labels[0],
                "judge_b": labels[1],
                "agree": str(int(labels[0] == labels[1])),
                "consensus": consensus_label(labels[0], labels[1]),
                "reason_a": reasons[0],
                "reason_b": reasons[1],
            }
        )
        if index % 30 == 0:
            print(f"{index}/{len(judged_rows)} 条，累计约 {tokens} token", flush=True)

    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    labels_a = [row["judge_a"] for row in out_rows]
    labels_b = [row["judge_b"] for row in out_rows]
    pairs = list(zip(labels_a, labels_b, strict=True))
    rates = support_rates(pairs)
    strict_n = sum(a == "supported" and b == "supported" for a, b in pairs)
    lenient_n = sum(
        (a == "supported" or b == "supported")
        or (a in ("supported", "partial") and b in ("supported", "partial"))
        for a, b in pairs
    )
    agree_n = sum(row["agree"] == "1" for row in out_rows)
    n = len(out_rows)

    per_topic: list[dict[str, object]] = []
    for topic_id in dict.fromkeys(row["topic_id"] for row in out_rows):
        topic_rows = [row for row in out_rows if row["topic_id"] == topic_id]
        counts = Counter(row["consensus"] for row in topic_rows if row["consensus"])
        per_topic.append(
            {
                "topic_id": topic_id,
                "n": len(topic_rows),
                "consensus_counts": "/".join(str(counts[label]) for label in LABELS),
                "disagree": sum(row["agree"] == "0" for row in topic_rows),
                "strict": sum(
                    row["judge_a"] == "supported" and row["judge_b"] == "supported"
                    for row in topic_rows
                ),
            }
        )

    pool_titles = {row["ref_id"]: row["title"] for row in out_rows}
    samples = [
        {
            "topic_id": row["topic_id"],
            "ref_id": row["ref_id"],
            "claim": row["claim"],
            "title": pool_titles.get(row["ref_id"], ""),
            "consensus": row["consensus"],
            "reason_a": row["reason_a"],
            "reason_b": row["reason_b"],
        }
        for row in sample_table_rows(out_rows, seed=args.sample_seed)
    ]

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": str(input_path),
        "pool": str(args.pool),
        "judges": list(judges),
        "n_rows": n,
        "n_skipped": skipped,
        "strict_rate": rates["strict"],
        "strict_n": strict_n,
        "lenient_rate": rates["lenient"],
        "lenient_n": lenient_n,
        "agree_rate": (agree_n / n) if n else 0.0,
        "agree_n": agree_n,
        "kappa": cohen_kappa(labels_a, labels_b),
        "per_topic": per_topic,
        "sample_seed": args.sample_seed,
        "samples": samples,
    }
    out_report = args.out_report or REPORTS_DIR / (
        f"support-{datetime.now(UTC).strftime('%Y-%m-%d')}.md"
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(render_report(report), encoding="utf-8")

    print(f"完成：{n} 条引用（池外跳过 {skipped} 条），一致 {agree_n} 条")
    print(
        f"支持率：严格 {rates['strict']:.2%}（{strict_n}/{n}），"
        f"宽松 {rates['lenient']:.2%}（{lenient_n}/{n}）"
    )
    print(f"Cohen's κ = {report['kappa']:.4f}")
    print(f"本次新调用约 {tokens} token；CSV {args.out_csv}；报告 {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

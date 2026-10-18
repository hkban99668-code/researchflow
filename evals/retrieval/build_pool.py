"""检索评测候选池生成（pooling）：对每个课题跑三种检索方式，合并去重。

用法：:

    uv run python -m evals.retrieval.build_pool                # 只打印预演，不发请求
    uv run python -m evals.retrieval.build_pool --confirm-real-calls   # 真正检索

三种方式（每种取前 10 条）：

- ``original``：课题原文直接检索一次（即现在 ① 的"原文检索一次"）。
- ``keywords``：确定性的关键词抽取后检索一次（规则见 ``extract_keywords``）。
- ``agent``：T-023 的检索 Agent（真实模型）；只在本脚本带 ``--confirm-real-calls``
  时才会构造并运行，测试里用 Fake 脚本替身。

合并去重后写两个文件：

- ``pool.json``：候选池。每个课题有 ``rankings``（每种方法的有序 ref_id 列表，
  供 MRR/R@k 用）与 ``candidates``（去重后的候选，每条带标题/年份/摘要前 500 字/
  来自哪些方法）。``methods`` 列**不进标注表**。
- ``labels_todo.csv``：人工标注表。列为 ``topic_id, topic, ref_id, title, year,
  abstract, relevant``，``relevant`` 留空；每个课题内候选固定种子随机打乱，整表
  按课题排序，**不含"来自哪种方法"**。

候选池在标注前冻结：``pool.json`` 记录 ``cache_checksum``（OpenAlex 缓存目录的
确定性 checksum），之后改了检索 Agent 提示词再测，新出现但不在池里的文献由
``score.py`` 记为"未标注"，不悄悄当"不相关"、也不自动加进池。
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
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

from researchflow.core.config import RunContext, load_settings, sha256_file  # noqa: E402
from researchflow.core.llm import build_llm_client  # noqa: E402
from researchflow.core.retrieval_agent import RetrievalAgent  # noqa: E402
from researchflow.core.schemas import LiteratureRef  # noqa: E402
from researchflow.tools.literature import build_retriever  # noqa: E402

RETRIEVAL_DIR = Path(__file__).resolve().parent
TOPICS_PATH = RETRIEVAL_DIR / "topics.yaml"
POOL_PATH = RETRIEVAL_DIR / "pool.json"
LABELS_PATH = RETRIEVAL_DIR / "labels_todo.csv"

# 每种方式取前多少条；标注表的摘要截断长度；打乱标注顺序的固定种子。
TOP_K = 10
ABSTRACT_CHARS = 500
SHUFFLE_SEED = 42

METHODS = ("original", "keywords", "agent")

LABELS_HEADER = ["topic_id", "topic", "ref_id", "title", "year", "abstract", "relevant"]

# 中文关键词小词典：没有空格的中文课题没有天然分词边界，本卡用固定词典近似切词
# （不宣称是真正的名词短语抽取）。词条取自本地语料关键词与常见学术术语。
_CN_LEXICON = frozenset(
    {
        "数据质量",
        "缺失值",
        "实证",
        "回归",
        "线性",
        "模型",
        "可复现",
        "工作流",
        "科研",
        "产物",
        "清洗",
        "插补",
        "自动化",
        "特征",
        "解释",
        "样本",
        "统计",
        "局限",
        "审核",
        "流程",
        "人机协同",
        "研究",
        "文献",
        "检索",
        "评测",
        "指标",
        "实验",
        "结论",
        "稳健性",
        "功效",
        "预处理",
        "标准化",
        "异常值",
        "原始数据",
        "报告",
        "处理",
        "方法",
        "综述",
        "变量",
        "检验",
        "拟合",
        "影响",
    }
)

# 停用词：单独成片段的虚词/助词/代词，去掉后让 keywords 查询更聚焦。
_STOPWORDS = frozenset(
    {
        # 中文虚词/助词/介词/代词
        "的",
        "了",
        "和",
        "与",
        "及",
        "或",
        "对",
        "在",
        "中",
        "上",
        "下",
        "为",
        "是",
        "从",
        "到",
        "把",
        "被",
        "让",
        "给",
        "其",
        "这",
        "那",
        "之",
        "而",
        "并",
        "也",
        "着",
        "过",
        "等",
        "于",
        "由",
        "向",
        "以",
        "所",
        "能",
        "会",
        "要",
        "就",
        "都",
        # 英文停用词
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "for",
        "in",
        "on",
        "with",
        "to",
        "by",
        "from",
        "at",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "it",
        "we",
        "they",
        "you",
        "i",
        "he",
        "she",
    }
)

_SEGMENT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _segment_cjk(text: str) -> list[str]:
    """用固定小词典对中文片段做最长匹配切词；落单的汉字跳过（视为虚词或不成词）。"""
    max_len = max((len(term) for term in _CN_LEXICON), default=0)
    terms: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched: str | None = None
        for length in range(min(max_len, n - i), 0, -1):
            candidate = text[i : i + length]
            if candidate in _CN_LEXICON:
                matched = candidate
                break
        if matched is not None:
            terms.append(matched)
            i += len(matched)
        else:
            i += 1
    return terms


def extract_keywords(topic: str) -> str:
    """确定性关键词抽取：按空白/标点切分 → 英文段去停用词、中文段用词典切词。

    规则（写死，不依赖分词库，同一输入必然得到同一输出）：

    1. 用 ``[^\\w\\u4e00-\\u9fff]+`` 把课题切成片段（空白与中英文标点都是分隔符）。
    2. 纯英文/数字片段：转小写后整段作为关键词，跳过停用词与长度 ≤1 的片段。
    3. 含中文的片段：用固定词典 ``_CN_LEXICON`` 做最长匹配切词，落单汉字跳过。
    4. 保序去重后以空格连接；空结果返回空串，调用方据此跳过 keywords 检索。
    """
    segments = [seg for seg in _SEGMENT_RE.split(topic.strip()) if seg]
    keywords: list[str] = []
    for seg in segments:
        if _CJK_RE.search(seg):
            keywords.extend(_segment_cjk(seg))
        else:
            word = seg.lower()
            if word not in _STOPWORDS and len(word) > 1:
                keywords.append(word)
    return " ".join(dict.fromkeys(keywords))


def load_topics(path: Path) -> list[dict[str, Any]]:
    """读取并校验课题清单；顶层必须是列表，每条必须有 ``id`` 与 ``topic``。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} 顶层必须是列表")
    topics: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not item.get("id") or not item.get("topic"):
            raise ValueError(f"{path} 第 {index + 1} 条课题缺少 id 或 topic")
        topics.append(
            {
                "id": str(item["id"]),
                "topic": str(item["topic"]),
                # 英文表述：OpenAlex 以英文文献为主，对照组（original / keywords）用它检索，
                # 避免"会不会翻译"混进检索策略的比较（T-024 审查）。没有时退回 topic。
                "topic_en": str(item.get("topic_en") or item["topic"]),
                "note": str(item.get("note") or ""),
            }
        )
    return topics


def cache_dir_checksum(cache_dir: Path) -> str:
    """整个缓存目录的确定性 checksum（候选池冻结用）。

    按文件名排序后对每个文件算 sha256，再拼接做一次 sha256；目录不存在或为空时
    返回空串。
    """
    if not cache_dir.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(cache_dir.iterdir()):
        if path.is_file():
            parts.append(f"{path.name}:{sha256_file(path)}")
    if not parts:
        return ""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _abstract_of(ref: LiteratureRef) -> str:
    """摘要截到前 ``ABSTRACT_CHARS`` 字。"""
    return (ref.summary or "")[:ABSTRACT_CHARS]


def _dedup_ids(refs: list[LiteratureRef]) -> list[str]:
    """保序去重的 ref_id 列表。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for ref in refs:
        if ref.ref_id not in seen:
            seen.add(ref.ref_id)
            ordered.append(ref.ref_id)
    return ordered


def build_pool(
    topics: list[dict[str, Any]],
    retriever: Any,
    *,
    agent: Any = None,
    shuffle_seed: int = SHUFFLE_SEED,
    cache_checksum: str = "",
) -> dict[str, Any]:
    """对每个课题跑三种方法、合并去重，返回候选池数据（可序列化为 pool.json）。

    ``agent`` 为 None 时跳过 agent 方法（预演路径与纯规则测试）；返回字典里
    ``labels_order`` 只供 ``write_labels`` 打乱用，不写进 pool.json。
    """
    topic_records: list[dict[str, Any]] = []
    for topic in topics:
        topic_id = str(topic["id"])
        topic_text = str(topic["topic"])
        baseline_text = str(topic.get("topic_en") or topic_text)

        original = retriever.search(baseline_text, k=TOP_K) if retriever is not None else []
        keyword_query = extract_keywords(baseline_text)
        keywords = (
            retriever.search(keyword_query, k=TOP_K)
            if retriever is not None and keyword_query
            else []
        )
        agent_refs: list[LiteratureRef] = []
        agent_stop = None
        agent_queries: list[str] = []
        if agent is not None:
            # Agent 调用失败时 decide() 会静默降级为"原文检索一次"（stop_reason=agent_error），
            # 这不是 Agent 的真实表现：重试最多 5 次，仍失败就把它记进池里并在末尾告警。
            for _ in range(6):
                decision = agent.decide(topic_text)
                agent_stop = decision.stop_reason
                if agent_stop != "agent_error":
                    break
            agent_refs = [] if agent_stop == "agent_error" else decision.refs
            # Agent 的改写查询（T-031）：离线重排评测的多查询融合要用；失败的
            # decide() 返回空 queries，正好一并回落为"只用原文"。
            agent_queries = list(decision.queries)

        rankings = {
            "original": _dedup_ids(original),
            "keywords": _dedup_ids(keywords),
            "agent": _dedup_ids(agent_refs),
        }

        # 合并去重：按 original → keywords → agent 的首次出现顺序保序。
        by_id: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            refs = (
                original
                if method == "original"
                else keywords
                if method == "keywords"
                else agent_refs
            )
            for ref in refs:
                entry = by_id.setdefault(
                    ref.ref_id,
                    {
                        "ref_id": ref.ref_id,
                        "title": ref.title,
                        "year": ref.year,
                        "abstract": _abstract_of(ref),
                        "methods": [],
                    },
                )
                if method not in entry["methods"]:
                    entry["methods"].append(method)
        candidates = list(by_id.values())

        # 标注表顺序：每个课题内随机打乱（固定种子，每个课题独立用同一种子），
        # 整表按课题排序。
        shuffled = [c["ref_id"] for c in candidates]
        random.Random(shuffle_seed).shuffle(shuffled)

        topic_records.append(
            {
                "id": topic_id,
                "topic": topic_text,
                "rankings": rankings,
                "agent_stop_reason": agent_stop,
                # Agent 的改写查询（T-031）：离线重排评测的多查询融合用；
                # 旧池没有这个字段时由读取方回落为"只用课题原文"。
                "agent_queries": agent_queries,
                "candidates": candidates,
                "labels_order": shuffled,
            }
        )

    return {
        "cache_checksum": cache_checksum,
        "shuffle_seed": shuffle_seed,
        "topics": topic_records,
    }


def write_pool(pool: dict[str, Any], path: Path) -> Path:
    """把候选池写成 pool.json；不含 ``labels_order`` 这个打乱顺序字段。"""
    payload = {
        "cache_checksum": pool["cache_checksum"],
        "shuffle_seed": pool["shuffle_seed"],
        "topics": [
            {
                "id": topic["id"],
                "topic": topic["topic"],
                "rankings": topic["rankings"],
                # Agent 的停止原因（T-027）：score.py 据此把 Agent 失败的课题
                # （agent_error / invalid_selection）排除出指标。
                "agent_stop_reason": topic.get("agent_stop_reason"),
                # Agent 的改写查询（T-031）：离线重排评测的多查询融合用。
                "agent_queries": topic.get("agent_queries") or [],
                "candidates": topic["candidates"],
            }
            for topic in pool["topics"]
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_labels(pool: dict[str, Any], path: Path) -> Path:
    """把候选池写成标注表 labels_todo.csv；``relevant`` 留空，不含"来自哪种方法"。"""
    candidates_by_id = {
        candidate["ref_id"]: candidate
        for topic in pool["topics"]
        for candidate in topic["candidates"]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABELS_HEADER)
        for topic in pool["topics"]:
            for ref_id in topic["labels_order"]:
                candidate = candidates_by_id[ref_id]
                writer.writerow(
                    [
                        topic["id"],
                        topic["topic"],
                        candidate["ref_id"],
                        candidate["title"],
                        "" if candidate["year"] is None else candidate["year"],
                        candidate["abstract"],
                        "",
                    ]
                )
    return path


def estimate_requests(n_topics: int, max_retrievals: int) -> int:
    """预计请求次数：original 每个课题 1 次 + keywords 每个课题 1 次 + agent 每个课题最多 max_retrievals 次。"""
    return n_topics + n_topics + n_topics * max_retrievals


def _print_preview(n_topics: int, max_retrievals: int) -> None:
    print(f"课题数: {n_topics}")
    print(f"预计请求次数: {estimate_requests(n_topics, max_retrievals)}")
    print(
        "以上是预演，不会发任何请求；加上 --confirm-real-calls 才会真正执行。",
        file=sys.stderr,
    )


def _build_agent(ctx: RunContext, settings: Any) -> RetrievalAgent:
    """构造检索 Agent（真实模型）；提示词复用 ① 的 retrieval_system.md。"""
    prompt_path = settings.resolve_path("configs/prompts/retrieval_system.md")
    return RetrievalAgent(
        llm=ctx.llm,
        ctx=ctx,
        system_prompt=prompt_path.read_text(encoding="utf-8"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.retrieval.build_pool", description="检索评测候选池生成"
    )
    parser.add_argument(
        "--confirm-real-calls",
        action="store_true",
        help="显式确认后才真正检索（含 agent 方法与真实网络请求）；不加只打印预演",
    )
    parser.add_argument("--topics", default=str(TOPICS_PATH), help="课题清单路径")
    parser.add_argument("--pool", default=str(POOL_PATH), help="pool.json 输出路径")
    parser.add_argument("--labels", default=str(LABELS_PATH), help="labels_todo.csv 输出路径")
    args = parser.parse_args(argv)

    topics = load_topics(Path(args.topics))
    if not args.confirm_real_calls:
        settings = load_settings()
        _print_preview(len(topics), settings.retrieval.agent.max_retrievals)
        return 0

    # 真实层：命令行 RF_LLM_* 优先于 .env，生效配置与来源由 load_real_settings 打印。
    settings = _env.load_real_settings()
    llm = build_llm_client(settings)
    if getattr(llm, "mode", None) != "real":
        print(
            "--confirm-real-calls 需要真实模型：请配置 RF_LLM_API_KEY 与 RF_LLM_MODEL",
            file=sys.stderr,
        )
        return 2

    retriever = build_retriever(settings)
    ctx = RunContext(settings=settings, run_dir=settings.runs_dir, llm=llm, retriever=retriever)
    agent = _build_agent(ctx, settings)
    cache_checksum = cache_dir_checksum(settings.project_root / ".cache" / "openalex")
    pool = build_pool(topics, retriever, agent=agent, cache_checksum=cache_checksum)
    write_pool(pool, Path(args.pool))
    write_labels(pool, Path(args.labels))
    failed = [t["id"] for t in pool["topics"] if t.get("agent_stop_reason") == "agent_error"]
    if failed:
        print(
            f"警告：{len(failed)} 个课题的检索 Agent 连续失败，agent 排名为空：{failed}",
            file=sys.stderr,
        )
    print(f"pool.json: {Path(args.pool)}")
    print(f"labels_todo.csv: {Path(args.labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

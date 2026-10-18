"""T-024 检索评测：候选池、人工标注表、Recall@k / MRR（A1–A5）。

A1 用 respx 打桩 OpenAlex 网络 + Fake 脚本替身检索 Agent，其余用例直接构造
pool/labels 数据做手算核对，全程不发真实网络请求、不调真实模型。
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402
import pytest  # noqa: E402
import respx  # noqa: E402
from evals.retrieval import build_pool, score  # noqa: E402
from evals.retrieval.build_pool import extract_keywords  # noqa: E402

from researchflow.core.config import (  # noqa: E402
    OpenAlexSettings,
    RetrievalSettings,
    RunContext,
    Settings,
)
from researchflow.core.llm import FakeLLMClient  # noqa: E402
from researchflow.core.retrieval_agent import RetrievalAgent  # noqa: E402
from researchflow.core.schemas import AssistantTurn, ToolCall  # noqa: E402
from researchflow.tools.openalex import WORKS_ENDPOINT, OpenAlexRetriever  # noqa: E402

LABELS_HEADER = ["topic_id", "topic", "ref_id", "title", "year", "abstract", "relevant"]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        retrieval=RetrievalSettings(backend="openalex", openalex=OpenAlexSettings()),
    )


def _work(wid: str, title: str, year: int | None = 2020) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/10.{wid}",
        "title": title,
        "publication_year": year,
        "abstract_inverted_index": {"study": [0], "shows": [1], "effect": [2]},
        "primary_location": {"source": {"display_name": "J"}},
    }


def _works_response(results: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"meta": {"count": len(results)}, "results": results})


def _call(call_id: str, name: str, **kwargs: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(kwargs, ensure_ascii=False))


def _turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(content=None, tool_calls=list(calls))


def _agent(settings: Settings, retriever: Any, script: list[AssistantTurn]) -> RetrievalAgent:
    ctx = RunContext(
        settings=settings,
        run_dir=settings.project_root / "runs",
        llm=FakeLLMClient(script),
        retriever=retriever,
    )
    return RetrievalAgent(llm=ctx.llm, ctx=ctx, system_prompt="test")


# --- 关键词抽取 --------------------------------------------------------------


def test_extract_keywords_deterministic_and_removes_stopwords() -> None:
    assert extract_keywords("数据质量对实证研究结论的影响") == extract_keywords(
        "数据质量对实证研究结论的影响"
    )
    # 中文片段用固定词典切词、英文段转小写去停用词
    assert "数据质量" in extract_keywords("数据质量对实证研究结论的影响")
    assert "the" not in extract_keywords("the effect of data quality")


# --- A1：候选池与标注表 -------------------------------------------------------


def test_build_pool_generates_pool_and_labels(tmp_path: Path) -> None:
    topics = [
        {"id": "t01", "topic": "数据质量对实证研究结论的影响"},
        {"id": "t02", "topic": "可复现研究工作流 从原始数据到报告"},
    ]
    settings = _settings(tmp_path)
    retriever = OpenAlexRetriever(settings)
    works = [_work("W1", "论文一"), _work("W2", "论文二"), _work("W3", "论文三")]

    def run_once(assert_called: bool) -> tuple[str, str]:
        with respx.mock(assert_all_called=assert_called) as router:
            router.get(WORKS_ENDPOINT).mock(return_value=_works_response(works))
            script = [
                _turn(_call("c1", "search_literature", query="q", k=6)),
                _turn(
                    _call(
                        "c2",
                        "finish_retrieval",
                        selected_ref_ids=["openalex:W2"],
                        reason="够用",
                    )
                ),
            ]
            agent = _agent(settings, retriever, script)
            pool = build_pool.build_pool(topics, retriever, agent=agent)
            pool_path = build_pool.write_pool(pool, tmp_path / "pool.json")
            labels_path = build_pool.write_labels(pool, tmp_path / "labels_todo.csv")
        return pool_path.read_text(encoding="utf-8"), labels_path.read_text(encoding="utf-8")

    pool_json_1, labels_1 = run_once(True)
    pool_json_2, labels_2 = run_once(False)

    # 同一种子两次生成结果相同
    assert pool_json_1 == pool_json_2
    assert labels_1 == labels_2

    payload = json.loads(pool_json_1)
    assert payload["topics"][0]["rankings"]["original"] == [
        "openalex:W1",
        "openalex:W2",
        "openalex:W3",
    ]
    # 每条候选带 methods，且来自多种方法时并集正确
    candidate = payload["topics"][0]["candidates"][0]
    assert candidate["methods"] == ["original", "keywords"]

    # 标注表：表头不含"来自哪种方法"，relevant 留空
    reader = csv.reader(labels_1.splitlines())
    assert next(reader) == LABELS_HEADER
    for method in ("original", "keywords", "agent"):
        assert method not in labels_1
    for row in reader:
        assert row[6] == ""

    # 每个课题内顺序按固定种子打乱：CSV 里该课题的 ref_id 顺序等于 Random(42) 打乱后的候选顺序
    for topic in payload["topics"]:
        candidate_refs = [c["ref_id"] for c in topic["candidates"]]
        expected = list(candidate_refs)
        random.Random(42).shuffle(expected)
        csv_refs = [row[2] for row in csv.reader(labels_1.splitlines()) if row[0] == topic["id"]]
        assert csv_refs == expected


# --- A2：不带 --confirm-real-calls 不发请求 -----------------------------------


def test_build_pool_preview_makes_no_requests(capsys: pytest.CaptureFixture[str]) -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(WORKS_ENDPOINT).mock(side_effect=AssertionError("预演不应发请求"))
        exit_code = build_pool.main([])

    assert exit_code == 0
    captured = capsys.readouterr()
    expected_topics = len(build_pool.load_topics(build_pool.TOPICS_PATH))
    assert f"课题数: {expected_topics}" in captured.out
    assert "预计请求次数:" in captured.out
    assert len(router.calls) == 0


# --- A3：指标与手算一致 -------------------------------------------------------


def _pool() -> dict[str, Any]:
    return {
        "cache_checksum": "",
        "shuffle_seed": 42,
        "topics": [
            {
                "id": "t1",
                "topic": "课题一",
                "rankings": {
                    "original": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                    "keywords": ["e", "f", "a", "b", "c", "d"],
                    "agent": [],
                },
                "candidates": [],
            },
            {
                "id": "t2",
                "topic": "课题二",
                "rankings": {
                    "original": ["g", "h", "i", "j", "k"],
                    "keywords": ["h", "i", "j", "k", "g"],
                    "agent": [],
                },
                "candidates": [],
            },
        ],
    }


def test_score_matches_hand_computed() -> None:
    labels = {
        "t1": {"a": 1, "b": 1, "c": 1, "d": 1},
        "t2": {"g": 1},
    }
    report = score.score(_pool(), labels)

    t1 = report["topics"][0]["methods"]
    # original: top5=[a,b,c,d,e]，4 相关 → R@5=1.0, R@10=1.0, P@5=0.8, MRR=1.0
    assert t1["original"]["recall_at_5"] == pytest.approx(1.0)
    assert t1["original"]["recall_at_10"] == pytest.approx(1.0)
    assert t1["original"]["precision_at_5"] == pytest.approx(0.8)
    assert t1["original"]["mrr"] == pytest.approx(1.0)
    # keywords: top5=[e,f,a,b,c]，3 相关 → R@5=0.75, R@10=1.0, P@5=0.6, MRR=1/3
    assert t1["keywords"]["recall_at_5"] == pytest.approx(0.75)
    assert t1["keywords"]["recall_at_10"] == pytest.approx(1.0)
    assert t1["keywords"]["precision_at_5"] == pytest.approx(0.6)
    assert t1["keywords"]["mrr"] == pytest.approx(1 / 3)

    t2 = report["topics"][1]["methods"]
    assert t2["original"]["recall_at_5"] == pytest.approx(1.0)
    assert t2["original"]["precision_at_5"] == pytest.approx(0.2)
    assert t2["original"]["mrr"] == pytest.approx(1.0)
    assert t2["keywords"]["mrr"] == pytest.approx(0.2)

    # 汇总取平均（round 4）
    summary = report["summary"]
    assert summary["original"]["recall_at_5"] == pytest.approx(1.0)
    assert summary["original"]["precision_at_5"] == pytest.approx(0.5)
    assert summary["keywords"]["recall_at_5"] == pytest.approx(0.875)
    assert summary["keywords"]["precision_at_5"] == pytest.approx(0.4)
    # agent 空排名 → 全 0
    assert summary["agent"]["mrr"] == pytest.approx(0.0)


# --- A4：标注值非法报错并指行号 -----------------------------------------------


@pytest.mark.parametrize(
    ("data_lines", "expected_line"),
    [
        ("t1,课题,a,标题A,2020,摘要A,\n", 2),  # 空值，第 2 行
        ("t1,课题,a,标题A,2020,摘要A,1\nt1,课题,b,标题B,2020,摘要B,\n", 3),  # 空值，第 3 行
        ("t1,课题,a,标题A,2020,摘要A,2\n", 2),  # 非 0/1，第 2 行
        ("t1,课题,a,标题A,2020,摘要A,yes\n", 2),
    ],
)
def test_labels_rejects_invalid_relevant(
    tmp_path: Path, data_lines: str, expected_line: int
) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "topic_id,topic,ref_id,title,year,abstract,relevant\n" + data_lines,
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        score.load_labels(path)
    assert f"第 {expected_line} 行" in str(exc.value)


# --- A5：池外文献记为"未标注"，不进分母也不当不相关 ---------------------------


def test_score_reports_unlabeled(tmp_path: Path) -> None:
    pool = {
        "cache_checksum": "",
        "shuffle_seed": 42,
        "topics": [
            {
                "id": "t1",
                "topic": "课题一",
                "rankings": {"original": ["a", "b", "c"], "keywords": [], "agent": []},
                "candidates": [
                    {
                        "ref_id": "a",
                        "title": "A",
                        "year": 2020,
                        "abstract": "",
                        "methods": ["original"],
                    },
                    {
                        "ref_id": "b",
                        "title": "B",
                        "year": 2020,
                        "abstract": "",
                        "methods": ["original"],
                    },
                    {
                        "ref_id": "c",
                        "title": "C",
                        "year": 2020,
                        "abstract": "",
                        "methods": ["original"],
                    },
                ],
            }
        ],
    }
    labels = {"t1": {"a": 1, "b": 0, "c": 0}}
    new_run = {"t1": ["a", "z", "b"]}  # z 不在池里

    report = score.score(pool, labels, new_run=new_run)

    assert report["unlabeled_total"] == 1
    new_run_topic = report["topics"][0]["new_run"]
    assert new_run_topic["unlabeled"] == ["z"]
    assert new_run_topic["unlabeled_count"] == 1
    # z 被排除，judged 排位只剩 [a, b]；相关只有 a
    assert new_run_topic["metrics"]["recall_at_5"] == pytest.approx(1.0)
    assert new_run_topic["metrics"]["precision_at_5"] == pytest.approx(0.2)
    assert new_run_topic["metrics"]["mrr"] == pytest.approx(1.0)


def test_baselines_use_english_topic_when_given():
    """original / keywords 用 topic_en 检索，agent 拿到的仍是课题原文（T-024 审查）。"""
    from evals.retrieval.build_pool import build_pool

    seen: list[str] = []

    class _Recorder:
        name = "rec"

        def search(self, query, *, k):
            seen.append(query)
            return []

    topics = [
        {"id": "t01", "topic": "缺失值处理方法综述", "topic_en": "missing data imputation review"}
    ]
    build_pool(topics, _Recorder())
    assert seen[0] == "missing data imputation review"
    assert all("缺失" not in q for q in seen)


# --- T-027：Agent 失败的课题写入 pool.json 并被指标排除 --------------------------


class _FailingAgent:
    """每次 decide 都返回 agent_error 的检索 Agent 替身。"""

    def decide(self, topic: str):
        from researchflow.core.schemas import LLMUsage, RetrievalDecision

        return RetrievalDecision(
            refs=[], queries=[], stop_reason="agent_error", usage=LLMUsage(attempts=0)
        )


def test_write_pool_includes_agent_stop_reason(tmp_path: Path) -> None:
    """build_pool 已记录 agent_stop_reason，现在要写进 pool.json（T-027）。"""
    topics = [{"id": "t01", "topic": "数据质量对实证研究结论的影响"}]

    class _Recorder:
        name = "rec"

        def search(self, query, *, k):
            return []

    pool = build_pool.build_pool(topics, _Recorder(), agent=_FailingAgent())
    pool_path = build_pool.write_pool(pool, tmp_path / "pool.json")
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    assert payload["topics"][0]["agent_stop_reason"] == "agent_error"


def test_build_pool_records_agent_queries(tmp_path: Path) -> None:
    """Agent 的改写查询写进 pool.json（T-031）；失败降级时为空列表。"""
    topics = [{"id": "t01", "topic": "数据质量对实证研究结论的影响"}]
    settings = _settings(tmp_path)
    retriever = OpenAlexRetriever(settings)
    works = [_work("W1", "论文一"), _work("W2", "论文二")]

    class _Recorder:
        name = "rec"

        def search(self, query, *, k):
            return []

    # 正常路径：Agent 改写查询 q1 被 search_literature 记录，随 finish 一起返回。
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKS_ENDPOINT).mock(return_value=_works_response(works))
        script = [
            _turn(_call("c1", "search_literature", query="q1", k=6)),
            _turn(_call("c2", "finish_retrieval", selected_ref_ids=["openalex:W1"], reason="够用")),
        ]
        agent = _agent(settings, retriever, script)
        pool = build_pool.build_pool(topics, _Recorder(), agent=agent)
    pool_path = build_pool.write_pool(pool, tmp_path / "pool.json")
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    assert payload["topics"][0]["agent_queries"] == ["q1"]

    # 失败降级路径：agent_error 时 queries 为空，pool.json 里也是空列表。
    pool = build_pool.build_pool(topics, _Recorder(), agent=_FailingAgent())
    pool_path = build_pool.write_pool(pool, tmp_path / "pool_failed.json")
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    assert payload["topics"][0]["agent_queries"] == []


def test_score_excludes_agent_failed_topics() -> None:
    """agent_error / invalid_selection 的课题排除出所有方法的汇总指标，并列出编号。"""
    pool = _pool()
    pool["topics"][0]["agent_stop_reason"] = "agent_error"
    pool["topics"][1]["agent_stop_reason"] = "invalid_selection"
    labels = {
        "t1": {"a": 1, "b": 1, "c": 1, "d": 1},
        "t2": {"g": 1},
    }

    report = score.score(pool, labels)

    assert report["excluded_topics"] == ["t1", "t2"]
    assert [t["excluded"] for t in report["topics"]] == [True, True]
    # 全部课题都被排除：汇总指标为 0（_avg 空列表），不崩
    assert report["summary"]["original"]["mrr"] == 0.0


def test_score_excludes_only_failed_topic_keeps_rest_in_denominator() -> None:
    """只排除失败课题；其余课题的分母不变（对所有方法一致）。"""
    pool = _pool()
    pool["topics"][0]["agent_stop_reason"] = "agent_error"
    labels = {
        "t1": {"a": 1, "b": 1, "c": 1, "d": 1},
        "t2": {"g": 1},
    }

    report = score.score(pool, labels)

    assert report["excluded_topics"] == ["t1"]
    # 汇总只来自 t2：original R@5=1.0 / P@5=0.2 / MRR=1.0（与手算一致）
    assert report["summary"]["original"]["recall_at_5"] == pytest.approx(1.0)
    assert report["summary"]["original"]["precision_at_5"] == pytest.approx(0.2)
    assert report["summary"]["original"]["mrr"] == pytest.approx(1.0)


def test_score_without_agent_stop_reason_unchanged() -> None:
    """旧 pool.json 没有 agent_stop_reason 字段：不排除任何课题，行为不变。"""
    labels = {
        "t1": {"a": 1, "b": 1, "c": 1, "d": 1},
        "t2": {"g": 1},
    }
    report = score.score(_pool(), labels)
    assert report["excluded_topics"] == []
    assert all(not t["excluded"] for t in report["topics"])
    assert report["summary"]["original"]["precision_at_5"] == pytest.approx(0.5)


def test_render_report_lists_excluded_topics() -> None:
    pool = _pool()
    pool["topics"][0]["agent_stop_reason"] = "agent_error"
    labels = {"t1": {"a": 1}, "t2": {"g": 1}}
    report = score.score(pool, labels)
    text = score.render_report(report)
    assert "排除了 1 个课题：t1" in text
    assert "已排除：Agent 失败" in text

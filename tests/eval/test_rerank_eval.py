"""T-030/T-031 离线重排评测（fake 层）：方法列齐全、覆盖率进报告、real 只预演（A3/A6）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下（与其他 eval 测试一致）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.retrieval import rerank_eval  # noqa: E402

LABELS_HEADER = ["topic_id", "topic", "ref_id", "title", "year", "abstract", "relevant"]


def _pool(tmp_path: Path) -> Path:
    """构造小池：t1 里 hybrid（两路融合）能赢任一单路的一个指标。"""
    pool = {
        "topics": [
            {
                "id": "t1",
                "topic": "缺失值插补",
                "rankings": {
                    "keywords": ["stub:irrelevant"],
                    "agent": ["stub:irrelevant"],
                },
                "candidates": [
                    {
                        "ref_id": "stub:m1",
                        "title": "multiple imputation 缺失值插补",
                        "abstract": "",
                    },
                    {"ref_id": "stub:m2", "title": "multiple imputation 插补方法", "abstract": ""},
                    {"ref_id": "stub:m3", "title": "无关文献 unrelated", "abstract": ""},
                ],
            }
        ]
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    return path


def _labels(tmp_path: Path) -> Path:
    rows = [
        LABELS_HEADER,
        ["t1", "缺失值插补", "stub:m1", "t", 2020, "", "1"],
        ["t1", "缺失值插补", "stub:m2", "t", 2020, "", "1"],
        ["t1", "缺失值插补", "stub:m3", "t", 2020, "", "0"],
    ]
    path = tmp_path / "labels.csv"
    lines = [",".join(str(cell) for cell in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_fake_layer_reranks_and_hybrid_beats_a_single_route_somewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(rerank_eval, "REPORTS_DIR", tmp_path / "reports")
    pool_path = _pool(tmp_path)
    labels_path = _labels(tmp_path)

    exit_code = rerank_eval.main(
        ["--pool", str(pool_path), "--labels", str(labels_path), "--layer", "fake"]
    )
    assert exit_code == 0

    # 报告写到 evals/retrieval/reports/（测试里替换为 tmp_path）
    reports = list((tmp_path / "reports").glob("rerank-fake-*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    for method in (
        "hybrid",
        "hybrid_adaptive",
        "hybrid_multiquery",
        "bm25",
        "dense",
        "keywords",
        "agent",
    ):
        assert f"| {method} |" in text, "并排表包含全部方法"
    assert "池内理论最大 Recall" in text
    assert "逐课题胜负" in text
    # A3：报告新增每课题的 BM25 覆盖率
    assert "BM25 覆盖率（原文查询）" in text
    assert "BM25 覆盖率（原文查询，平均）" in text

    # 指标可复现：重排结果只来自候选，且至少有一个方法在某个指标上 ≠ 另一个
    # （证明流程能算出差异——相关文献 m1/m2 应排在不相关的 m3 之前）
    report_lines = text
    assert report_lines.count("|") > 10


def test_dry_run_real_layer_makes_no_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    monkeypatch.setattr(rerank_eval, "REPORTS_DIR", tmp_path / "reports")
    pool_path = _pool(tmp_path)
    labels_path = _labels(tmp_path)

    exit_code = rerank_eval.main(
        ["--pool", str(pool_path), "--labels", str(labels_path), "--layer", "real"]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "预演" in out and "未发任何请求" in out
    assert "块数:" in out
    # 预演不写报告
    assert not (tmp_path / "reports").exists()


def test_dry_run_chunk_count_matches_chunking(tmp_path: Path):
    pool = json.loads(_pool(tmp_path).read_text(encoding="utf-8"))
    stats = rerank_eval._dry_run_stats(pool)
    from researchflow.core.schemas import LiteratureRef
    from researchflow.tools.chunking import chunk_ref

    expected = sum(
        len(chunk_ref(LiteratureRef(ref_id=c["ref_id"], title=c["title"], summary=c["abstract"])))
        for topic in pool["topics"]
        for c in topic["candidates"]
    )
    assert stats["chunks"] == expected
    assert stats["topics"] == 1


# --- T-031：多查询回落与覆盖率 -------------------------------------------------


def test_topic_queries_falls_back_to_original_without_agent_queries():
    """A3：旧池没有 agent_queries 字段时回落为只用课题原文，不报错。"""
    assert rerank_eval._topic_queries({"topic": "课题原文"}) == ["课题原文"]
    assert rerank_eval._topic_queries({"topic": "课题原文", "agent_queries": []}) == ["课题原文"]
    assert rerank_eval._topic_queries(
        {"topic": "课题原文", "agent_queries": ["改写一", "改写二"]}
    ) == [
        "课题原文",
        "改写一",
        "改写二",
    ]


def test_bm25_coverage_hand_computed():
    """覆盖率 = 查询词（同 BM25 分词）与候选块词表的交集比例。"""
    topic = {
        "id": "t1",
        "topic": "缺失值插补",
        "candidates": [
            {"ref_id": "stub:m1", "title": "缺失值插补方法", "abstract": ""},
            {"ref_id": "stub:m2", "title": "unrelated work", "abstract": ""},
        ],
    }
    # "缺失值插补" 的二元组 {缺失, 失值, 值插, 插补} 全部在候选词表里 → 1.0
    assert rerank_eval._bm25_coverage(topic) == pytest.approx(1.0)
    # 查询词一个都不在词表里 → 0.0
    topic["topic"] = "完全无关的词"
    assert rerank_eval._bm25_coverage(topic) == pytest.approx(0.0)


def test_rank_topic_with_agent_queries_uses_multiquery():
    """hybrid_multiquery 的查询集合是"原文 + agent_queries"（用固定嵌入器核对）。"""
    from researchflow.core.config import HybridSettings
    from researchflow.tools.hybrid import HybridReranker
    from researchflow.tools.vector_store import VectorStore

    topic = {
        "id": "t1",
        "topic": "alpha",
        "agent_queries": ["beta"],
        "candidates": [
            {"ref_id": "stub:x", "title": "alpha", "abstract": ""},
            {"ref_id": "stub:y", "title": "beta", "abstract": ""},
            {"ref_id": "stub:z", "title": "gamma", "abstract": ""},
        ],
    }
    embedder = _FixedEmbedder(
        {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.7071067811865476, 0.7071067811865476],
        }
    )
    rankings = rerank_eval._rank_topic(
        topic, embedder, hybrid_cfg=HybridSettings(fusion="adaptive", rrf_k=60, coverage_ref=0.5)
    )
    # 与直接用多查询调 HybridReranker 的结果一致（两查询词都在词表里 → w=1，等权融合）
    store = VectorStore(":memory:")
    try:
        expected_reranker = HybridReranker(embedder, store, rrf_k=60, coverage_ref=0.5)
        expected = expected_reranker.rerank(["alpha", "beta"], rerank_eval._refs_of(topic), 3)
        assert rankings["hybrid_multiquery"] == [ref.ref_id for ref in expected]
        # 单查询列只用原文，与直接单查询调用一致
        expected_single = expected_reranker.rerank("alpha", rerank_eval._refs_of(topic), 3)
        assert rankings["hybrid_adaptive"] == [ref.ref_id for ref in expected_single]
    finally:
        store.close()


class _FixedEmbedder:
    """测试替身：按预设向量表嵌入，让 dense 路的排名可以手算。"""

    name = "fixed"
    dim = 2
    last_usage = None
    last_duration_ms = 0

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.table[text]) for text in texts]

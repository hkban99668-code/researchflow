"""T-030/T-031 混合重排：RRF 手算例子一致、同分稳定、chunk_id 填入、结果只来自候选；
adaptive 覆盖率加权与多查询融合（A1/A2）。"""

from __future__ import annotations

import pytest

from researchflow.core.schemas import LiteratureRef
from researchflow.tools.hybrid import HybridReranker
from researchflow.tools.vector_store import VectorStore


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


def _refs() -> list[LiteratureRef]:
    return [
        LiteratureRef(ref_id="stub:x", title="alpha"),
        LiteratureRef(ref_id="stub:y", title="beta"),
        LiteratureRef(ref_id="stub:z", title="gamma"),
    ]


def _embedder() -> _FixedEmbedder:
    # query "alpha beta" -> [1,0]；alpha 同向（dense 第 1），beta/gamma 正交且同分
    return _FixedEmbedder(
        {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.0, 1.0],
            "alpha beta": [1.0, 0.0],
        }
    )


def test_rrf_hand_computed_fusion_order():
    # BM25：alpha 命中 rank1；beta/gamma 0 分同分按 chunk_id → y#c0 rank2, z#c0 rank3
    # dense：alpha=1.0 rank1；beta/gamma=0 同分按 chunk_id → rank2/rank3
    # RRF(k=60)：alpha = 1/61+1/61；beta = 1/62+1/62；gamma = 1/63+1/63
    store = VectorStore(":memory:")
    reranker = HybridReranker(_embedder(), store, rrf_k=60)
    ranked = reranker.rerank("alpha beta", _refs(), 3)
    assert [ref.ref_id for ref in ranked] == ["stub:x", "stub:y", "stub:z"]
    assert ranked[0].score == pytest.approx(2 / 61)
    assert ranked[1].score == pytest.approx(2 / 62)
    assert ranked[2].score == pytest.approx(2 / 63)
    assert ranked[0].chunk_id == "stub:x#c0", "chunk_id 填入命中块"
    store.close()


def test_results_only_from_given_candidates():
    store = VectorStore(":memory:")
    reranker = HybridReranker(_embedder(), store)
    refs = _refs()
    ranked = reranker.rerank("alpha", refs, 3)
    assert {ref.ref_id for ref in ranked} <= {ref.ref_id for ref in refs}
    store.close()


def test_k_limits_output_and_empty_candidates():
    store = VectorStore(":memory:")
    reranker = HybridReranker(_embedder(), store)
    assert len(reranker.rerank("alpha", _refs(), 2)) == 2
    assert reranker.rerank("alpha", [], 3) == []
    store.close()


def test_tie_broken_by_ref_id_stably():
    # 两篇文献内容完全相同 → 各路分数相同 → 按 ref_id 稳定排序。
    # 查询词须落在候选词表里（coverage=1 → w_bm25=1 等权融合）；
    # 覆盖率为 0 时 adaptive 等价纯向量，同分的两路 rank 不同、分数不再相等。
    embedder = _FixedEmbedder({"same text": [1.0, 0.0], "query": [1.0, 0.0], "same": [1.0, 0.0]})
    store = VectorStore(":memory:")
    reranker = HybridReranker(embedder, store)
    refs = [
        LiteratureRef(ref_id="stub:bb", title="same text"),
        LiteratureRef(ref_id="stub:aa", title="same text"),
    ]
    ranked = reranker.rerank("same", refs, 2)
    assert [ref.ref_id for ref in ranked] == ["stub:aa", "stub:bb"]
    assert ranked[0].score == ranked[1].score
    store.close()


def test_bm25_only_and_dense_only_routes():
    store = VectorStore(":memory:")
    reranker = HybridReranker(_embedder(), store)
    refs = _refs()
    bm25 = reranker.rank_bm25_only("alpha", refs, 3)
    dense = reranker.rank_dense_only("alpha", refs, 3)
    assert [ref.ref_id for ref in bm25] == ["stub:x", "stub:y", "stub:z"]
    assert [ref.ref_id for ref in dense] == ["stub:x", "stub:y", "stub:z"]
    # 单路分数是 1/(60+rank)：alpha=1/61，beta=1/62，gamma=1/63
    assert bm25[0].score == pytest.approx(1 / 61)
    assert dense[0].score == pytest.approx(1 / 61)
    store.close()


# --- T-031 adaptive 融合 -------------------------------------------------------


def _zero_coverage_embedder() -> _FixedEmbedder:
    """查询 "zzz qqq" 与全部候选无词面重叠；dense 排名 beta > gamma > alpha。"""
    return _FixedEmbedder(
        {
            "alpha": [0.0, 1.0],
            "beta": [1.0, 0.0],
            "gamma": [0.7071067811865476, 0.7071067811865476],
            "zzz qqq": [1.0, 0.0],
        }
    )


def test_adaptive_zero_coverage_equals_dense():
    """A1：覆盖率为 0 时 adaptive 排名与纯向量完全一致（w_bm25=0）。"""
    store = VectorStore(":memory:")
    reranker = HybridReranker(_zero_coverage_embedder(), store, coverage_ref=0.5)
    refs = _refs()
    adaptive = reranker.rerank("zzz qqq", refs, 3)
    dense = reranker.rank_dense_only("zzz qqq", refs, 3)
    assert [ref.ref_id for ref in adaptive] == [ref.ref_id for ref in dense]
    for a, d in zip(adaptive, dense, strict=True):
        assert a.score == pytest.approx(d.score)
        assert a.chunk_id == d.chunk_id
    store.close()


def test_adaptive_full_coverage_equals_rrf():
    """A1：覆盖率 ≥ coverage_ref 时 adaptive 与等权 RRF 完全一致（w_bm25=1）。"""
    store = VectorStore(":memory:")
    refs = _refs()
    # 查询 "alpha beta" 的 2 个词都在候选词表里 → coverage=1 ≥ 0.5。
    adaptive = HybridReranker(_embedder(), store, coverage_ref=0.5)
    rrf = HybridReranker(_embedder(), store, coverage_ref=0.5, fusion="rrf")
    a = adaptive.rerank("alpha beta", refs, 3)
    r = rrf.rerank("alpha beta", refs, 3)
    assert [ref.ref_id for ref in a] == [ref.ref_id for ref in r]
    for item_a, item_r in zip(a, r, strict=True):
        assert item_a.score == pytest.approx(item_r.score)
    store.close()


def test_adaptive_partial_coverage_hand_computed():
    """A1：中间值手算一致——coverage=0.25、coverage_ref=0.5 → w_bm25=0.5。

    候选 x="alpha"、y="beta"、z="gamma"；查询 "alpha zzz yyy www" 分词后
    {alpha, zzz, yyy, www}，只有 alpha 在候选词表里 → coverage=0.25。
    BM25：x 命中 rank1，y/z 0 分按 chunk_id rank2/3；dense（查询 [0,1]）：
    y=[0,1] rank1、z=[0.9,0.9] rank2、x=[1,0] rank3。
    分数：x = 0.5/61 + 1/63；y = 0.5/62 + 1/61；z = 0.5/63 + 1/62。
    """
    embedder = _FixedEmbedder(
        {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.9, 0.9],
            "alpha zzz yyy www": [0.0, 1.0],
        }
    )
    store = VectorStore(":memory:")
    reranker = HybridReranker(embedder, store, rrf_k=60, coverage_ref=0.5)
    ranked = reranker.rerank("alpha zzz yyy www", _refs(), 3)
    assert [ref.ref_id for ref in ranked] == ["stub:y", "stub:x", "stub:z"]
    assert ranked[0].score == pytest.approx(0.5 / 62 + 1 / 61)
    assert ranked[1].score == pytest.approx(0.5 / 61 + 1 / 63)
    assert ranked[2].score == pytest.approx(0.5 / 63 + 1 / 62)
    store.close()


def test_multiquery_best_rank_per_route_hand_computed():
    """A2：多查询每路取每篇最好排名再融合，手算一致。

    查询 ["alpha", "beta"]；候选 x="alpha"、y="beta"、z="gamma"。
    BM25：q1 x r1/y r2/z r3；q2 y r1/x r2/z r3 → 最好权重 x=y=1/61，z=1/63。
    dense（x=[1,0]、y=[0,1]、z=[√2/2,√2/2]，查询各自同向）：
    q1 x r1/z r2/y r3；q2 y r1/z r2/x r3 → 最好权重 x=y=1/61，z=1/62。
    coverage：两查询的词都在词表里 → w_bm25=1。
    融合分：x=y=2/61（同分按 ref_id），z=1/63+1/62。
    """
    embedder = _FixedEmbedder(
        {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.7071067811865476, 0.7071067811865476],
        }
    )
    store = VectorStore(":memory:")
    reranker = HybridReranker(embedder, store, rrf_k=60, coverage_ref=0.5)
    ranked = reranker.rerank(["alpha", "beta"], _refs(), 3)
    assert [ref.ref_id for ref in ranked] == ["stub:x", "stub:y", "stub:z"]
    assert ranked[0].score == pytest.approx(2 / 61)
    assert ranked[1].score == pytest.approx(2 / 61)
    assert ranked[2].score == pytest.approx(1 / 63 + 1 / 62)
    store.close()


def test_single_string_and_list_equivalent():
    """A2：单字符串输入与等价单元素列表行为完全一致（行为不变）。"""
    store = VectorStore(":memory:")
    reranker = HybridReranker(_embedder(), store)
    refs = _refs()
    single = reranker.rerank("alpha beta", refs, 3)
    as_list = reranker.rerank(["alpha beta"], refs, 3)
    assert [ref.ref_id for ref in single] == [ref.ref_id for ref in as_list]
    for a, b in zip(single, as_list, strict=True):
        assert a.score == pytest.approx(b.score)
    store.close()


def test_invalid_fusion_and_coverage_ref_rejected():
    """参数校验：fusion 只能是 adaptive/rrf，coverage_ref 必须大于 0。"""
    store = VectorStore(":memory:")
    with pytest.raises(ValueError, match="fusion"):
        HybridReranker(_embedder(), store, fusion="bogus")
    with pytest.raises(ValueError, match="coverage_ref"):
        HybridReranker(_embedder(), store, coverage_ref=0.0)
    store.close()


# --- T-031 配置接线：retrieval.hybrid 进 settings.yaml --------------------------


def _write_project(tmp_path, hybrid_yaml: str) -> None:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "settings.yaml").write_text(
        f"retrieval:\n  hybrid:\n{hybrid_yaml}", encoding="utf-8"
    )


def test_hybrid_settings_loaded_from_yaml(tmp_path):
    """rrf_k / coverage_ref / fusion 从 settings.yaml 读入，默认值与文档一致。"""
    from researchflow.core.config import load_settings

    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "settings.yaml").write_text("llm:\n  mode: auto\n", encoding="utf-8")
    settings = load_settings(project_root=tmp_path, load_env_file=False)
    assert settings.retrieval.hybrid.fusion == "adaptive"
    assert settings.retrieval.hybrid.rrf_k == 60
    assert settings.retrieval.hybrid.coverage_ref == 0.5

    _write_project(tmp_path, "    fusion: rrf\n    rrf_k: 30\n    coverage_ref: 0.4\n")
    settings = load_settings(project_root=tmp_path, load_env_file=False)
    assert settings.retrieval.hybrid.fusion == "rrf"
    assert settings.retrieval.hybrid.rrf_k == 30
    assert settings.retrieval.hybrid.coverage_ref == 0.4


def test_hybrid_settings_validation(tmp_path):
    """非法 fusion 与 coverage_ref ≤ 0 报 ConfigError。"""
    from researchflow.core.config import ConfigError, load_settings

    _write_project(tmp_path, "    fusion: bogus\n")
    with pytest.raises(ConfigError, match="fusion"):
        load_settings(project_root=tmp_path, load_env_file=False)

    _write_project(tmp_path, "    coverage_ref: 0\n")
    with pytest.raises(ConfigError, match="coverage_ref"):
        load_settings(project_root=tmp_path, load_env_file=False)

"""混合重排器：BM25（词法）+ 向量（语义）两路，RRF 或 adaptive 融合（T-030/T-031）。

流程：对一批候选文献切块 → 两路各自对**块**排名 → 融合到块 →
按文献聚合（取其最高分块）→ 返回前 k 篇。检索结果仍以文献为单位，
`LiteratureRef.chunk_id` 填入命中的最高分块，`score` 填融合分。

RRF 公式：`score = Σ 1/(rrf_k + rank_i)`（rank 从 1 起），常数 60 可配置，不调参。
同分按 `ref_id` 稳定排序（确定性）。结果只来自传入的候选文献。

融合模式（T-031）：

- `adaptive`（默认）：每个课题先算 BM25 一路的"有效性"——查询词（同 BM25 分词）
  在候选块词表中的覆盖率 `coverage ∈ [0,1]`，融合分为
  `w_bm25 * 1/(rrf_k+rank_bm25) + 1/(rrf_k+rank_dense)`，其中
  `w_bm25 = clamp(coverage / coverage_ref, 0, 1)`。覆盖率为 0 时等价于纯向量
  （BM25 一路失效不再拖累），`coverage ≥ coverage_ref` 时等价于等权 RRF。
- `rrf`：等权 RRF（T-030 原行为），保留供对比。

多查询（T-031）：`rerank` 接受 `list[str] | str`；BM25 与向量各自对多条查询
取每篇文献的最好排名（= 最大 RRF 权重）后再融合；覆盖率取各查询覆盖率的
最大值。单字符串输入行为不变。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import HYBRID_FUSION_MODES
from ..core.embedding import Embedder
from ..core.schemas import LiteratureRef
from .bm25 import BM25Index, tokenize
from .chunking import Chunk, chunk_ref
from .vector_store import VectorStore

DEFAULT_RRF_K = 60
DEFAULT_COVERAGE_REF = 0.5


@dataclass
class _RankedRef:
    """一个候选文献的融合结果：融合分与命中块。"""

    ref_id: str
    score: float
    chunk: Chunk


def _as_query_list(queries: list[str] | str) -> list[str]:
    """查询参数归一化：单字符串当成一条查询，列表原样。"""
    if isinstance(queries, str):
        return [queries]
    return list(queries)


def _query_coverage(query: str, vocab: set[str]) -> float:
    """查询词（同 BM25 分词）在候选块词表中的覆盖率 ∈ [0,1]；空查询为 0。"""
    tokens = set(tokenize(query))
    if not tokens:
        return 0.0
    return len(tokens & vocab) / len(tokens)


def _best_weights(weight_lists: list[dict[str, float]]) -> dict[str, float]:
    """多路查询权重合并：每个块取各查询里的最大权重（即最好排名）。"""
    best: dict[str, float] = {}
    for weights in weight_lists:
        for chunk_id, weight in weights.items():
            best[chunk_id] = max(best.get(chunk_id, 0.0), weight)
    return best


class HybridReranker:
    """BM25 + 向量的混合重排器，融合模式 adaptive（默认）或 rrf。"""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        coverage_ref: float = DEFAULT_COVERAGE_REF,
        fusion: str = "adaptive",
    ) -> None:
        if fusion not in HYBRID_FUSION_MODES:
            raise ValueError(
                f"fusion 取值非法: {fusion!r}（只能是 {' / '.join(HYBRID_FUSION_MODES)}）"
            )
        if coverage_ref <= 0:
            raise ValueError(f"coverage_ref 必须大于 0，收到 {coverage_ref!r}")
        self.embedder = embedder
        self.store = store
        self.rrf_k = rrf_k
        self.coverage_ref = coverage_ref
        self.fusion = fusion

    def rerank(
        self, queries: list[str] | str, refs: list[LiteratureRef], k: int
    ) -> list[LiteratureRef]:
        """混合重排：返回前 k 篇（`chunk_id` / `score` 已填），只来自传入候选。"""
        return self._rank(_as_query_list(queries), refs, k, mode="hybrid")

    def rank_bm25_only(
        self, queries: list[str] | str, refs: list[LiteratureRef], k: int
    ) -> list[LiteratureRef]:
        """仅 BM25 消融路：供离线评测对照。"""
        return self._rank(_as_query_list(queries), refs, k, mode="bm25")

    def rank_dense_only(
        self, queries: list[str] | str, refs: list[LiteratureRef], k: int
    ) -> list[LiteratureRef]:
        """仅向量消融路：供离线评测对照。"""
        return self._rank(_as_query_list(queries), refs, k, mode="dense")

    def _rank(
        self, queries: list[str], refs: list[LiteratureRef], k: int, *, mode: str
    ) -> list[LiteratureRef]:
        if not refs or k <= 0 or not queries:
            return []
        chunks = [chunk for ref in refs for chunk in chunk_ref(ref)]
        by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}

        # 两路块级权重（rank 从 1 起；多条查询时每路先取各块的最好排名）。
        bm25_best: dict[str, float] | None = None
        dense_best: dict[str, float] | None = None
        if mode in ("hybrid", "bm25"):
            index = BM25Index.build([chunk.text for chunk in chunks])
            bm25_best = _best_weights(
                [
                    _rrf_weights(
                        [
                            (chunks[doc_index].chunk_id, score)
                            for doc_index, score in index.rank(query, len(chunks))
                        ],
                        self.rrf_k,
                    )
                    for query in queries
                ]
            )
        if mode in ("hybrid", "dense"):
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            self.store.add(chunks, vectors, self.embedder.name)
            dense_best = _best_weights(
                [
                    _rrf_weights(
                        self.store.search(
                            self.embedder.embed([query])[0],
                            len(chunks),
                            embedder_name=self.embedder.name,
                        ),
                        self.rrf_k,
                    )
                    for query in queries
                ]
            )

        # 融合到块：adaptive 按覆盖率给 BM25 一路加权，rrf 等权相加；
        # 单路消融时即该路的 1/(rrf_k + rank)。
        fused: dict[str, float] = {}
        if mode == "hybrid" and self.fusion == "adaptive":
            vocab: set[str] = set()
            for chunk in chunks:
                vocab.update(tokenize(chunk.text))
            coverage = max(_query_coverage(query, vocab) for query in queries)
            w_bm25 = min(coverage / self.coverage_ref, 1.0)
            assert bm25_best is not None and dense_best is not None  # mode=hybrid 必有两路
            fused = {
                chunk_id: w_bm25 * bm25_best.get(chunk_id, 0.0) + dense_best.get(chunk_id, 0.0)
                for chunk_id in by_chunk_id
            }
        else:
            for weights in (bm25_best, dense_best):
                for chunk_id, weight in (weights or {}).items():
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + weight

        # 按文献聚合：取该文献得分最高的块。
        best: dict[str, _RankedRef] = {}
        for chunk_id, score in fused.items():
            chunk = by_chunk_id[chunk_id]
            current = best.get(chunk.ref_id)
            # 同块分时按 chunk_id 稳定（fused 的迭代顺序由字典插入序决定，
            # 显式比较保证确定）。
            if (
                current is None
                or score > current.score
                or (score == current.score and chunk.chunk_id < current.chunk.chunk_id)
            ):
                best[chunk.ref_id] = _RankedRef(ref_id=chunk.ref_id, score=score, chunk=chunk)

        ranked = sorted(best.values(), key=lambda item: (-item.score, item.ref_id))[:k]
        by_ref_id = {ref.ref_id: ref for ref in refs}
        return [
            LiteratureRef(
                **{
                    **by_ref_id[item.ref_id].model_dump(),
                    "chunk_id": item.chunk.chunk_id,
                    "score": item.score,
                }
            )
            for item in ranked
        ]


def _rrf_weights(ranked: list, rrf_k: int) -> dict[str, float]:
    """把一路排名（[(id, score)]，rank 从 1 起）转成 {id: 1/(rrf_k + rank)}。"""
    return {item[0]: 1.0 / (rrf_k + rank) for rank, item in enumerate(ranked, start=1)}

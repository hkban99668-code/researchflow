"""T-030 向量库：幂等写入、余弦手算一致、嵌入器名隔离（A3）。"""

from __future__ import annotations

import pytest

from researchflow.tools.chunking import Chunk
from researchflow.tools.vector_store import VectorStore


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    ref_id = chunk_id.split("#")[0]
    return Chunk(chunk_id=chunk_id, ref_id=ref_id, text=text)


def test_add_idempotent_overwrites_same_chunk_id():
    store = VectorStore(":memory:")
    chunks = [_chunk("stub:r1#c0")]
    store.add(chunks, [[1.0, 0.0]], "emb-a")
    store.add(chunks, [[0.0, 1.0]], "emb-a")  # 同 chunk_id 覆盖
    assert store.count("emb-a") == 1
    hits = store.search([0.0, 1.0], 1, embedder_name="emb-a")
    assert hits == [("stub:r1#c0", pytest.approx(1.0))], "覆盖后按新向量检索"
    store.close()


def test_cosine_matches_hand_computed_example():
    store = VectorStore(":memory:")
    chunks = [_chunk("stub:r1#c0"), _chunk("stub:r2#c0"), _chunk("stub:r3#c0")]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "emb-a")
    hits = store.search([2.0, 0.0], 3, embedder_name="emb-a")
    # 手算：cos([2,0],[1,0])=1；cos([2,0],[0,1])=0；cos([2,0],[1,1])=2/(2*√2)=1/√2
    assert hits[0] == ("stub:r1#c0", pytest.approx(1.0))
    assert hits[1] == ("stub:r3#c0", pytest.approx(1 / 2**0.5))
    assert hits[2] == ("stub:r2#c0", pytest.approx(0.0))
    store.close()


def test_search_filters_by_embedder_name():
    store = VectorStore(":memory:")
    chunks = [_chunk("stub:r1#c0"), _chunk("stub:r2#c0")]
    store.add(chunks[:1], [[1.0, 0.0]], "emb-a")
    store.add(chunks[1:], [[1.0, 0.0]], "emb-b")
    # 换嵌入器名：旧向量不参与
    assert store.search([1.0, 0.0], 5, embedder_name="emb-b") == [
        ("stub:r2#c0", pytest.approx(1.0))
    ]
    assert store.count("emb-a") == 1
    assert store.count() == 2
    # 未知的嵌入器名：空结果，而不是报错或混用
    assert store.search([1.0, 0.0], 5, embedder_name="emb-c") == []
    store.close()


def test_add_length_mismatch_raises():
    store = VectorStore(":memory:")
    with pytest.raises(ValueError):
        store.add([_chunk("stub:r1#c0")], [], "emb-a")
    store.close()


def test_tie_broken_by_chunk_id():
    store = VectorStore(":memory:")
    chunks = [_chunk("stub:rb#c0"), _chunk("stub:ra#c0")]
    store.add(chunks, [[1.0, 0.0], [1.0, 0.0]], "emb-a")
    hits = store.search([1.0, 0.0], 2, embedder_name="emb-a")
    assert [chunk_id for chunk_id, _ in hits] == ["stub:ra#c0", "stub:rb#c0"], (
        "同分按 chunk_id 稳定排序"
    )
    store.close()

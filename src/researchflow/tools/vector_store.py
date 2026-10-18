"""向量库：SQLite + numpy 暴力余弦，零额外依赖（T-030）。

设计判断（任务卡）：候选规模是几百到几千块，暴力检索足够；规模到十万级以上、
需要过滤或并发写时换 Qdrant/pgvector，换的位置只有 `VectorStore` 这一个接口。

- `add(chunks, vectors)` 幂等：同 `chunk_id` 覆盖（INSERT OR REPLACE）；
- `search(query_vec, k)` 余弦相似度排序，同分按 `chunk_id` 稳定排序；
- 向量按 `embedder_name` 隔离：换嵌入器后旧向量不参与检索（查询只看同名）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    ref_id        TEXT NOT NULL,
    text          TEXT NOT NULL,
    vector        BLOB NOT NULL,
    embedder_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_embedder ON chunks(embedder_name);
"""


class VectorStore:
    """SQLite 存储的暴力余弦向量库。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def add(self, chunks: list[Chunk], vectors: list[list[float]], embedder_name: str) -> None:
        """写入或覆盖（同 chunk_id 幂等）；向量按 float32 小端存 BLOB。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 数量不一致: {len(chunks)} != {len(vectors)}")
        rows = [
            (
                chunk.chunk_id,
                chunk.ref_id,
                chunk.text,
                np.asarray(vector, dtype="<f4").tobytes(),
                embedder_name,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks"
                " (chunk_id, ref_id, text, vector, embedder_name)"
                " VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def count(self, embedder_name: str | None = None) -> int:
        """块数量；传嵌入器名时只数该名的向量。"""
        if embedder_name is None:
            row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedder_name = ?", (embedder_name,)
            ).fetchone()
        return int(row[0])

    def search(
        self, query_vec: list[float], k: int, *, embedder_name: str
    ) -> list[tuple[str, float]]:
        """余弦相似度检索，只看同名嵌入器的向量；返回 [(chunk_id, score)]，分数降序。

        同分按 chunk_id 稳定排序，保证确定性（A3/A5）。
        """
        query = np.asarray(query_vec, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        rows = self._conn.execute(
            "SELECT chunk_id, vector FROM chunks WHERE embedder_name = ?", (embedder_name,)
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for chunk_id, blob in rows:
            vec = np.frombuffer(blob, dtype="<f4").astype(np.float64)
            denom = query_norm * float(np.linalg.norm(vec))
            score = float(np.dot(query, vec) / denom) if denom > 0 else 0.0
            scored.append((chunk_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: max(k, 0)]

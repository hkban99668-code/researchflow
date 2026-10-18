"""T-030 嵌入器：HashEmbedder 确定性（A1）与 OpenAICompatEmbedder 的 respx 测试（A7）。"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from researchflow.core.config import Settings
from researchflow.core.embedding import HashEmbedder, OpenAICompatEmbedder, build_embedder
from researchflow.core.errors import ConfigError, LLMError

# --- A1：HashEmbedder 确定性 --------------------------------------------------


def test_hash_embedder_same_text_same_vector():
    embedder = HashEmbedder()
    vectors = embedder.embed(["缺失值插补方法 missing data", "缺失值插补方法 missing data"])
    assert vectors[0] == vectors[1]


def test_hash_embedder_different_text_different_vector():
    embedder = HashEmbedder()
    vectors = embedder.embed(["缺失值插补", "回归系数估计"])
    assert vectors[0] != vectors[1]


def test_hash_embedder_l2_normalized():
    embedder = HashEmbedder(dim=256)
    vectors = embedder.embed(["multiple imputation for missing data", "缺失值插补"])
    for vector in vectors:
        norm = sum(component**2 for component in vector) ** 0.5
        assert norm == pytest.approx(1.0)
    # 空文本没有特征：零向量（不抛错，norm 保持 0）
    empty = embedder.embed([""])[0]
    assert all(component == 0.0 for component in empty)


def test_hash_embedder_dim_and_no_network():
    embedder = HashEmbedder(dim=128)
    vectors = embedder.embed(["any text"])
    assert embedder.dim == 128
    assert len(vectors[0]) == 128
    # Fake 用量恒为 0
    assert embedder.last_usage is not None
    assert embedder.last_usage.total_tokens == 0


# --- A7：OpenAICompatEmbedder（respx 打桩，不访问真实网络） -------------------


def _embed_response(vectors: list[list[float]], prompt_tokens: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [{"index": i, "embedding": vec} for i, vec in enumerate(vectors)],
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        },
    )


def test_openai_embedder_batches_and_preserves_order():
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://emb.test/v1/embeddings").mock(
            side_effect=[
                _embed_response([[1.0, 0.0], [0.0, 1.0]], 10),
                _embed_response([[1.0, 1.0], [0.5, 0.5]], 8),
                _embed_response([[0.2, 0.8]], 2),
            ]
        )
        embedder = OpenAICompatEmbedder(
            "https://emb.test/v1", "key", "text-embedding-test", batch_size=2
        )
        vectors = embedder.embed([f"text-{i}" for i in range(5)])

        assert len(vectors) == 5
        assert vectors[0] == [1.0, 0.0]
        assert vectors[4] == [0.2, 0.8], "同序返回"
        assert route.call_count == 3, "5 条文本按 batch_size=2 切成 3 批"
        # 批量切分正确：每批 input 长度 2/2/1
        sizes = [len(json.loads(call.request.content)["input"]) for call in route.calls]
        assert sizes == [2, 2, 1]
        # 用量按批累计（response_model="embedding" 的口径由调用方记录）
        assert embedder.last_usage is not None
        assert embedder.last_usage.prompt_tokens == 20
        assert embedder.last_usage.attempts == 3


def test_openai_embedder_error_tells_how_to_fix():
    with respx.mock:
        respx.post("https://emb.test/v1/embeddings").mock(
            return_value=httpx.Response(500, json={"error": "boom"})
        )
        embedder = OpenAICompatEmbedder("https://emb.test/v1", "key", "m")
        with pytest.raises(LLMError) as exc:
            embedder.embed(["text"])
        assert "嵌入请求失败" in str(exc.value)
        assert "RF_EMBED_BASE_URL" in json.dumps(exc.value.detail, ensure_ascii=False), (
            "错误信息要告诉怎么改"
        )


def test_openai_embedder_empty_input_no_request():
    embedder = OpenAICompatEmbedder("https://emb.test/v1", "key", "m")
    assert embedder.embed([]) == []


def test_build_embedder_missing_model_config_error():
    # Settings 默认没有 embedding.model，也没有 llm.api_key：报清楚的 ConfigError
    with pytest.raises(ConfigError) as exc:
        build_embedder(Settings())
    assert "embedding.model" in str(exc.value)


def test_build_embedder_falls_back_to_llm_config():
    settings = Settings()
    settings.llm.base_url = "https://llm.test/v1"
    settings.llm.api_key = "llm-key"
    settings.embedding.model = "text-embedding-test"
    embedder = build_embedder(settings)
    assert isinstance(embedder, OpenAICompatEmbedder)
    assert embedder.base_url == "https://llm.test/v1", "base_url 缺省回落 llm"
    assert embedder.api_key == "llm-key", "api_key 缺省回落 llm"


def test_build_embedder_missing_api_key_config_error():
    settings = Settings()
    settings.embedding.model = "m"
    with pytest.raises(ConfigError) as exc:
        build_embedder(settings)
    assert "api_key" in str(exc.value)

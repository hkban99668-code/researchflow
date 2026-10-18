"""嵌入器：Fake（确定性特征哈希）与 OpenAI 兼容 /embeddings 共用同一契约。

- `Embedder` 是 Protocol：`name` / `dim` / `embed()`，与 `Retriever` 同风格。
- `HashEmbedder` 不联网、确定性，供全部测试与离线评测的 fake 层使用。
- `OpenAICompatEmbedder` 走 OpenAI 兼容 `/embeddings` 接口，由 Claude 在真实层运行。

用量口径沿用 LLM 适配器：`last_usage` 旁路带出（token 与请求次数），
调用方以 `response_model="embedding"` 记录；Fake 恒为 0。
配置项 `embedding.base_url / model / api_key`，缺省回落到 `llm` 的同名配置；
环境变量 `RF_EMBED_BASE_URL / RF_EMBED_MODEL / RF_EMBED_API_KEY` 可覆盖。
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Protocol

import httpx

from .config import Settings
from .errors import ConfigError, LLMError
from .schemas import LLMUsage

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
# 单个 CJK 字符的区间判断用不到，tokenize 里按段处理


def _tokens(text: str) -> list[str]:
    """分词：英文小写词（长度 1 的丢弃）+ 中文字二元组（单字成段时保留该字）。

    与 `tools/bm25.py` 的分词规则保持一致（词法两路对同一文本应看到同一组特征）。
    这里不 import tools，避免 core 反向依赖上层。
    """
    tokens: list[str] = []
    for segment in _TOKEN_RE.findall(text.lower()):
        if "\u4e00" <= segment[0] <= "\u9fff":
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        elif len(segment) >= 2:
            tokens.append(segment)
    return tokens


class Embedder(Protocol):
    """嵌入器契约：批量、同序返回。"""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本嵌入为向量，顺序与输入一致。"""
        ...

    @property
    def last_usage(self) -> LLMUsage | None:
        """上一次 embed() 的累计用量；调用前为 None。"""
        ...

    @property
    def last_duration_ms(self) -> int | None:
        """上一次 embed() 的墙钟耗时；调用前为 None。"""
        ...


class HashEmbedder:
    """特征哈希嵌入器（Fake）：确定性、不联网。

    英文按词、中文按字二元组分词，每个 token 经 blake2b 哈希到固定维度的一个
    桶并累加，最后 L2 归一化。Python 内置 `hash()` 有进程级盐化，不可用于
    可复现测试，所以用 hashlib。
    """

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim
        # fake 层与 FakeLLM 同口径：tracing 记成本 0（不算"未配置"）。
        self.model = "fake"
        self.last_usage: LLMUsage | None = None
        self.last_duration_ms: int | None = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dim
                vec[index] += 1.0
            norm = sum(component * component for component in vec) ** 0.5
            if norm > 0:
                vec = [component / norm for component in vec]
            vectors.append(vec)
        # Fake 不花钱：用量恒为 0，耗时恒为 0，保证 trace 确定性。
        self.last_usage = LLMUsage(attempts=0)
        self.last_duration_ms = 0
        return vectors


class OpenAICompatEmbedder:
    """OpenAI 兼容 `/embeddings` 适配器，批大小可配（默认 16）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        batch_size: int = 16,
        timeout_seconds: float = 60.0,
    ) -> None:
        if batch_size <= 0:
            raise ConfigError(f"embedding.batch_size 必须为正整数，收到 {batch_size}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.name = f"openai-compat:{model}"
        self.dim = 0  # 首次成功响应后回填，接口要求字段存在
        self.last_usage: LLMUsage | None = None
        self.last_duration_ms: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入：按 `batch_size` 切分逐批请求，同序拼回。"""
        if not texts:
            self.last_usage = LLMUsage(attempts=0)
            self.last_duration_ms = 0
            return []

        usage = LLMUsage(attempts=0)
        self.last_usage = None
        vectors: list[list[float] | None] = [None] * len(texts)
        started = time.perf_counter()
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                payload = self._post(batch)
                # 失败的请求同样花钱：attempts 每批 +1，token 从 usage 读（缺失记 0）。
                raw_usage = payload.get("usage")
                usage = LLMUsage(
                    prompt_tokens=usage.prompt_tokens
                    + (
                        int(raw_usage.get("prompt_tokens") or 0)
                        if isinstance(raw_usage, dict)
                        else 0
                    ),
                    attempts=usage.attempts + 1,
                )
                self.last_usage = usage
                data = payload.get("data")
                if not isinstance(data, list) or len(data) != len(batch):
                    raise LLMError(
                        f"嵌入响应的 data 条数与输入不一致: "
                        f"{len(data) if isinstance(data, list) else '缺失'} != {len(batch)}",
                        detail=_how_to_fix_detail(),
                    )
                for item in data:
                    index = int(item.get("index", 0))
                    embedding = item.get("embedding")
                    if not isinstance(embedding, list):
                        raise LLMError(
                            "嵌入响应的 embedding 不是数组",
                            detail=_how_to_fix_detail(),
                        )
                    if index < 0 or start + index >= len(texts):
                        raise LLMError(
                            f"嵌入响应的 index 越界: {index}",
                            detail=_how_to_fix_detail(),
                        )
                    vectors[start + index] = [float(x) for x in embedding]
            missing = [i for i, vec in enumerate(vectors) if vec is None]
            if missing:
                raise LLMError(
                    f"嵌入响应缺少 {len(missing)} 条向量（index: {missing[:5]}…）",
                    detail=_how_to_fix_detail(),
                )
            first = vectors[0] or []
            if self.dim == 0:
                self.dim = len(first)
            elif len(first) != self.dim:
                raise LLMError(
                    f"同一批嵌入的维度不一致: {len(first)} != {self.dim}",
                    detail=_how_to_fix_detail(),
                )
            self.last_usage = usage
            return [vec for vec in vectors if vec is not None]
        finally:
            self.last_duration_ms = int((time.perf_counter() - started) * 1000)

    def _post(self, batch: list[str]) -> dict:
        """发一批嵌入请求并返回响应体（内部方法，形状不对外承诺）。"""
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            # 服务端的错误体通常直接说明原因（如批大小超限），截断后带上，方便调用方改配置。
            body = exc.response.text[:300]
            raise LLMError(
                f"嵌入请求失败: HTTP {exc.response.status_code}: {body}"
                "（常见原因：批大小超过服务商上限，可设置 RF_EMBED_BATCH_SIZE 调小）",
                detail=_how_to_fix_detail(),
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(
                f"嵌入请求失败: {exc}",
                detail=_how_to_fix_detail(),
            ) from exc
        if not isinstance(payload, dict):
            raise LLMError(
                f"嵌入响应不是 JSON 对象: {type(payload).__name__}",
                detail=_how_to_fix_detail(),
            )
        return payload


def _how_to_fix_detail() -> dict:
    """嵌入失败时告诉调用方"怎么改"（沿用 T-013 工具错误的做法）。"""
    return {
        "how_to_fix": (
            "检查 embedding 配置：RF_EMBED_BASE_URL / RF_EMBED_MODEL / "
            "RF_EMBED_API_KEY（或 settings.yaml 的 embedding 段），"
            "base_url 缺省回落 llm.base_url，api_key 缺省回落 llm.api_key；"
            "确认模型名是该 Provider 的嵌入模型（如 text-embedding-3-small）。"
        )
    }


class UsageTrackingEmbedder:
    """累计多次 `embed()` 的用量与耗时（T-034）。

    `HybridReranker` 一次重排会分多次调用 `embed()`（候选块一次、每条查询各一次），
    底层适配器的 `last_usage` 只保留最后一次，直接记账会漏掉候选块那批 token。
    包装后 `last_usage` / `last_duration_ms` 表示 `reset_usage()` 之后的累计值，
    供 ① 节点写一条 `response_model="embedding"` 的 `LLMCallRecord`。
    """

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        # 契约要求 `name` / `dim` 是属性而不是只读 property；委托给底层实例的取值
        # 在构造时固定即可（`dim` 只在首次响应后回填，重排器不读它）。
        self.name = inner.name
        self.dim = inner.dim
        self.reset_usage()

    @property
    def model(self) -> str | None:
        """真实嵌入模型名（透传给 tracing 查价格表）；HashEmbedder 没有这个属性。"""
        model = getattr(self.inner, "model", None)
        return model if isinstance(model, str) and model else None

    def reset_usage(self) -> None:
        """清零累计用量；每次重排前调用，避免把上一次回路的 token 记进来。"""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._attempts = 0
        self._duration_ms = 0
        self.last_usage: LLMUsage | None = None
        self.last_duration_ms: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.inner.embed(texts)
        usage = self.inner.last_usage
        if usage is not None:
            self._prompt_tokens += usage.prompt_tokens
            self._completion_tokens += usage.completion_tokens
            self._attempts += usage.attempts
        duration = self.inner.last_duration_ms
        if duration is not None:
            self._duration_ms += duration
        self.last_usage = LLMUsage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            attempts=self._attempts,
        )
        self.last_duration_ms = self._duration_ms
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    """按配置构造真实嵌入器；缺配置报清楚的 `ConfigError`（不是模糊的运行时错）。"""
    emb = settings.embedding
    if not emb.model:
        raise ConfigError(
            "使用嵌入时必须配置 embedding.model（RF_EMBED_MODEL 或 settings.yaml 的 "
            "embedding.model）"
        )
    api_key = emb.api_key or settings.llm.api_key
    if not api_key:
        raise ConfigError(
            "使用嵌入时缺少 api_key：配置 RF_EMBED_API_KEY 或 settings.yaml 的 "
            "embedding.api_key（缺省回落 llm.api_key，当前两者都为空）"
        )
    return OpenAICompatEmbedder(
        base_url=emb.base_url or settings.llm.base_url,
        api_key=api_key,
        model=emb.model,
        batch_size=emb.batch_size,
        timeout_seconds=settings.llm.timeout_seconds,
    )

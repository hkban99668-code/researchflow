"""OpenAlex 检索器：`Retriever` 的第一个真实学术数据源实现。

设计要点（T-022）：

- **可复现性靠缓存**：真实 API 的结果会随时间变化（新论文、排序调整），破坏产物
  可复现。所以每次请求的响应按"规范化查询 + k"做键缓存在本地（`.cache/openalex/`），
  同一查询第二次运行时直接读缓存，不发请求；`LiteratureRef.cached` 标记结果来自
  缓存（True）还是网络（False）。想要新结果就显式清缓存，而不是悄悄变化。
- **网络失败不能伪装成"没找到"**：超时、HTTP 4xx/5xx、响应结构异常都抛业务异常，
  不返回空列表（空列表在这个项目里的意思是"真的没命中"，会触发"检索为空"终止）。
- **连接类错误退避重试**（T-028）：本机代理不稳时 OpenAlex 会偶发 `ConnectError`/
  `ReadError`/SSL EOF，直接抛会白占一次检索预算。`_fetch` 对 `httpx.TransportError`
  的子类（超时除外）退避重试最多 2 次（0.5s、1s）；HTTP 5xx 与 429 重试一次；
  4xx 与响应结构异常不重试。sleep 可注入，测试不真睡。

OpenAlex `works` 响应结构（见汇报"调查结论"）：信封 `{meta, results[]}`，每个 work
含 `id`（完整 URL）、`doi`（URL 或 null）、`title`、`publication_year`（int 或 null）、
`abstract_inverted_index`（word -> 位置列表的倒排索引）、`primary_location`（内含
dehydrated 的 `source.display_name`）。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ..core.config import Settings
from ..core.errors import RetrievalError
from ..core.schemas import LiteratureRef

# OpenAlex works 列表端点
WORKS_ENDPOINT = "https://api.openalex.org/works"

# 请求只取需要的字段，减小响应体积；`primary_location` 返回 dehydrated 对象，
# 仍含 `source.display_name`。
SELECT_FIELDS = "id,doi,title,publication_year,abstract_inverted_index,primary_location"

# 单次请求超时（秒）；超时后重试一次，第二次放宽到 2 倍（2026-10-16：经代理访问时 10s 偶发超时，
# 评测建池因此整体失败）。最坏情况单次查询 10 + 20 秒，仍是有界的。
TIMEOUT_SECONDS = 10.0
TIMEOUT_RETRY_FACTOR = 2.0

# 连接类错误（httpx.TransportError，超时除外）的退避重试延迟：长度即最多重试次数。
CONNECT_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.0)

# HTTP 5xx / 429 只重试一次（瞬时故障；再多重试也不改变服务端的状态）。
SERVER_RETRY_ATTEMPTS = 1
SERVER_RETRY_BACKOFF_SECONDS = 0.5

# 缓存键规范化用：合并连续空白（含换行），与本地语料的标题规范化保持一致
_WHITESPACE_RE = re.compile(r"\s+")

# 缓存文件名用规范化查询 + k 的 sha256 前 16 位，避免长查询撑爆文件名
_CACHE_KEY_LEN = 16


def _normalize_query(query: str) -> str:
    """规范化查询：去首尾空白、合并连续空白、转小写（与任务卡口径一致）。"""
    return _WHITESPACE_RE.sub(" ", query.strip()).lower()


def _cache_filename(query: str, k: int) -> str:
    """缓存文件名：`sha256(规范化查询 | k)` 的前 16 位十六进制。"""
    payload = f"{_normalize_query(query)}|{k}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_CACHE_KEY_LEN]


def _abstract_from_inverted_index(inverted: dict | None) -> str:
    """把 `abstract_inverted_index` 还原成摘要文本（按位置排序拼接）。

    倒排索引是 `{word: [位置...]}`，位置 0 起；把每个词放到对应位置，按位置升序
    拼起来即可。没有摘要时返回空串。
    """
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, indices in inverted.items():
        for index in indices:
            positions[index] = word
    return " ".join(positions[index] for index in sorted(positions))


def _work_to_ref(work: dict, *, cached: bool) -> LiteratureRef:
    """把一条 work 记录转成 `LiteratureRef`；缺失字段落空值，不因缺字段而抛错。"""
    work_id = str(work.get("id") or "")
    # ref_id 从 id 的 URL 末段取 W 编号，例如 `openalex:W2741809807`
    openalex_id = work_id.rsplit("/", 1)[-1] if "/" in work_id else work_id
    primary = work.get("primary_location") or {}
    source = (primary.get("source") or {}).get("display_name") or ""
    return LiteratureRef(
        ref_id=f"openalex:{openalex_id}",
        title=str(work.get("title") or ""),
        source=source,
        year=work.get("publication_year"),
        url=str(work.get("doi") or ""),  # url 优先 DOI 链接
        summary=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
        retriever="openalex",
        cached=cached,
    )


class OpenAlexRetriever:
    """`Retriever` 的 OpenAlex 实现：免 Key、限流友好，带本地缓存。"""

    def __init__(self, settings: Settings, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self._settings = settings
        self._cache_dir = settings.project_root / ".cache" / "openalex"
        # 重试之间的等待；可注入，测试里换成记录器就不真睡。
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "openalex"

    def _cache_path(self, query: str, k: int) -> Path:
        return self._cache_dir / _cache_filename(query, k)

    def _load_cached(self, path: Path) -> list[dict] | None:
        """读缓存；存在且是合法 JSON 时返回 results 列表，否则返回 None。"""
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        results = payload.get("results") if isinstance(payload, dict) else None
        return results if isinstance(results, list) else None

    def _fetch(self, query: str, k: int) -> list[dict]:
        """请求 works 列表，返回 results；网络/结构失败一律抛业务异常。

        连接类错误（`httpx.TransportError`，超时除外）退避重试最多 2 次；超时重试一次
        （超时时间翻倍）；HTTP 5xx 与 429 重试一次；4xx 与响应结构异常不重试。重试耗尽时错误信息写明已重试
        次数，便于事后区分"网络抖了一下"和"一直连不上"。
        """
        params: dict[str, str] = {
            "search": query,
            "per-page": str(k),
            "select": SELECT_FIELDS,
        }
        mailto = self._settings.retrieval.openalex.mailto
        if mailto:
            params["mailto"] = mailto

        attempt = 0
        timeouts = 0
        while True:
            attempt += 1
            timeout = TIMEOUT_SECONDS * (TIMEOUT_RETRY_FACTOR if timeouts else 1)
            try:
                response = httpx.get(WORKS_ENDPOINT, params=params, timeout=timeout)
            except httpx.TimeoutException as exc:
                # 超时只重试一次且放宽超时时间，避免卡住的代理把等待叠加得太久。
                timeouts += 1
                if timeouts == 1:
                    continue
                raise RetrievalError(
                    f"OpenAlex 请求超时（{TIMEOUT_SECONDS}s，重试一次后放宽到 "
                    f"{TIMEOUT_SECONDS * TIMEOUT_RETRY_FACTOR}s 仍超时）"
                ) from exc
            except httpx.TransportError as exc:
                # 连接类错误（ConnectError/ReadError/SSL EOF 等）：退避重试后仍失败才抛。
                if attempt <= len(CONNECT_RETRY_BACKOFF_SECONDS):
                    self._sleep(CONNECT_RETRY_BACKOFF_SECONDS[attempt - 1])
                    continue
                raise RetrievalError(
                    f"OpenAlex 请求失败（网络错误，已重试 {len(CONNECT_RETRY_BACKOFF_SECONDS)} 次）: "
                    f"{type(exc).__name__}"
                ) from exc
            except httpx.HTTPError as exc:
                raise RetrievalError(
                    f"OpenAlex 请求失败（网络错误）: {type(exc).__name__}"
                ) from exc

            if response.status_code >= 400:
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt <= SERVER_RETRY_ATTEMPTS:
                        self._sleep(SERVER_RETRY_BACKOFF_SECONDS)
                        continue
                raise RetrievalError(f"OpenAlex 请求失败（HTTP {response.status_code}）")
            try:
                payload = response.json()
            except ValueError as exc:
                raise RetrievalError("OpenAlex 响应不是合法 JSON") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise RetrievalError("OpenAlex 响应结构异常：缺少 results 列表")
            return payload["results"]

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        """按查询检索 OpenAlex，最多 `k` 条；命中缓存时不发请求。"""
        cache_path = self._cache_path(query, k)
        cached = self._load_cached(cache_path)
        if cached is not None:
            return [_work_to_ref(work, cached=True) for work in cached]

        results = self._fetch(query, k)
        # 先写缓存再返回：同一次进程里第二次调用也能命中
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8"
        )
        return [_work_to_ref(work, cached=False) for work in results]

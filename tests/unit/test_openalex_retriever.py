"""T-022：OpenAlex 检索器、本地缓存与 build_retriever 选择（A1–A6）。

全部用 respx 打桩，不发真实网络请求。
"""

from __future__ import annotations

import httpx
import pytest
import respx

from researchflow.core.config import OpenAlexSettings, RetrievalSettings, Settings, load_settings
from researchflow.core.errors import ConfigError, ResearchFlowError, RetrievalError
from researchflow.tools.literature import LocalCorpusRetriever, build_retriever
from researchflow.tools.openalex import (
    OpenAlexRetriever,
    _abstract_from_inverted_index,
    _cache_filename,
    _normalize_query,
)

WORKS_URL = "https://api.openalex.org/works"

# 一条典型 work：倒排索引还原后应为 "study shows an effect"
WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1234/example",
    "title": "数据质量对实证研究结论的影响",
    "publication_year": 2021,
    "abstract_inverted_index": {"study": [0], "shows": [1], "an": [2], "effect": [3]},
    "primary_location": {"source": {"display_name": "Journal of Data Quality"}},
}


class SleepRecorder:
    """记录退避等待时长、不真睡；注入给 `OpenAlexRetriever` 做重试测试。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _settings(tmp_path, *, mailto: str = "") -> Settings:
    return Settings(
        project_root=tmp_path,
        retrieval=RetrievalSettings(backend="openalex", openalex=OpenAlexSettings(mailto=mailto)),
    )


def _retriever(tmp_path, *, mailto: str = "") -> OpenAlexRetriever:
    return OpenAlexRetriever(_settings(tmp_path, mailto=mailto))


def _retrying_retriever(tmp_path) -> tuple[OpenAlexRetriever, SleepRecorder]:
    """不真睡的检索器 + 记录到的退避时长。"""
    sleeper = SleepRecorder()
    return OpenAlexRetriever(_settings(tmp_path), sleep=sleeper), sleeper


def _works_response(results: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"meta": {"count": len(results)}, "results": results})


# --- 摘要还原与缓存键（辅助函数直接测） ---------------------------------------


def test_normalize_query_and_cache_filename() -> None:
    assert _normalize_query("  数据  质量 ") == "数据 质量"
    assert _cache_filename("  数据  质量 ", 3) == _cache_filename("数据 质量", 3)
    assert _cache_filename("数据 质量", 3) != _cache_filename("数据 质量", 4)


def test_abstract_from_inverted_index() -> None:
    assert _abstract_from_inverted_index(WORK["abstract_inverted_index"]) == "study shows an effect"
    assert _abstract_from_inverted_index(None) == ""
    assert _abstract_from_inverted_index({}) == ""


# --- A1 响应解析 ---------------------------------------------------------------


def test_search_parses_work_into_literature_ref(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(return_value=_works_response([WORK]))
        refs = _retriever(tmp_path).search("数据质量", k=3)

    assert route.call_count == 1
    assert len(refs) == 1
    ref = refs[0]
    assert ref.ref_id == "openalex:W2741809807"
    assert ref.title == "数据质量对实证研究结论的影响"
    assert ref.source == "Journal of Data Quality"
    assert ref.year == 2021
    assert ref.url == "https://doi.org/10.1234/example"
    assert ref.summary == "study shows an effect"
    assert ref.retriever == "openalex"
    assert ref.cached is False


# --- A2/A3 缓存 ----------------------------------------------------------------


def test_second_call_reads_cache_without_request(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(return_value=_works_response([WORK]))
        retriever = _retriever(tmp_path)
        first = retriever.search("数据质量", k=3)
        second = retriever.search("数据质量", k=3)

    assert route.call_count == 1
    assert first[0].cached is False
    assert second[0].cached is True
    assert [r.ref_id for r in first] == [r.ref_id for r in second]


def test_case_and_whitespace_variants_share_cache(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(return_value=_works_response([WORK]))
        retriever = _retriever(tmp_path)
        retriever.search("  数据  质量 ", k=3)
        second = retriever.search("数据\t质量", k=3)

    assert route.call_count == 1
    assert second[0].cached is True


# --- A4 网络/结构失败抛业务异常，不返回空列表 ----------------------------------


def test_timeout_raises_business_error(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(side_effect=httpx.ReadTimeout("模拟超时"))
        with pytest.raises(ResearchFlowError):
            _retriever(tmp_path).search("数据质量", k=3)

    # 超时只重试一次（第二次放宽超时时间），仍超时就抛业务异常
    assert route.call_count == 2


def test_timeout_then_success_retries_once(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(
            side_effect=[httpx.ReadTimeout("模拟超时"), httpx.Response(200, json={"results": []})]
        )
        assert _retriever(tmp_path).search("数据质量", k=3) == []

    assert route.call_count == 2


def test_http_500_raises_business_error(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKS_URL).mock(return_value=httpx.Response(500, json={}))
        retriever, sleeper = _retrying_retriever(tmp_path)
        with pytest.raises(ResearchFlowError):
            retriever.search("数据质量", k=3)

    # 5xx 重试一次后仍失败
    assert sleeper.delays == [0.5]


def test_http_404_raises_business_error(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKS_URL).mock(return_value=httpx.Response(404, json={}))
        with pytest.raises(ResearchFlowError):
            _retriever(tmp_path).search("数据质量", k=3)


def test_unexpected_structure_raises_business_error(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKS_URL).mock(return_value=httpx.Response(200, json={"meta": {}}))
        with pytest.raises(ResearchFlowError):
            _retriever(tmp_path).search("数据质量", k=3)


# --- T-028 A2：连接错误退避重试、4xx 不重试、缓存命中不发请求 -------------------


def test_connect_error_retries_with_backoff_then_succeeds_and_caches(tmp_path) -> None:
    """连接错误第 1、2 次失败、第 3 次成功：退避 0.5s/1s，结果入缓存且只写一份。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(
            side_effect=[
                httpx.ConnectError("模拟连接失败"),
                httpx.ReadError("模拟读取失败"),
                _works_response([WORK]),
            ]
        )
        retriever, sleeper = _retrying_retriever(tmp_path)
        refs = retriever.search("数据质量", k=3)
        second = retriever.search("数据质量", k=3)  # 缓存命中，不再发请求

    assert route.call_count == 3
    assert sleeper.delays == [0.5, 1.0]
    assert [ref.ref_id for ref in refs] == ["openalex:W2741809807"]
    assert refs[0].cached is False
    assert second[0].cached is True
    assert len(list((tmp_path / ".cache" / "openalex").iterdir())) == 1


def test_connect_error_exhausted_raises_with_retry_count(tmp_path) -> None:
    """连续 3 次连接失败：抛 RetrievalError，信息仍是"网络错误"并写明重试次数。"""
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(side_effect=httpx.ConnectError("模拟连接失败"))
        retriever, sleeper = _retrying_retriever(tmp_path)
        with pytest.raises(RetrievalError) as excinfo:
            retriever.search("数据质量", k=3)

    assert route.call_count == 3
    assert sleeper.delays == [0.5, 1.0]
    assert "网络错误" in str(excinfo.value)
    assert "已重试 2 次" in str(excinfo.value)


def test_http_404_is_not_retried(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(return_value=httpx.Response(404, json={}))
        retriever, sleeper = _retrying_retriever(tmp_path)
        with pytest.raises(RetrievalError, match="HTTP 404"):
            retriever.search("数据质量", k=3)

    assert route.call_count == 1
    assert sleeper.delays == []


@pytest.mark.parametrize("status", [429, 500, 503])
def test_server_error_retried_once_then_succeeds(tmp_path, status: int) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(
            side_effect=[httpx.Response(status, json={}), _works_response([WORK])]
        )
        retriever, sleeper = _retrying_retriever(tmp_path)
        refs = retriever.search("数据质量", k=3)

    assert route.call_count == 2
    assert sleeper.delays == [0.5]
    assert [ref.ref_id for ref in refs] == ["openalex:W2741809807"]


@pytest.mark.parametrize("status", [429, 500])
def test_server_error_exhausts_after_one_retry(tmp_path, status: int) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(return_value=httpx.Response(status, json={}))
        retriever, _sleeper = _retrying_retriever(tmp_path)
        with pytest.raises(RetrievalError, match=f"HTTP {status}"):
            retriever.search("数据质量", k=3)

    assert route.call_count == 2


def test_cache_hit_makes_no_request(tmp_path) -> None:
    """缓存命中不发任何请求：第二次检索在"任何请求都报错"的路由下也能拿到结果。"""
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKS_URL).mock(return_value=_works_response([WORK]))
        first = _retriever(tmp_path).search("数据质量", k=3)

    assert first[0].cached is False

    with respx.mock(assert_all_called=False) as router2:
        route = router2.get(WORKS_URL).mock(side_effect=AssertionError("缓存命中不应发请求"))
        second = _retriever(tmp_path).search("数据质量", k=3)

    assert route.call_count == 0
    assert second[0].cached is True


# --- A5 真实 0 命中 ------------------------------------------------------------


def test_empty_results_returns_empty_list(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKS_URL).mock(return_value=_works_response([]))
        refs = _retriever(tmp_path).search("完全无关的查询 xyz", k=3)

    assert refs == []


# --- 请求参数与 mailto ---------------------------------------------------------


def test_request_params_and_mailto(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(return_value=_works_response([]))
        _retriever(tmp_path, mailto="researcher@example.com").search("数据 质量", k=5)

    params = route.calls[0].request.url.params
    assert params["search"] == "数据 质量"
    assert params["per-page"] == "5"
    assert "id" in params["select"]
    assert "abstract_inverted_index" in params["select"]
    assert params["mailto"] == "researcher@example.com"


def test_mailto_omitted_when_empty(tmp_path) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(WORKS_URL).mock(return_value=_works_response([]))
        _retriever(tmp_path).search("数据", k=3)

    assert "mailto" not in route.calls[0].request.url.params


# --- A6 build_retriever 选择 ---------------------------------------------------


def test_build_retriever_env_override_to_openalex(monkeypatch, project_root) -> None:
    monkeypatch.setenv("RF_RETRIEVER", "openalex")
    settings = load_settings(project_root=project_root)
    retriever = build_retriever(settings)

    assert isinstance(retriever, OpenAlexRetriever)
    assert retriever.name == "openalex"


def test_build_retriever_invalid_backend_raises(monkeypatch, project_root) -> None:
    monkeypatch.setenv("RF_RETRIEVER", "arxiv")
    settings = load_settings(project_root=project_root)

    with pytest.raises(ConfigError):
        build_retriever(settings)


def test_build_retriever_default_local(monkeypatch, project_root) -> None:
    monkeypatch.delenv("RF_RETRIEVER", raising=False)
    settings = load_settings(project_root=project_root)
    retriever = build_retriever(settings)

    assert isinstance(retriever, LocalCorpusRetriever)
    assert retriever.name == "local"

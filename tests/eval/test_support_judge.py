"""T-032 引用支持度评测：指标纯函数手算、旧报告报错、预演不发请求、缓存命中。

A1 缺字段报错、A2 预演零请求、A3 缓存第二次不发请求（respx 计数）、A4 κ 与两种
口径支持率手算一致、A5 抽样表固定种子每档最多 5 条。全程不联网、不调真实模型。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import respx

# 让 `import evals` 可用（与 test_citation_eval 同法）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.citation import support_judge as sj  # noqa: E402
from evals.citation.run import _citation_records  # noqa: E402

S, P, U = "supported", "partial", "unsupported"


# ---------------------------------------------------------------------------
# A4：κ 与两种口径支持率，手算例子
# ---------------------------------------------------------------------------


def test_cohen_kappa_hand_computed() -> None:
    """8 对标签的手算例子：p_o=6/8，p_e=22/64，κ=(0.75-0.34375)/0.65625=13/21。"""
    a = [S, S, P, U, S, P, U, U]
    b = [S, P, P, U, S, S, U, U]

    # 边际：两个评委都是 s=3, p=2, u=3 → p_e = (9+4+9)/64 = 22/64
    assert sj.cohen_kappa(a, b) == pytest.approx(13 / 21)


def test_cohen_kappa_boundaries_no_division_by_zero() -> None:
    """全一致 κ=1；空输入 κ=0；p_e=1（分布完全错开）κ=0，不除零。"""
    assert sj.cohen_kappa([S, P, U], [S, P, U]) == pytest.approx(1.0)
    assert sj.cohen_kappa([], []) == 0.0
    assert sj.cohen_kappa([S, S, S], [U, U, U]) == 0.0


def test_cohen_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="标签数不一致"):
        sj.cohen_kappa([S], [])


def test_support_rates_hand_computed() -> None:
    """5 对标签：严格 1/5（只有 (S,S)），宽松 4/5（(U,U) 除外）。"""
    pairs = [(S, S), (S, U), (P, P), (U, U), (S, P)]
    rates = sj.support_rates(pairs)
    assert rates["strict"] == pytest.approx(1 / 5)
    assert rates["lenient"] == pytest.approx(4 / 5)


def test_support_rates_empty_no_division_by_zero() -> None:
    assert sj.support_rates([]) == {"strict": 0.0, "lenient": 0.0}


def test_support_rates_lenient_single_partial_not_counted() -> None:
    """单个 partial + 另一个 unsupported：既不严格也不宽松。"""
    assert sj.support_rates([(P, U), (U, P)]) == {"strict": 0.0, "lenient": 0.0}


# ---------------------------------------------------------------------------
# A1：真实层 JSON 的装载与旧报告缺字段的清楚报错
# ---------------------------------------------------------------------------


def _write_real_report(path: Path, cases: list[dict]) -> Path:
    path.write_text(
        json.dumps({"layer": "real", "cases": cases}, ensure_ascii=False), encoding="utf-8"
    )
    return path


def test_load_citation_rows_flattens_topic_id_ref_claim(tmp_path: Path) -> None:
    report = _write_real_report(
        tmp_path / "r.json",
        [
            {
                "id": "t01",
                "citation_stats": {"total": 2, "invalid": 0, "hallucination_rate": 0.0},
                "citations": [
                    {"ref_id": "openalex:W1", "claim": "方法 A 降低偏差"},
                    {"ref_id": "openalex:W2", "claim": "方法 B 更稳健"},
                ],
            },
            {"id": "t02", "citations": []},
        ],
    )
    rows = sj.load_citation_rows(report)
    assert rows == [
        {"topic_id": "t01", "ref_id": "openalex:W1", "claim": "方法 A 降低偏差"},
        {"topic_id": "t01", "ref_id": "openalex:W2", "claim": "方法 B 更稳健"},
    ]


def test_load_citation_rows_old_report_clear_error(tmp_path: Path) -> None:
    """A1：旧报告缺 `citations` 字段时，报错说明是旧报告并给出重跑命令。"""
    report = _write_real_report(
        tmp_path / "old.json",
        [{"id": "t01", "topic": "课题", "citation_stats": {"total": 6, "invalid": 0}}],
    )
    with pytest.raises(ValueError, match="citations") as exc:
        sj.load_citation_rows(report)
    assert "重跑" in str(exc.value)
    assert "evals.citation.run" in str(exc.value)


def test_run_real_record_contains_citations() -> None:
    """A1：`_citation_records` 把每条引用写成 {ref_id, claim}（真实层 JSON 的新字段）。"""
    from researchflow.core.schemas import Citation, DraftContent

    content = DraftContent(
        background="背景",
        findings="发现",
        limitation="局限",
        citations=[
            Citation(ref_id="openalex:W1", claim="claim 一"),
            Citation(ref_id="openalex:W1", claim="claim 二"),
        ],
    )
    assert _citation_records(content) == [
        {"ref_id": "openalex:W1", "claim": "claim 一"},
        {"ref_id": "openalex:W1", "claim": "claim 二"},
    ]


# ---------------------------------------------------------------------------
# A2：预演不发请求（respx 计数 0），打印预计调用次数
# ---------------------------------------------------------------------------


def _preview_inputs(tmp_path: Path) -> tuple[Path, Path]:
    report = _write_real_report(
        tmp_path / "r.json",
        [
            {"id": "t01", "citations": [{"ref_id": "W1", "claim": "c1"}]},
            {"id": "t02", "citations": [{"ref_id": "W404", "claim": "c2"}]},
        ],
    )
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {"topics": [{"candidates": [{"ref_id": "W1", "title": "T1", "abstract": "A1"}]}]}
        ),
        encoding="utf-8",
    )
    return report, pool


def test_preview_makes_no_requests_and_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A2：不带 --confirm-real-calls 不发任何请求，打印 1 条×2 评委=2 次与池外 1 条。"""
    report, pool = _preview_inputs(tmp_path)
    with respx.mock(assert_all_called=False) as router:
        exit_code = sj.main(["--input", str(report), "--pool", str(pool)])

    assert exit_code == 0
    assert len(router.calls) == 0, "预演不得发出任何 HTTP 请求"
    out = capsys.readouterr().out
    assert "1 条引用 × 2 个评委 = 2 次调用" in out
    assert "真实请求次数: 0" in out
    assert "池外引用" in out and "1 条" in out


def test_missing_input_report_exits_2(tmp_path: Path) -> None:
    exit_code = sj.main(["--input", str(tmp_path / "nope.json")])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# A3：缓存——同一 (模型, ref_id, claim) 第二次不发请求
# ---------------------------------------------------------------------------


def _judge_response(label: str, reason: str) -> dict:
    return {"choices": [{"message": {"content": json.dumps({"label": label, "reason": reason})}}]}


def test_cache_second_call_no_request(tmp_path: Path) -> None:
    """A3：同一 (模型, ref_id, claim) 第二次命中缓存，respx 计数为 1。"""
    pool = {"W1": {"title": "T1", "abstract": "A1"}}
    row = {"topic_id": "t01", "ref_id": "W1", "claim": "同一个 claim"}
    cache_dir = tmp_path / "cache"
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://judge.test/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=_judge_response(S, "摘要明确支持")),
            ]
        )
        client = sj.OpenAICompatClient(
            base_url="https://judge.test/v1", api_key="k", model="judge-model"
        )
        first = sj.judge_row(client, row, pool, cache_dir=cache_dir)
        second = sj.judge_row(client, row, pool, cache_dir=cache_dir)

        assert route.call_count == 1, "第二次必须命中缓存，不得再发请求"
        assert first.label == second.label == S
        assert first.reason == second.reason


def test_cache_key_distinguishes_claim(tmp_path: Path) -> None:
    """claim 不同（哪怕 ref_id 相同）不共用缓存。"""
    pool = {"W1": {"title": "T1", "abstract": "A1"}}
    cache_dir = tmp_path / "cache"
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://judge.test/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=_judge_response(S, "支持")),
                httpx.Response(200, json=_judge_response(U, "无关")),
            ]
        )
        client = sj.OpenAICompatClient(
            base_url="https://judge.test/v1", api_key="k", model="judge-model"
        )
        row_a = {"topic_id": "t01", "ref_id": "W1", "claim": "claim A"}
        row_b = {"topic_id": "t01", "ref_id": "W1", "claim": "claim B"}
        sj.judge_row(client, row_a, pool, cache_dir=cache_dir)
        sj.judge_row(client, row_b, pool, cache_dir=cache_dir)

        assert route.call_count == 2


# ---------------------------------------------------------------------------
# A5：抽样表固定种子、每档最多 5 条
# ---------------------------------------------------------------------------


def _labeled_rows(n_per_label: int = 8) -> list[dict[str, str]]:
    rows = []
    for label in (S, P, U):
        for i in range(n_per_label):
            rows.append(
                {
                    "topic_id": f"t{i:02d}",
                    "ref_id": f"W{i}",
                    "claim": f"claim {label} {i}",
                    "title": f"标题 {i}",
                    "judge_a": label,
                    "judge_b": label,
                    "agree": "1",
                    "consensus": label,
                    "reason_a": "理由A",
                    "reason_b": "理由B",
                }
            )
    return rows


def test_sample_table_fixed_seed_and_max_five_per_label() -> None:
    rows = _labeled_rows()
    first = sj.sample_table_rows(rows, seed=42)
    again = sj.sample_table_rows(rows, seed=42)
    other = sj.sample_table_rows(rows, seed=7)

    assert first == again, "同一种子必须抽出同样的行"
    assert first != other, "不同种子应抽出不同（本数据量下大概率不同）的行"
    assert len(first) == 15, "三档各最多 5 条"
    for label in (S, P, U):
        assert sum(row["consensus"] == label for row in first) == 5


def test_sample_table_fewer_rows_than_quota() -> None:
    rows = _labeled_rows(2)
    sampled = sj.sample_table_rows(rows, seed=42)
    assert len(sampled) == 6, "每档只有 2 条时全取"


def test_render_report_contains_rates_kappa_and_sampling(tmp_path: Path) -> None:
    report = {
        "generated_at": "2026-10-17T00:00:00+00:00",
        "input": "r.json",
        "pool": "pool.json",
        "judges": ["judge-a", "judge-b"],
        "n_rows": 2,
        "n_skipped": 1,
        "strict_rate": 0.5,
        "strict_n": 1,
        "lenient_rate": 1.0,
        "lenient_n": 2,
        "agree_rate": 1.0,
        "agree_n": 2,
        "kappa": 0.6190,
        "per_topic": [
            {"topic_id": "t01", "n": 2, "consensus_counts": "1/1/0", "disagree": 0, "strict": 1}
        ],
        "sample_seed": 42,
        "samples": [
            {
                "topic_id": "t01",
                "ref_id": "W1",
                "claim": "claim x",
                "title": "标题",
                "consensus": S,
                "reason_a": "a",
                "reason_b": "b",
            }
        ],
    }
    text = sj.render_report(report)
    assert "支持率（严格口径" in text and "50.00%" in text
    assert "宽松口径" in text and "100.00%" in text
    assert "Cohen's κ（三档）：0.6190" in text
    assert "抽样表" in text
    assert "随机种子 42" in text

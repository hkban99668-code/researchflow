"""REST 端点验收：A1、A5、A6、A7、A8、A9、A10 与 /trace、/healthz。"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from api.helpers import client_for, read_events_file, topic_body, wait_status
from researchflow.api import create_app

pytestmark = pytest.mark.anyio


# --- A1：发起 → 审核载荷 → resume → 终态摘要 ---------------------------------


async def test_full_run_approve_returns_review_then_manifest_summary(api_env):
    async with client_for(api_env) as client:
        created = await client.post("/runs", json=topic_body())
        assert created.status_code == 202
        assert created.json()["status"] == "running"
        run_id = created.json()["run_id"]

        assert await wait_status(client, run_id, "awaiting_review") == "awaiting_review"
        detail = (await client.get(f"/runs/{run_id}")).json()
        review = detail["review"]
        # 与 CLI 审核提示同一份信息：规模字段必须齐
        assert review["stage"] == "data_review"
        assert review["rows"] > 0 and review["columns"] > 0
        assert "missing_rate" in review and "dropped_rows" in review

        resumed = await client.post(
            f"/runs/{run_id}/resume", json={"decision": "approve", "reason": "api 测试"}
        )
        assert resumed.status_code == 202

        assert await wait_status(client, run_id, "ok") == "ok"
        summary = (await client.get(f"/runs/{run_id}")).json()["summary"]
        assert summary["status"] == "ok"
        assert summary["termination"] is None
        assert summary["validation"]["passed"] is True
        # A1：摘要含 trace 的 token / 成本 / 未配置计数；T-027 起 total_cost 是按
        # 币种分组的字典，currency 仅对旧 manifest 有值（新 manifest 为 None）。
        assert set(summary["trace"]) == {"total_tokens", "total_cost", "unpriced_calls", "currency"}
        assert isinstance(summary["trace"]["total_cost"], dict)
        assert summary["trace"]["currency"] is None


async def test_run_list_and_trace_and_healthz(api_env):
    async with client_for(api_env) as client:
        assert (await client.get("/healthz")).json() == {"status": "ok"}

        run_id = (await client.post("/runs", json=topic_body())).json()["run_id"]
        await wait_status(client, run_id, "awaiting_review")
        # 未结束：trace 404，列表里状态是 awaiting_review
        assert (await client.get(f"/runs/{run_id}/trace")).status_code == 404
        listed = (await client.get("/runs")).json()
        entry = next(item for item in listed if item["run_id"] == run_id)
        assert entry["status"] == "awaiting_review" and entry["topic"]

        await client.post(f"/runs/{run_id}/resume", json={"decision": "approve"})
        await wait_status(client, run_id, "ok")
        trace = await client.get(f"/runs/{run_id}/trace")
        assert trace.status_code == 200
        assert "trace_id" in trace.json()


async def test_legacy_manifest_trace_summary_displayed(api_env, api_settings):
    """A1：旧 manifest（`total_cost` 是数字、`currency` 是字符串）经 `GET /runs/{id}`
    摘要原样显示，不被新结构破坏；旧记录没有的 `unpriced_calls` 为 None。"""
    run_id = "0" * 12
    run_dir = api_settings.runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        json.dumps({"type": "run_end", "status": "ok"}) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "validation": {"passed": True, "summary": "旧运行"},
                "trace": {"total_tokens": 100, "total_cost": 0.0123, "currency": "CNY"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    async with client_for(api_env) as client:
        summary = (await client.get(f"/runs/{run_id}")).json()["summary"]
    trace = summary["trace"]
    assert trace["total_cost"] == 0.0123  # 旧数字型原样透传
    assert trace["currency"] == "CNY"  # 旧字符串币种原样透传
    assert trace["unpriced_calls"] is None  # 旧 manifest 没有该字段


# --- A5：路径穿越 -------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["abc", "0" * 13, "0" * 11])
async def test_bad_run_id_shapes_rejected(api_env, bad_id):
    async with client_for(api_env) as client:
        for path in (f"/runs/{bad_id}", f"/runs/{bad_id}/events", f"/runs/{bad_id}/trace"):
            response = await client.get(path)
            assert response.status_code == 404, path
            assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("suffix", ["", "/events", "/trace"])
async def test_dot_segments_rejected(api_env, suffix):
    """`..`/`.`（含百分号编码）一律 404：httpx 会归一化裸 `..`，编码形态到得了服务端。"""
    async with client_for(api_env) as client:
        for encoded in ["%2e%2e", "%2e%2e%2fx", "..%2fx", "%2E%2E/x"]:
            response = await client.get(f"/runs/{encoded}{suffix}")
            assert response.status_code == 404, encoded


# --- A6：状态机与并发 resume ---------------------------------------------------


async def test_resume_rejected_for_wrong_states(api_env):
    async with client_for(api_env) as client:
        run_id = (await client.post("/runs", json=topic_body())).json()["run_id"]
        await wait_status(client, run_id, "awaiting_review")

        # ① "正在运行"分支：塞一个不退出的线程占住 run（绕开竞态的确定性构造）
        blocker = threading.Event()
        assert api_env.state.manager.submit(run_id, blocker.wait) is True
        conflict = await client.post(f"/runs/{run_id}/resume", json={"decision": "approve"})
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "conflict"
        blocker.set()
        while api_env.state.manager.is_active(run_id):  # 等占位线程真正退出
            await asyncio.sleep(0.01)

        # ② 已结束分支
        resumed = await client.post(f"/runs/{run_id}/resume", json={"decision": "approve"})
        assert resumed.status_code == 202
        await wait_status(client, run_id, "ok")
        again = await client.post(f"/runs/{run_id}/resume", json={"decision": "approve"})
        assert again.status_code == 409
        assert "awaiting_review" in again.json()["error"]["message"]


async def test_concurrent_resume_only_one_wins(api_env):
    async with client_for(api_env) as client:
        run_id = (await client.post("/runs", json=topic_body())).json()["run_id"]
        await wait_status(client, run_id, "awaiting_review")
        responses = await asyncio.gather(
            client.post(f"/runs/{run_id}/resume", json={"decision": "approve"}),
            client.post(f"/runs/{run_id}/resume", json={"decision": "approve"}),
        )
        codes = sorted(response.status_code for response in responses)
        assert codes == [202, 409], codes
        await wait_status(client, run_id, "ok")


# --- A7：real 模式默认拒绝且不建目录 -------------------------------------------


async def test_real_mode_forbidden_without_env(api_env, api_settings, monkeypatch):
    monkeypatch.delenv("RF_API_ALLOW_REAL", raising=False)
    async with client_for(api_env) as client:
        response = await client.post("/runs", json={**topic_body(), "mode": "real"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "real_not_allowed"
        assert "RF_API_ALLOW_REAL" in response.json()["error"]["message"]
        # A7：一个 run 目录都不能建
        assert not api_settings.runs_dir.exists() or not any(api_settings.runs_dir.iterdir())


# --- A8：4 个并发 run 不串 -----------------------------------------------------


async def test_four_concurrent_runs_isolated(api_env, api_settings):
    async with client_for(api_env) as client:
        created = await asyncio.gather(*[client.post("/runs", json=topic_body()) for _ in range(4)])
        assert all(response.status_code == 202 for response in created)
        run_ids = [response.json()["run_id"] for response in created]
        assert len(set(run_ids)) == 4

        statuses = await asyncio.gather(
            *[wait_status(client, run_id, "awaiting_review") for run_id in run_ids]
        )
        assert statuses == ["awaiting_review"] * 4

        # 各自 events.jsonl 的 run_id 不混
        for run_id in run_ids:
            events = read_events_file(api_settings, run_id)
            assert events, run_id
            assert {event["run_id"] for event in events} == {run_id}

        # 共享 settings 不被串改（调查问题 2 的回归）
        assert api_settings.llm.mode == "auto"


# --- A9：线程内异常收敛成 error -------------------------------------------------


async def test_failing_topic_ends_as_error(api_env):
    async with client_for(api_env) as client:
        bad = topic_body(dataset="no_such_dataset.csv")
        run_id = (await client.post("/runs", json=bad)).json()["run_id"]
        assert await wait_status(client, run_id, "error") == "error"
        summary = (await client.get(f"/runs/{run_id}")).json()["summary"]
        assert summary["termination"]["kind"] == "error"
        assert summary["validation"]["passed"] is None


# --- A10：模拟重启（新 app、空线程表） ------------------------------------------


async def test_new_app_resumes_old_run(api_env, api_settings):
    async with client_for(api_env) as client:
        run_id = (await client.post("/runs", json=topic_body())).json()["run_id"]
        await wait_status(client, run_id, "awaiting_review")

    # 新 create_app()：线程表为空，只有磁盘上的 events/checkpoint/run_meta
    app2 = create_app(api_settings)
    try:
        async with client_for(app2) as restarted:
            detail = (await restarted.get(f"/runs/{run_id}")).json()
            assert detail["status"] == "awaiting_review" and detail["review"]["rows"] > 0
            resumed = await restarted.post(f"/runs/{run_id}/resume", json={"decision": "approve"})
            assert resumed.status_code == 202
            assert await wait_status(restarted, run_id, "ok") == "ok"
    finally:
        app2.state.manager.shutdown()

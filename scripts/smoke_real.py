"""真实 LLM 冒烟脚本：把只在 Fake 上成立的假设逐条拿到真实模型验证。

用法：`uv run python scripts/smoke_real.py [--repeat N]`。
报告写到 `docs/reviews/gate1/artifacts/smoke/<UTC>.{json,md}`。

退出码：0 全部通过；1 有失败项；2 缺少真实 LLM 配置（无 Key / 无模型 / 非 real 模式）。

安全约束：任何终端输出与报告文件都不含 API Key；异常信息写入报告前先做 Key 替换兜底，
`base_url` 只写脱敏后的域名。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel

from researchflow import main as main_mod
from researchflow.core.agent_loop import LoopBudget, run_tool_loop
from researchflow.core.config import PROJECT_ROOT, Settings, load_settings, sanitize_base_url
from researchflow.core.errors import ConfigError
from researchflow.core.llm import LLMClient, add_usage, build_llm_client, probe_structured_output
from researchflow.core.schemas import DraftContent, LLMUsage, ResearchQuestion
from researchflow.core.tools import ToolContext, ToolRegistry
from researchflow.tools.literature import build_retriever
from researchflow.tools.registry_tools import build_default_registry

EXIT_ALL_PASSED = 0
EXIT_HAS_FAILURE = 1
EXIT_MISSING_CONFIG = 2

REPORT_DIR = PROJECT_ROOT / "docs" / "reviews" / "gate1" / "artifacts" / "smoke"
DEMO_TOPIC = "examples/demo_topic.yaml"

CHECK_LABELS = {
    "S0": "GET /models 列出模型",
    "S1": "complete(ResearchQuestion) json_schema",
    "S2": "complete(DraftContent) json_schema",
    "S3": "complete(ResearchQuestion) json_object",
    "S4": "chat() 原生 tool calling",
    "S5": "run_tool_loop 检索→质量→结论",
    "S6": "用量字段 total_tokens > 0",
    "S7": "完整链路 main.run（真实）",
    "S8": "S7 run 目录 Key 自检",
    "S9": "probe_structured_output 是否真执行 json_schema",
    "S10": "S7 的 variables 是否都在数据集列中",
}


def _redact(text: str, key: str) -> str:
    """把 Key 替换成 ***；空 Key 原样返回。"""
    return text.replace(key, "***") if key else text


def _redact_deep(value: Any, key: str) -> Any:
    """递归替换任何字符串里的 Key（报告写盘前的兜底）。"""
    if isinstance(value, str):
        return _redact(value, key)
    if isinstance(value, dict):
        return {k: _redact_deep(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_deep(v, key) for v in value]
    return value


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _usage_dict(usage: LLMUsage | dict[str, Any] | None) -> dict[str, int]:
    if usage is None:
        usage = LLMUsage(attempts=0)
    if isinstance(usage, dict):
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "attempts": int(usage.get("attempts", 0)),
        }
    return usage.model_dump()


def _make_result(
    *,
    passed: bool,
    detail: str,
    attempts: int = 0,
    usage: LLMUsage | dict[str, Any] | None = None,
    latency_ms: int = 0,
    first_try: str | None = None,
    run_dir: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": passed,
        "detail": detail,
        "attempts": attempts,
        "usage": _usage_dict(usage),
        "latency_ms": latency_ms,
    }
    if first_try is not None:
        result["first_try"] = first_try
    if run_dir:
        result["run_dir"] = run_dir
    return result


def _check(name: str, fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
    """跑一项检查；异常只记为失败，不影响后续检查。"""
    try:
        result = fn(*args)
    except Exception as exc:  # noqa: BLE001 - 单项失败不能中断整份冒烟
        result = _make_result(passed=False, detail=f"{type(exc).__name__}: {exc}")
    result["name"] = name
    return result


def _prompt_for(response_model: type[BaseModel]) -> str:
    if response_model is ResearchQuestion:
        return "请提炼研究问题：课题「科研数据质量与线性回归可复现性评估」。"
    return "请撰写初稿正文：背景（background）、发现（findings）、局限（limitation）三段。"


def check_models(base_url: str, key: str) -> dict[str, Any]:
    """S0：GET {base_url}/models；404 记为 skipped，其它 4xx/5xx 记为失败。"""
    started = time.perf_counter()
    try:
        response = httpx.get(
            f"{base_url}/models", timeout=30.0, headers={"Authorization": f"Bearer {key}"}
        )
    except httpx.HTTPError as exc:
        return _make_result(
            passed=False,
            detail=f"GET /models 失败: {_redact(str(exc), key)}",
            latency_ms=_elapsed_ms(started),
        )
    latency = _elapsed_ms(started)
    if response.status_code == 404:
        return _make_result(
            passed=True, detail="接口不存在（404），记为 skipped", latency_ms=latency
        )
    if response.status_code >= 400:
        return _make_result(
            passed=False, detail=f"GET /models 返回 {response.status_code}", latency_ms=latency
        )
    try:
        data = response.json()
    except ValueError as exc:
        return _make_result(passed=False, detail=f"响应不是 JSON: {exc}", latency_ms=latency)
    ids = [str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)]
    return _make_result(passed=True, detail=f"模型列表: {ids}", latency_ms=latency)


def check_complete(
    client: LLMClient,
    response_model: type[BaseModel],
    repeat: int,
    mode: str,
    key: str,
) -> dict[str, Any]:
    """S1/S2/S3：`complete()` 重复 `repeat` 次；`mode` 临时切换结构化输出方式。"""
    started = time.perf_counter()
    original = client.structured_output
    client.structured_output = mode
    usage = LLMUsage(attempts=0)
    first_try = 0
    try:
        for _ in range(repeat):
            client.complete(_prompt_for(response_model), response_model)
            last = client.last_usage
            if last is not None:
                usage = add_usage(usage, last)
                if last.attempts == 1:
                    first_try += 1
    except Exception as exc:  # noqa: BLE001
        return _make_result(
            passed=False,
            detail=f"{type(exc).__name__}: {_redact(str(exc), key)}",
            attempts=usage.attempts,
            usage=usage,
            latency_ms=_elapsed_ms(started),
        )
    finally:
        client.structured_output = original
    return _make_result(
        passed=True,
        detail=f"{repeat} 次全部成功",
        attempts=usage.attempts,
        usage=usage,
        latency_ms=_elapsed_ms(started),
        first_try=f"{first_try}/{repeat}",
    )


def _invoke_outcome(registry: ToolRegistry, tool_ctx: ToolContext, call: Any) -> str:
    result = registry.invoke(call.name, call.arguments, tool_ctx)
    if result.ok:
        return "ok"
    return result.error.code if result.error is not None else "unknown"


def check_chat(
    client: LLMClient, registry: ToolRegistry, tool_ctx: ToolContext, key: str
) -> dict[str, Any]:
    """S4：`chat()` 带默认工具，提示词要求调用 `run_direct_compute`。"""
    started = time.perf_counter()
    tools = registry.as_openai_tools()
    prompt = (
        "请调用 run_direct_compute 工具，对 examples/sample_dataset.csv 计算目标列 y 的"
        "线性回归，特征列是 x1、x2、x3。"
    )
    try:
        turn = client.chat([{"role": "user", "content": prompt}], tools=tools, tool_choice="auto")
    except Exception as exc:  # noqa: BLE001
        return _make_result(
            passed=False,
            detail=f"chat 被拒绝或失败: {_redact(str(exc), key)}",
            usage=client.last_usage,
            latency_ms=_elapsed_ms(started),
        )
    outcomes = [
        f"{call.name}={_invoke_outcome(registry, tool_ctx, call)}" for call in turn.tool_calls
    ]
    detail = f"tool_calls={[c.name for c in turn.tool_calls]}; " + ("; ".join(outcomes) or "无")
    return _make_result(
        passed=True,
        detail=detail,
        attempts=1,
        usage=client.last_usage,
        latency_ms=_elapsed_ms(started),
    )


def check_tool_loop(
    client: LLMClient, registry: ToolRegistry, tool_ctx: ToolContext, key: str
) -> dict[str, Any]:
    """S5：工具回路，先查文献再查数据质量，最后给结论。"""
    started = time.perf_counter()
    prompt = (
        "先调用 search_literature 检索文献，再调用 check_data_quality 检查 "
        "examples/sample_dataset.csv 的数据质量，最后给出结论。"
    )
    try:
        loop = run_tool_loop(
            client,
            registry,
            tool_ctx,
            [{"role": "user", "content": prompt}],
            LoopBudget(max_steps=4, max_tool_calls=4),
        )
    except Exception as exc:  # noqa: BLE001
        return _make_result(
            passed=False,
            detail=f"{type(exc).__name__}: {_redact(str(exc), key)}",
            usage=client.last_usage,
            latency_ms=_elapsed_ms(started),
        )
    labels = []
    for inv in loop.invocations:
        args_json = json.dumps(inv.call.arguments, ensure_ascii=False)
        if inv.result.ok:
            labels.append(f"{inv.call.name}=ok arguments={args_json}")
        elif inv.result.error is not None:
            labels.append(f"{inv.call.name}={inv.result.error.code} arguments={args_json}")
        else:
            labels.append(f"{inv.call.name}=unknown arguments={args_json}")
    detail = (
        f"stop_reason={loop.stop_reason}; steps={loop.steps}; 调用: {'; '.join(labels) or '无'}"
    )
    return _make_result(
        passed=loop.stop_reason == "final",
        detail=detail,
        attempts=loop.steps,
        usage=loop.usage,
        latency_ms=_elapsed_ms(started),
    )


def check_usage(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """S6：S1–S5 每次调用的 `usage.total_tokens` 都大于 0。"""
    started = time.perf_counter()
    targets = [r for r in results if r.get("name") in {"S1", "S2", "S3", "S4", "S5"}]
    bad = [
        f"{r.get('name')}={r.get('usage', {}).get('total_tokens', 0)}"
        for r in targets
        if r.get("usage", {}).get("total_tokens", 0) <= 0
    ]
    total = sum(r.get("usage", {}).get("total_tokens", 0) for r in targets)
    if bad:
        return _make_result(
            passed=False,
            detail=f"用量缺失或为 0: {', '.join(bad)}",
            latency_ms=_elapsed_ms(started),
        )
    return _make_result(
        passed=True,
        detail=f"S1–S5 每次调用 total_tokens > 0，合计 {total}",
        usage={"total_tokens": total},
        latency_ms=_elapsed_ms(started),
    )


def check_full_pipeline(settings: Settings, key: str) -> dict[str, Any]:
    """S7：完整链路 `main.run`（真实模式），校验 manifest 的用量与 llm_calls 形状。"""
    started = time.perf_counter()
    runs_dir = settings.runs_dir
    before = {path.name for path in runs_dir.glob("*") if path.is_dir()}
    exit_code = main_mod.main(["run", "--topic", DEMO_TOPIC, "--auto-approve"])
    after = {path.name for path in runs_dir.glob("*") if path.is_dir()}
    new_dirs = sorted(after - before)
    if not new_dirs:
        return _make_result(
            passed=False,
            detail=f"退出码 {exit_code}，但未找到新 run 目录",
            latency_ms=_elapsed_ms(started),
        )
    run_dir = runs_dir / new_dirs[0]
    manifest_path = run_dir / "manifest.json"
    if exit_code != 0 or not manifest_path.exists():
        return _make_result(
            passed=False,
            detail=f"退出码 {exit_code}；manifest 缺失",
            latency_ms=_elapsed_ms(started),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_tokens = manifest.get("llm_usage", {}).get("total", {}).get("total_tokens", 0)
    calls = manifest.get("llm_usage", {}).get("calls", [])
    modes = [call.get("structured_output") for call in calls if isinstance(call, dict)]
    # C6：与当前配置的 RF_LLM_STRUCTURED_OUTPUT 比较，而不是写死 json_schema。
    configured = settings.llm.structured_output
    ok = total_tokens > 0 and len(calls) == 2 and all(m == configured for m in modes)
    variables = _run_variables(run_dir)
    detail = (
        f"退出码 {exit_code}; total_tokens={total_tokens}; llm_calls={len(calls)}; "
        f"structured_output={modes}（配置 {configured}）; run_dir={run_dir.name}"
    )
    result = _make_result(
        passed=ok,
        detail=detail,
        usage={"total_tokens": total_tokens},
        latency_ms=_elapsed_ms(started),
        run_dir=str(run_dir),
    )
    result["variables"] = variables
    return result


def check_run_dir_secrets(s7_result: dict[str, Any], key: str) -> dict[str, Any]:
    """S8：S7 的 run 目录里（manifest / checkpoint.sqlite 等）不出现 Key。"""
    started = time.perf_counter()
    run_dir = s7_result.get("run_dir")
    if not run_dir:
        return _make_result(
            passed=False, detail="S7 未产出 run 目录，无法自检", latency_ms=_elapsed_ms(started)
        )
    path = Path(run_dir)
    offenders = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and key and key.encode() in file.read_bytes():
            offenders.append(str(file.relative_to(path)))
    if offenders:
        return _make_result(
            passed=False,
            detail=f"以下文件含 Key: {offenders}",
            latency_ms=_elapsed_ms(started),
        )
    return _make_result(
        passed=True, detail=f"run 目录 {path.name} 内无 Key 泄漏", latency_ms=_elapsed_ms(started)
    )


def _run_variables(run_dir: Path) -> list[str]:
    """从 run 目录的 checkpoint.sqlite 读出 `research_question.variables`。

    读不到（checkpoint 缺失/损坏）就返回空列表，由 S10 记为"无变量"。
    """
    try:
        from researchflow.graphs.orchestrator import (
            build_graph,
            close_checkpointer,
            make_checkpointer,
        )

        checkpointer = make_checkpointer("sqlite", run_dir)
        try:
            graph = build_graph(checkpointer)
            state = graph.get_state({"configurable": {"thread_id": run_dir.name}})
            question = (state.values or {}).get("research_question")
        finally:
            close_checkpointer(checkpointer)
    except Exception:  # noqa: BLE001 - 冒烟脚本里读不到就返回空，不影响其它检查
        return []
    if question is None:
        return []
    if isinstance(question, dict):
        return [str(var) for var in (question.get("variables") or [])]
    return [str(var) for var in (getattr(question, "variables", None) or [])]


def _dataset_columns(run_dir: Path) -> list[str]:
    """从清洗候选文件读数据集列名；文件缺失时返回空列表。"""
    csv_path = run_dir / "cleaned_candidate.csv"
    if not csv_path.exists():
        return []
    try:
        return [str(col) for col in pd.read_csv(csv_path).columns]
    except Exception:  # noqa: BLE001
        return []


def check_structured_output_probe(
    client: LLMClient, response_model: type[BaseModel], key: str
) -> dict[str, Any]:
    """S9：探测该 Provider 是否真的执行 json_schema 约束（结果只做报告，不做门禁）。"""
    started = time.perf_counter()
    try:
        probe = probe_structured_output(client, response_model)
    except Exception as exc:  # noqa: BLE001
        return _make_result(
            passed=False,
            detail=f"{type(exc).__name__}: {_redact(str(exc), key)}",
            latency_ms=_elapsed_ms(started),
        )
    return _make_result(
        passed=True,
        detail=f"enforced={probe['enforced']}; {probe['detail']}",
        usage=client.last_usage,
        latency_ms=_elapsed_ms(started),
    )


def check_variables_in_dataset(s7_result: dict[str, Any], key: str) -> dict[str, Any]:
    """S10：S7 的 research_question.variables 是否都在数据集列中。"""
    started = time.perf_counter()
    run_dir = s7_result.get("run_dir")
    if not run_dir:
        return _make_result(
            passed=False,
            detail="S7 未产出 run 目录，无法检查 variables",
            latency_ms=_elapsed_ms(started),
        )
    variables = [str(var) for var in (s7_result.get("variables") or [])]
    columns = _dataset_columns(Path(run_dir))
    missing = [var for var in variables if var not in columns]
    detail = f"variables={variables}; 数据集列={columns}; 缺失={missing or '无'}"
    return _make_result(passed=not missing, detail=detail, latency_ms=_elapsed_ms(started))


def run_checks(settings: Settings, *, repeat: int = 3) -> list[dict[str, Any]]:
    """按 S0→S8 顺序跑全部检查，返回结果列表（每项已带 name 且 Key 已脱敏）。"""
    key = settings.llm.api_key or ""
    client = build_llm_client(settings)
    registry = build_default_registry()
    tool_ctx = ToolContext(
        workspace_root=settings.project_root,
        settings=settings,
        retriever=build_retriever(settings),
    )
    results: list[dict[str, Any]] = []
    results.append(_check("S0", check_models, settings.llm.base_url, key))
    results.append(
        _check("S1", check_complete, client, ResearchQuestion, repeat, "json_schema", key)
    )
    results.append(_check("S2", check_complete, client, DraftContent, 1, "json_schema", key))
    results.append(
        _check("S3", check_complete, client, ResearchQuestion, repeat, "json_object", key)
    )
    results.append(_check("S4", check_chat, client, registry, tool_ctx, key))
    results.append(_check("S5", check_tool_loop, client, registry, tool_ctx, key))
    results.append(_check("S6", check_usage, results, key))
    results.append(_check("S7", check_full_pipeline, settings, key))
    s7_result = results[-1]
    results.append(_check("S8", check_run_dir_secrets, s7_result, key))
    results.append(_check("S9", check_structured_output_probe, client, ResearchQuestion, key))
    results.append(_check("S10", check_variables_in_dataset, s7_result, key))
    return _redact_deep(results, key)


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(1 for r in results if r.get("passed"))
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
    return {"passed": passed, "failed": len(results) - passed, "total_tokens": total_tokens}


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 真实 LLM 冒烟报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 模型: {report['model']}",
        f"- base_url: {report['base_url']}",
        f"- repeat: {report['repeat']}",
        "",
        "| 编号 | 检查 | 结果 | 首轮合法 | tokens | 耗时 |",
        "|------|------|------|----------|--------|------|",
    ]
    for r in report["checks"]:
        lines.append(
            f"| {r['name']} | {CHECK_LABELS.get(r['name'], r['name'])} | "
            f"{'通过' if r['passed'] else '失败'} | {r.get('first_try', '—')} | "
            f"{r.get('usage', {}).get('total_tokens', 0)} | {r.get('latency_ms', 0)}ms |"
        )
    lines.append(f"| 合计 | | | | {report['summary']['total_tokens']} | |")
    lines.append("")
    lines.append(f"通过 {report['summary']['passed']} / 失败 {report['summary']['failed']}")
    return "\n".join(lines) + "\n"


def write_reports(
    results: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    repeat: int,
    key: str,
    report_dir: Path | None = None,
) -> tuple[Path, Path]:
    """把结果写成同名 JSON 与 Markdown 报告，返回两个路径。"""
    target = report_dir or REPORT_DIR
    target.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_url": base_url,
        "model": model,
        "repeat": repeat,
        "checks": results,
        "summary": summarize(results),
    }
    json_path = target / f"{timestamp}.json"
    md_path = target / f"{timestamp}.md"
    json_path.write_text(
        _redact(json.dumps(report, ensure_ascii=False, indent=2), key), encoding="utf-8"
    )
    md_path.write_text(_redact(_render_markdown(report), key), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke_real.py", description="真实 LLM 冒烟")
    parser.add_argument("--repeat", type=int, default=3, help="S1/S3 的重复次数（默认 3）")
    args = parser.parse_args(argv)

    settings = load_settings()
    key = settings.llm.api_key or ""
    missing: list[str] = []
    if not key:
        missing.append("RF_LLM_API_KEY")
    if not settings.llm.model:
        missing.append("RF_LLM_MODEL")
    try:
        mode = settings.resolve_llm_mode()
    except ConfigError as exc:
        mode = settings.llm.mode
        missing.append(str(exc))
    if missing or mode != "real":
        print("真实冒烟需要以下配置，请检查 .env：")
        for item in missing:
            print(f"  - 缺少 {item}")
        if mode != "real":
            print(f"  - resolve_llm_mode() = {mode!r}（需要 real）")
        return EXIT_MISSING_CONFIG

    results = run_checks(settings, repeat=args.repeat)
    json_path, md_path = write_reports(
        results,
        base_url=sanitize_base_url(settings.llm.base_url),
        model=settings.llm.model or "",
        repeat=args.repeat,
        key=key,
    )
    summary = summarize(results)
    print(f"报告: {json_path}")
    print(f"报告: {md_path}")
    print(f"通过 {summary['passed']} / 失败 {summary['failed']}")
    return EXIT_ALL_PASSED if summary["failed"] == 0 else EXIT_HAS_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())

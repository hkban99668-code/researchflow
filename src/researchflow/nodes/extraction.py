"""② 问题提炼：Prompt + LLM（Fake/Real 共用同一 Schema）。"""

from __future__ import annotations

from langgraph.runtime import Runtime

from ..core.config import RunContext
from ..core.schemas import LLMCallRecord, ResearchQuestion
from ..core.step import step


def _render_prompt(template: str, topic: str, refs: list, columns: list[str]) -> str:
    literature = (
        "\n".join(f"- {ref.title}（{ref.year}）：{ref.summary}" for ref in refs) or "（无检索结果）"
    )
    columns_text = "、".join(columns) or "（未提供）"
    return (
        template.replace("{{topic}}", topic)
        .replace("{{literature}}", literature)
        .replace("{{columns}}", columns_text)
    )


@step("extraction")
def extraction_node(state: dict, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    params = state["param_set"].params

    template = ctx.settings.resolve_path(
        ctx.settings.raw.get("paths", {}).get(
            "extract_question_prompt", "configs/prompts/extract_question.md"
        )
    ).read_text(encoding="utf-8")

    columns = [*list(params.get("features") or []), params["target"]]
    prompt = _render_prompt(template, state["topic"], state.get("literature_refs") or [], columns)
    question = ctx.llm.complete(
        prompt,
        ResearchQuestion,
        {
            "topic": state["topic"],
            "target": params.get("target", "target"),
            "features": list(params.get("features") or []),
        },
    )

    usage = ctx.llm.last_usage
    llm_calls = (
        [
            LLMCallRecord(
                step="extraction",
                response_model=ResearchQuestion.__name__,
                usage=usage,
                structured_output=ctx.llm.structured_output,
                duration_ms=getattr(ctx.llm, "last_duration_ms", None),
            )
        ]
        if usage is not None
        else []
    )

    return {
        "research_question": question,
        "llm_calls": llm_calls,
        "step_status": {"extraction": "ok"},
    }

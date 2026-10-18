"""⑦ 硬规则校验：对指标、章节、产物与占位符做判定。"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from ..core.config import RunContext, sha256_file
from ..core.schemas import ArtifactRef, DraftContent, LiteratureRef
from ..core.step import current_attempt, step
from ..tools.chunking import chunk_ref
from ..tools.rules_engine import SubmissionRules, evaluate_submission, load_rules


def _citations_of(state: dict) -> list[Any]:
    """取 ⑥ 写进 state 的 `draft_content.citations`（兼容 checkpoint 反序列化后的字典形态）。"""
    content = state.get("draft_content")
    if content is None:
        return []
    if not isinstance(content, DraftContent):
        content = DraftContent.model_validate(content)
    return list(content.citations)


def _valid_ref_ids_of(state: dict) -> list[str]:
    """取本次 ① 检索到的全部 `ref_id`（字典形态的 LiteratureRef 同样兼容）。"""
    ids: list[str] = []
    for ref in state.get("literature_refs") or []:
        if isinstance(ref, LiteratureRef):
            ids.append(ref.ref_id)
        else:
            ids.append(str(LiteratureRef.model_validate(ref).ref_id))
    return ids


def _valid_chunk_ids_of(state: dict) -> set[str]:
    """本次索引块集合（T-030 块级引用）：对检索到的文献重新切块得到。

    切块是确定性的，与写作时用的同一函数，因此块 ID 一致；不带 chunk_id 的
    旧数据不走集合校验，行为不变。
    """
    chunk_ids: set[str] = set()
    for ref in state.get("literature_refs") or []:
        literature = ref if isinstance(ref, LiteratureRef) else LiteratureRef.model_validate(ref)
        chunk_ids.update(chunk.chunk_id for chunk in chunk_ref(literature))
    return chunk_ids


@step("validation")
def validation_node(state: dict, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    rules_path = ctx.settings.resolve_path(
        ctx.settings.raw.get("paths", {}).get(
            "submission_checks", "configs/rules/submission_checks.yaml"
        )
    )
    rules = load_rules(rules_path, model=SubmissionRules)

    draft_text = ctx.artifact_path(state["draft_path"]).read_text(encoding="utf-8")
    # ⑦ 的 required_artifacts 检查看 dict 的 key（P1-3）：key 就是 kind
    artifacts = {kind: True for kind in (state.get("artifacts") or {})}

    report = evaluate_submission(
        rules,
        metrics=state["compute_result"].metrics,
        draft_text=draft_text,
        artifacts=artifacts,
        row_count=state["cleaned_data"].rows,
        variables=state["research_question"].variables,
        dataset_columns=state["cleaned_data"].columns,
        # T-029 引用硬规则：citations 来自 ⑥ 的 draft_content，合法集合来自 ① 检索结果。
        # T-030 块级引用：chunk_id 非空时还校验块属于所引文献且在索引块集合里。
        citations=_citations_of(state),
        valid_ref_ids=_valid_ref_ids_of(state),
        valid_chunk_ids=_valid_chunk_ids_of(state),
    )

    path = ctx.artifact_path("validation_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    checksum = sha256_file(path)

    return {
        "validation_result": report,
        "artifacts": {
            "validation_report": ArtifactRef(
                kind="validation_report",
                path=ctx.relative_to_run(path),
                checksum=checksum,
                produced_by="validation",
                attempt=current_attempt(state),
            )
        },
        "step_status": {"validation": "ok" if report.passed else "failed"},
    }

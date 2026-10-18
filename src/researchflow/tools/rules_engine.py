"""⑦ 硬规则引擎：加载 configs/rules/*.yaml 并对产物做判定。

规则文件缺失或缺必填段时抛 ConfigError（RF-001），避免"规则没读到却判通过"的静默假阳性。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..core.errors import ConfigError
from ..core.schemas import CheckResult, CitationStats, ValidationReport


class AcceptanceRules(BaseModel):
    """③ 数据准入规则的最小 schema。"""

    required_columns: list[str]
    max_missing_rate: float = 0.0
    max_dropped_ratio: float = 1.0
    min_rows: int = 1
    ranges: dict[str, dict[str, float]] = Field(default_factory=dict)


class SubmissionRules(BaseModel):
    """⑦ 提交前硬规则的最小 schema。"""

    metric_thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
    required_sections: list[str]
    min_draft_chars: int = 0
    required_artifacts: list[str]
    forbidden_tokens: list[str] = Field(default_factory=list)
    min_rows: int | None = None
    # 每条检查归哪个节点负责（P1-8）：键是完整检查名（如 artifact:chart）或
    # 冒号前的前缀（如 metric），值是图里的节点名。领域知识写在这里，不写代码。
    owner_steps: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _every_check_has_owner(self) -> SubmissionRules:
        """规则会产生的每一类检查都必须能在 `owner_steps` 里找到负责节点。"""
        missing = [
            name
            for name in _produced_check_names(self)
            if resolve_owner(name, self.owner_steps) is None
        ]
        if missing:
            raise ValueError("owner_steps 缺少以下检查的负责节点：" + "、".join(sorted(missing)))
        return self


def resolve_owner(check_name: str, owner_steps: dict[str, str]) -> str | None:
    """按完整检查名查负责节点；找不到再按冒号前的前缀查（如 `metric`）。"""
    if check_name in owner_steps:
        return owner_steps[check_name]
    return owner_steps.get(check_name.split(":", 1)[0])


def _produced_check_names(rules: SubmissionRules) -> list[str]:
    """规则会产生的全部检查名清单，用于 load_rules 时校验 owner 是否配齐。"""
    names = [f"metric:{name}" for name in rules.metric_thresholds]
    names += [f"section:{section}" for section in rules.required_sections]
    names.append("draft_min_chars")
    names += [f"artifact:{artifact}" for artifact in rules.required_artifacts]
    names.append("forbidden_tokens")
    if rules.min_rows is not None:
        names.append("min_rows")
    names.append("variables_in_dataset")
    # 有据写作的引用硬规则（T-029）：两条都会产生，必须配 owner。
    names.append("citations_subset")
    names.append("citation_has_claim")
    return names


def _owner_of(name: str, rules: SubmissionRules) -> str:
    """取一条检查的负责节点；校验已保证存在，这里兜底防直接调 evaluate_submission。"""
    owner = resolve_owner(name, rules.owner_steps)
    if owner is None:
        raise ConfigError(f"检查 {name!r} 缺少 owner_step")
    return owner


def coerce_rules(rules: Any, model: type[BaseModel], *, source: str = "") -> Any:
    """把字典规则校验为模型；缺必填段或类型不符时抛 ConfigError。"""
    if isinstance(rules, model):
        return rules
    try:
        return model.model_validate(rules)
    except ValidationError as exc:
        origin = f"（{source}）" if source else ""
        raise ConfigError(f"规则内容不合法{origin}：{str(exc)[:300]}") from exc


def load_rules(
    path: str | Path,
    *,
    model: type[BaseModel] = AcceptanceRules,
    required: bool = True,
) -> Any:
    """加载并校验规则文件；文件不存在时默认抛 ConfigError。"""
    path = Path(path)
    if not path.exists():
        if required:
            raise ConfigError(f"缺少规则文件: {path}")
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return coerce_rules(raw, model, source=str(path))


def _citation_ref_id(citation: Any) -> str:
    """从 `Citation` 或旧 checkpoint 的字典形态取 `ref_id`；缺字段返回空串。"""
    if isinstance(citation, dict):
        return str(citation.get("ref_id") or "")
    return str(getattr(citation, "ref_id", "") or "")


def _citation_claim(citation: Any) -> str:
    """从 `Citation` 或字典形态取 `claim`。"""
    if isinstance(citation, dict):
        return str(citation.get("claim") or "")
    return str(getattr(citation, "claim", "") or "")


def _citation_chunk_id(citation: Any) -> str:
    """从 `Citation` 或字典形态取 `chunk_id`；缺字段返回空串（旧数据行为不变）。"""
    if isinstance(citation, dict):
        return str(citation.get("chunk_id") or "")
    return str(getattr(citation, "chunk_id", "") or "")


def compute_citation_stats(citations: list[Any], valid_ref_ids: list[str] | None) -> CitationStats:
    """算**校验前**的幻觉引用率：`invalid / total`，`total == 0` 时为 0.0（不除零）。"""
    valid = set(valid_ref_ids or [])
    total = len(citations)
    invalid = sum(1 for citation in citations if _citation_ref_id(citation) not in valid)
    rate = round(invalid / total, 4) if total else 0.0
    return CitationStats(total=total, invalid=invalid, hallucination_rate=rate)


def evaluate_submission(
    rules: Any,
    *,
    metrics: dict,
    draft_text: str,
    artifacts: dict[str, bool],
    row_count: int = 0,
    variables: list[str] | None = None,
    dataset_columns: list[str] | None = None,
    citations: list[Any] | None = None,
    valid_ref_ids: list[str] | None = None,
    valid_chunk_ids: list[str] | set[str] | None = None,
) -> ValidationReport:
    """对指标、章节、产物、占位符、样本量、变量来源与引用做硬规则判定。

    `valid_chunk_ids` 是本次索引块集合（T-030 块级引用）：引用带 `chunk_id` 时，
    该块必须以 `{ref_id}#c` 开头（属于所引文献）且存在于集合里，否则
    `citations_subset` 失败；不带 `chunk_id` 的旧数据行为不变。
    """
    parsed: SubmissionRules = coerce_rules(rules, SubmissionRules)
    checks: list[CheckResult] = []

    for name, rule in parsed.metric_thresholds.items():
        check_name = f"metric:{name}"
        value = metrics.get(name)
        if value is None:
            checks.append(
                CheckResult(
                    name=check_name,
                    passed=False,
                    detail="缺少指标",
                    owner_step=_owner_of(check_name, parsed),
                )
            )
            continue
        minimum = (rule or {}).get("min")
        passed = minimum is None or value >= float(minimum)
        detail = f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
        if not passed:
            detail = f"{detail} < {minimum}"
        checks.append(
            CheckResult(
                name=check_name,
                passed=passed,
                detail=detail,
                owner_step=_owner_of(check_name, parsed),
            )
        )

    for section in parsed.required_sections:
        check_name = f"section:{section}"
        passed = section in draft_text
        checks.append(
            CheckResult(
                name=check_name,
                passed=passed,
                detail="已包含" if passed else "缺少章节",
                owner_step=_owner_of(check_name, parsed),
            )
        )

    min_chars = parsed.min_draft_chars
    checks.append(
        CheckResult(
            name="draft_min_chars",
            passed=len(draft_text) >= min_chars,
            detail=f"{len(draft_text)} / {min_chars}",
            owner_step=_owner_of("draft_min_chars", parsed),
        )
    )

    for artifact in parsed.required_artifacts:
        check_name = f"artifact:{artifact}"
        passed = bool(artifacts.get(artifact))
        checks.append(
            CheckResult(
                name=check_name,
                passed=passed,
                detail="存在" if passed else "缺失",
                owner_step=_owner_of(check_name, parsed),
            )
        )

    hit_tokens = [token for token in parsed.forbidden_tokens if token in draft_text]
    checks.append(
        CheckResult(
            name="forbidden_tokens",
            passed=not hit_tokens,
            detail="命中：" + "、".join(hit_tokens) if hit_tokens else "无",
            owner_step=_owner_of("forbidden_tokens", parsed),
        )
    )

    min_rows = parsed.min_rows
    if min_rows is not None:
        passed = row_count >= int(min_rows)
        checks.append(
            CheckResult(
                name="min_rows",
                passed=passed,
                detail=f"{row_count} / {min_rows}",
                owner_step=_owner_of("min_rows", parsed),
            )
        )

    # C2：variables 必须都来自数据集列，模型编造的变量在这里被拦下。
    columns = set(dataset_columns or [])
    missing = [var for var in (variables or []) if var not in columns]
    checks.append(
        CheckResult(
            name="variables_in_dataset",
            passed=not missing,
            detail=("不存在：" + "、".join(missing)) if missing else "无",
            owner_step=_owner_of("variables_in_dataset", parsed),
        )
    )

    # T-029 引用硬规则：模型只能引用本次 retrieval 返回过的 ref_id（D-014
    # "模型输出必须由程序校验"），claim 不能为空。统计先于判定算好（校验前的
    # 幻觉引用率），随报告带出，进 validation_report.json 与 manifest（A6）。
    citation_list = list(citations or [])
    stats = compute_citation_stats(citation_list, valid_ref_ids)
    valid_ids = set(valid_ref_ids or [])
    # 失败信息逐条列出池外 ref_id，让草稿作者能直接定位是哪条引用不合法。
    outside_ids = [
        _citation_ref_id(citation) or "（空 ref_id）"
        for citation in citation_list
        if _citation_ref_id(citation) not in valid_ids
    ]
    # T-030 块级引用：chunk_id 非空时还必须属于所引文献且在本次索引块集合里。
    # 没传 valid_chunk_ids 视为空集合：带 chunk_id 的引用一律无法通过（宁可拦下
    # 也不能让"声称引用了某一块"绕过程序校验，D-014）。
    chunk_set = set(valid_chunk_ids or [])
    bad_chunks: list[str] = []
    for citation in citation_list:
        chunk_id = _citation_chunk_id(citation)
        if not chunk_id:
            continue
        ref_id = _citation_ref_id(citation)
        if not chunk_id.startswith(f"{ref_id}#c"):
            bad_chunks.append(f"{chunk_id}（不属于所引文献 {ref_id}）")
        elif chunk_id not in chunk_set:
            bad_chunks.append(f"{chunk_id}（不在本次索引块集合中）")
    checks.append(
        CheckResult(
            name="citations_subset",
            passed=not outside_ids and not bad_chunks,
            detail="；".join(
                part
                for part in (
                    (
                        "只能引用 retrieval 返回过的文献，池外 ref_id：" + "、".join(outside_ids)
                        if outside_ids
                        else ""
                    ),
                    ("块级引用不合法：" + "、".join(bad_chunks) if bad_chunks else ""),
                )
                if part
            )
            or f"{stats.total} 条引用全部在检索结果内",
            owner_step=_owner_of("citations_subset", parsed),
        )
    )
    empty_claims = [
        index
        for index, citation in enumerate(citation_list, start=1)
        if not _citation_claim(citation).strip()
    ]
    checks.append(
        CheckResult(
            name="citation_has_claim",
            passed=not empty_claims,
            detail=(
                "claim 不能为空，第 " + "、".join(str(i) for i in empty_claims) + " 条"
                if empty_claims
                else "每条引用都有 claim"
            ),
            owner_step=_owner_of("citation_has_claim", parsed),
        )
    )

    passed = all(check.passed for check in checks)
    failed = [check.name for check in checks if not check.passed]
    summary = "全部硬规则通过" if passed else "未通过：" + "、".join(failed)

    return ValidationReport(passed=passed, checks=checks, summary=summary, citation_stats=stats)

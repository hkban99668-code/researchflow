"""③ 数据质量：Pandera 预检。

预检为纯函数，保证 interrupt 恢复重跑时幂等。
checksum 计算在 `core/config.py::sha256_file`（core 不能反向依赖 tools）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa

from ..core.errors import DataValidationError
from ..core.schemas import QualitySummary
from .clean_strategies import row_missing_rate
from .rules_engine import AcceptanceRules, coerce_rules


def _build_schema(required_columns: list[str], ranges: dict) -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {}
    for column in required_columns:
        checks = []
        rule = ranges.get(column) or {}
        if "min" in rule:
            checks.append(pa.Check.greater_than_or_equal_to(float(rule["min"])))
        if "max" in rule:
            checks.append(pa.Check.less_than_or_equal_to(float(rule["max"])))
        columns[column] = pa.Column(float, checks=checks, nullable=True)
    return pa.DataFrameSchema(columns, strict=False)


def precheck(
    frame: pd.DataFrame,
    rules: Any,
    *,
    duplicates_removed: int = 0,
    raw_missing_rate: float | None = None,
    dropped_rows: int = 0,
    rows_before: int | None = None,
) -> QualitySummary:
    """按 configs/rules/data_acceptance.yaml 做硬阈值预检。

    任一硬规则不满足即抛 DataValidationError（RF-003）。
    缺失率必须在**删除缺失行之前**的数据上判定：传入 `raw_missing_rate`
    （由 `clean_frame` 给出）时按原始口径判定，否则退化为在传入 frame 上计算。
    """
    parsed: AcceptanceRules = coerce_rules(rules, AcceptanceRules)
    required = list(parsed.required_columns)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(f"缺少必需列: {', '.join(missing_columns)}")

    min_rows = parsed.min_rows
    if len(frame) < min_rows:
        raise DataValidationError(f"样本行数不足: {len(frame)} < {min_rows}")

    if raw_missing_rate is None:
        label, rate = "缺失率", row_missing_rate(frame, required)
    else:
        label, rate = "原始缺失率", float(raw_missing_rate)
    if rate > parsed.max_missing_rate:
        raise DataValidationError(f"{label}超标: {rate:.4f} > {parsed.max_missing_rate:.4f}")

    if rows_before:
        dropped_ratio = dropped_rows / rows_before
        if dropped_ratio > parsed.max_dropped_ratio:
            raise DataValidationError(
                f"丢弃行比例超标: {dropped_ratio:.4f} > {parsed.max_dropped_ratio:.4f}"
            )

    subset = frame[required]
    missing_rate = row_missing_rate(frame, required)
    schema = _build_schema(required, parsed.ranges)
    try:
        schema.validate(subset.astype(float), lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataValidationError(
            "数据未通过 Pandera 预检", detail={"errors": str(exc)[:500]}
        ) from exc

    notes = []
    if duplicates_removed:
        notes.append(f"移除重复行 {duplicates_removed} 行")
    notes.append(f"预检通过：{len(frame)} 行 / {len(frame.columns)} 列")

    return QualitySummary(
        rows=len(frame),
        columns=len(frame.columns),
        missing_rate=missing_rate,
        duplicates_removed=duplicates_removed,
        # 清洗前的口径随摘要一起带走：③ 拆成 prepare/review 后，审核节点只能从
        # `cleaned_candidate.quality` 取这两个数，审核载荷才不用重跑清洗。
        raw_missing_rate=float(raw_missing_rate) if raw_missing_rate is not None else rate,
        dropped_rows=dropped_rows,
        notes=notes,
    )

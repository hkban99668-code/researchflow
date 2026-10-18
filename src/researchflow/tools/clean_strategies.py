"""③ 清洗策略：纯函数，便于单测与 interrupt 恢复时重跑。

顺序：数值化 → 去重 → 丢弃关键列缺失的行。
去重必须在数值化之后，否则 "1" 与 "1.0" 不会被判为重复。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def row_missing_rate(frame: pd.DataFrame, columns: list[str]) -> float:
    """含缺失值的行占比（只统计关键列）。

    与逐单元格缺失率相比，行级口径更贴近"丢了多少样本"，
    也更能反映审核人关心的信息量损失。
    """
    present = [column for column in columns if column in frame.columns]
    if not len(frame) or not present:
        return 0.0
    return float(frame[present].isna().any(axis=1).to_numpy().mean())


def clean_frame(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    target: str | None = None,
    y_outlier_strategy: Literal["none", "iqr"] = "none",
    y_outlier_k: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """数值化 → 去重 → 丢弃关键列缺失的行 →（可选）裁剪 y 离群值。

    返回 (清洗后数据框, 统计字典)。统计中包含删除前的原始缺失率，
    缺失率阈值必须在原始数据上判定，否则 dropna 之后恒为 0。

    `y_outlier_strategy="iqr"` 时，在 dropna 之后去掉 `target` 列超出
    `[Q1 − k·IQR, Q3 + k·IQR]` 的行，统计里加 `y_outliers_removed`。
    默认值（`none`）下行为与之前完全一致，保证 T-006 可复现与 T-011 基线不变。
    """
    working = frame.copy()
    rows_before = len(working)
    present = [column for column in columns if column in working.columns]

    raw_missing_rate = row_missing_rate(working, columns)

    for column in present:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working = working.drop_duplicates()
    duplicates_removed = rows_before - len(working)

    rows_before_drop = len(working)
    working = working.dropna(subset=present)
    dropped_rows = rows_before_drop - len(working)

    y_outliers_removed = 0
    if y_outlier_strategy == "iqr":
        if target is None:
            raise ValueError("y_outlier_strategy='iqr' 时必须提供 target 列名")
        if target not in working.columns:
            raise ValueError(f"y 离群值裁剪的 target 列不存在: {target!r}")
        q1 = working[target].quantile(0.25)
        q3 = working[target].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - y_outlier_k * iqr
        upper = q3 + y_outlier_k * iqr
        within = working[target].between(lower, upper)
        y_outliers_removed = int((~within).sum())
        working = working[within]

    missing_rate = row_missing_rate(working, columns)
    dropped_ratio = float(dropped_rows / rows_before) if rows_before else 0.0

    notes: list[str] = []
    if duplicates_removed:
        notes.append(f"移除重复行 {duplicates_removed} 行")
    if dropped_rows:
        notes.append(f"移除含缺失值的行 {dropped_rows} 行")
    if y_outliers_removed:
        notes.append(f"按 IQR（k={y_outlier_k}）移除 {target} 列离群值 {y_outliers_removed} 行")

    stats = {
        "rows_before": rows_before,
        "raw_missing_rate": raw_missing_rate,
        "duplicates_removed": duplicates_removed,
        "dropped_rows": dropped_rows,
        "dropped_ratio": dropped_ratio,
        "missing_rate": missing_rate,
        "y_outliers_removed": y_outliers_removed,
        "notes": notes,
    }
    return working.reset_index(drop=True), stats

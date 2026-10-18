"""标签核验：用清洗函数和计算函数实际算一遍，核验"能不能修"的标签。

核验是**检查**标签，不是**生成**标签（T-018）。标签仍然按故障的定义手写在
`cases.yaml` 里，这里只负责发现写错的地方。对 `cross_step` 与 `human` 组的每条
用例：生成数据 → 在白名单内枚举"最多 2 次修法的组合"（每次修法从 iqr k=1.5、
iqr k=3、log 变换、排除每个单独特征中选一个）→ 用 `clean_frame` 清洗、`run_direct`
计算测试集 r2，同时检查清洗后行数是否 ≥ `min_rows`。记录能让 r2 ≥ 阈值且行数达标
的**最短**组合。

阈值与 `min_rows` 直接从 `configs/rules/*.yaml` 读，与真实链路保持一致，不在这里
复制魔法数字。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import yaml

from researchflow.compute.direct import run_direct
from researchflow.core.errors import ComputeError
from researchflow.tools.clean_strategies import clean_frame

from . import faults

RULES_DIR = faults.PROJECT_ROOT / "configs" / "rules"

# 课题默认列：与 examples/demo_topic.yaml 一致，也与 eval 运行器 `_materialize` 一致。
FEATURES = ["x1", "x2", "x3"]
TARGET = "y"

# 每种"修法"的名字 → 应用方式。iqr 属于 data_prepare（清洗），log / exclude 属于 compute。
# 键名同时用于核验表与 expected_fix 的对应（见 `expected_fix_op`）。
FIX_OPS: dict[str, dict[str, Any]] = {
    "iqr_k1_5": {"clean": ("iqr", 1.5)},
    "iqr_k3": {"clean": ("iqr", 3.0)},
    "log": {"target_transform": "log"},
    "exclude_x1": {"exclude": ["x1"]},
    "exclude_x2": {"exclude": ["x2"]},
    "exclude_x3": {"exclude": ["x3"]},
}


def _constants() -> tuple[int, float]:
    """从规则文件读 (min_rows, r2 阈值)。"""
    acceptance = yaml.safe_load((RULES_DIR / "data_acceptance.yaml").read_text(encoding="utf-8"))
    submission = yaml.safe_load((RULES_DIR / "submission_checks.yaml").read_text(encoding="utf-8"))
    return int(acceptance["min_rows"]), float(submission["metric_thresholds"]["r2"]["min"])


def _apply(frame: Any, ops: list[str]) -> tuple[float | None, int]:
    """按一组修法名清洗并计算 (r2, 行数)；log 遇 y ≤ 0 抛 RF-004，r2 记为 None。"""
    strategy: str = "none"
    k: float = 3.0
    target_transform = "none"
    exclude: list[str] = []
    for op in ops:
        spec = FIX_OPS[op]
        if "clean" in spec:
            strategy, k = spec["clean"]
        if "target_transform" in spec:
            target_transform = spec["target_transform"]
        if "exclude" in spec:
            exclude.extend(spec["exclude"])

    cleaned, _stats = clean_frame(
        frame,
        columns=[*FEATURES, TARGET],
        target=TARGET,
        y_outlier_strategy=strategy,  # type: ignore[arg-type]
        y_outlier_k=k,
    )
    try:
        result = run_direct(
            cleaned,
            target=TARGET,
            features=FEATURES,
            params={"test_size": 0.25},
            seed=42,
            algorithm="linear_regression",
            target_transform=target_transform,  # type: ignore[arg-type]
            exclude_features=exclude,
        )
        r2 = float(result.metrics["r2"])
    except ComputeError:
        r2 = None
    return r2, len(cleaned)


def expected_fix_op(expected_fix: dict | None) -> str | None:
    """把 `cases.yaml` 里的 `expected_fix` 映射成 `FIX_OPS` 里的修法名。

    也用于把一次归因决策（含 `patch` 键）映射成修法名：patch 经 Pydantic
    `model_dump()` 后所有字段都在（值为 None），所以这里一律用 `.get()` 判非空，
    不能用 `in` 判键存在。
    """
    if not expected_fix:
        return None
    patch = expected_fix.get("patch") or {}
    if patch.get("target_transform") == "log":
        return "log"
    if patch.get("exclude_features"):
        names = list(patch["exclude_features"])
        return f"exclude_{names[0]}" if len(names) == 1 else None
    if patch.get("y_outlier_strategy") == "iqr":
        k = patch.get("y_outlier_k")
        if k is not None and abs(float(k) - 1.5) < 1e-9:
            return "iqr_k1_5"
        if k is not None and abs(float(k) - 3.0) < 1e-9:
            return "iqr_k3"
    return None


def check_fixable(case: dict) -> dict:
    """核验一条 cross_step / human 用例的标签，返回各修法下的 r2、行数与最短可行组合。"""
    frame = faults.load_sample()
    mutated = faults.apply_fault(
        case["fault"], frame, int(case["seed"]), dict(case.get("params") or {})
    )

    min_rows, threshold = _constants()

    def _fixed(r2: float | None, rows: int) -> bool:
        return r2 is not None and r2 >= threshold and rows >= min_rows

    # 单个修法与"不处理"的 r2 / 行数（用于核验表）。
    r2: dict[str, float | None] = {}
    rows: dict[str, int] = {}
    r2["none"], rows["none"] = _apply(mutated, [])
    for op in FIX_OPS:
        r2[op], rows[op] = _apply(mutated, [op])

    # 枚举最多 2 次修法的组合，记录可行组合。
    feasible: list[list[str]] = []
    all_ops = list(FIX_OPS)
    for depth in (1, 2):
        for combo in combinations(all_ops, depth):
            # 一次清洗只有一个 k：两种 iqr 不能同时出现在一个组合里。
            if {"iqr_k1_5", "iqr_k3"} <= set(combo):
                continue
            rr, ro = _apply(mutated, list(combo))
            if _fixed(rr, ro):
                feasible.append(sorted(combo))

    shortest = sorted(feasible, key=lambda combo: (len(combo), combo))[0] if feasible else None
    return {
        "id": case["id"],
        "group": case["group"],
        "fixable": bool(feasible),
        "shortest": shortest,
        "feasible": feasible,
        "r2": r2,
        "rows": rows,
    }

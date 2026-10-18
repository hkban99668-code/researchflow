"""故障注入函数：从 `examples/sample_dataset.csv` 变异出新的 CSV 与课题。

每个数据故障是纯函数 `(frame, rng, **params) -> pd.DataFrame`，只依赖传入的
`rng`（`np.random.Generator`），保证同一种子得到同一份 CSV。改课题字段的两类
（`unknown_algorithm` / `empty_retrieval`）是 `(topic, **params) -> dict`，数据不变。

类型约定：
- `DATA_FAULTS`：数据变异，先复制再改列，返回新 DataFrame。
- `TOPIC_FAULTS`：课题变异，`deepcopy` 后改字段，返回新字典。
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "examples" / "sample_dataset.csv"

# 一个与本地语料所有关键词都不重叠的课题，用于 empty_retrieval 故障。
NO_MATCH_TOPIC = "黑洞视界上的量子信息悖论"


def load_sample() -> pd.DataFrame:
    """读取原始样例数据（61 行 + 表头，含 2 个缺失单元格与 1 行重复行）。"""
    return pd.read_csv(SAMPLE_CSV)


# --------------------------------------------------------------------------- #
# 数据故障：`(frame, rng, **params) -> pd.DataFrame`
# --------------------------------------------------------------------------- #


def clean(frame: pd.DataFrame, rng: np.random.Generator, **params) -> pd.DataFrame:
    """原样复制：clean 组基线。"""
    return frame.copy()


def duplicate_rows(
    frame: pd.DataFrame, rng: np.random.Generator, *, rows: int = 3, **params
) -> pd.DataFrame:
    """复制若干行（去重后通过预检）。"""
    idx = rng.choice(len(frame), size=rows, replace=True)
    return pd.concat([frame, frame.iloc[idx]], ignore_index=True)


def high_missing(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    column: str = "x2",
    rate: float = 0.3,
    **params,
) -> pd.DataFrame:
    """把某个 x 列按比例置空，缺失率超过 `max_missing_rate`。"""
    out = frame.copy()
    k = max(1, int(len(out) * rate))
    idx = rng.choice(len(out), size=k, replace=False)
    out.loc[idx, column] = np.nan
    return out


def drop_column(
    frame: pd.DataFrame, rng: np.random.Generator, *, column: str = "x2", **params
) -> pd.DataFrame:
    """删掉一个必需列。"""
    return frame.drop(columns=[column])


def rename_column(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    column: str = "x2",
    new_name: str = "x2_renamed",
    **params,
) -> pd.DataFrame:
    """改一个必需列的列名。"""
    return frame.rename(columns={column: new_name})


def x_out_of_range(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    column: str = "x1",
    value: float = 100.0,
    count: int = 5,
    **params,
) -> pd.DataFrame:
    """把某个 x 列的少量值改到 `ranges` 之外（Pandera 预检拦下）。"""
    out = frame.copy()
    idx = rng.choice(len(out), size=count, replace=False)
    out.loc[idx, column] = value
    return out


def too_few_rows(
    frame: pd.DataFrame, rng: np.random.Generator, *, rows: int = 10, **params
) -> pd.DataFrame:
    """只保留不到 `min_rows` 行。"""
    idx = rng.choice(len(frame), size=rows, replace=False)
    return frame.iloc[np.sort(idx)].reset_index(drop=True)


def type_pollution(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    column: str = "x1",
    rate: float = 0.2,
    **params,
) -> pd.DataFrame:
    """某个 x 列混入字符串：数值化（coerce）后成为缺失，比例够高即 RF-003。

    先转 `object` 再写字符串，否则往 float64 列里 `.loc` 赋值字符串会被 pandas
    拒绝（FutureWarning，值并不真正写入），故障就落空了。
    """
    out = frame.copy()
    out[column] = out[column].astype(object)
    k = max(1, int(len(out) * rate))
    idx = rng.choice(len(out), size=k, replace=False)
    out.loc[idx, column] = "not_a_number"
    return out


def y_outliers(
    frame: pd.DataFrame, rng: np.random.Generator, *, factor: float = 20.0, count: int = 8, **params
) -> pd.DataFrame:
    """y 列少量值放大为极端值（③ 不查 y 的范围，放行；⑦ r2 掉到阈值下）。"""
    out = frame.copy()
    idx = rng.choice(len(out), size=count, replace=False)
    out.loc[idx, "y"] = out.loc[idx, "y"] * factor
    return out


def y_outliers_offset(
    frame: pd.DataFrame, rng: np.random.Generator, *, offset: float = 40.0, count: int = 6, **params
) -> pd.DataFrame:
    """y 列少量值向同一方向平移（加固定 offset）：只往一个方向偏的离群值。

    与 `y_outliers`（按 factor 放大，方向取决于 y 的符号）不同，这里所有被选中的
    y 都往同一方向移动，模拟"某类样本被系统性高估/低估"的注入。③ 不查 y 的范围，
    放行；⑦ r2 掉到阈值下。回到 ③ 加一个 y 的离群值裁剪即可恢复。
    """
    out = frame.copy()
    idx = rng.choice(len(out), size=count, replace=False)
    out.loc[idx, "y"] = out.loc[idx, "y"] + offset
    return out


def y_noise(
    frame: pd.DataFrame, rng: np.random.Generator, *, scale: float = 30.0, **params
) -> pd.DataFrame:
    """给 y 加大幅高斯噪声，噪声大到 r2 明显低于 0.8。"""
    out = frame.copy()
    out["y"] = out["y"] + rng.normal(0.0, scale, size=len(out))
    return out


def y_shuffled(frame: pd.DataFrame, rng: np.random.Generator, **params) -> pd.DataFrame:
    """y 列整列随机打乱：特征与目标之间没有关系，应转人工。"""
    out = frame.copy()
    out["y"] = rng.permutation(out["y"].to_numpy(dtype=float))
    return out


def y_outliers_with_noise(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    factor: float = 20.0,
    count: int = 8,
    noise_scale: float = 1.0,
    **params,
) -> pd.DataFrame:
    """少量 y 极端值 + 轻微整列高斯噪声（混合故障，可修）。

    先按 `factor` 放大少量 y，再给整列 y 加 `noise_scale` 的高斯噪声。噪声要小到
    "去掉极端值后 r2 仍 ≥ 0.8"；否则就不是可修，而是退化成 `y_outliers_heavy_noise`。
    症状在 compute（⑦ r2 低），根因在 data_prepare（离群值可裁剪 + 噪声不影响关系）。
    """
    out = frame.copy()
    idx = rng.choice(len(out), size=count, replace=False)
    out.loc[idx, "y"] = out.loc[idx, "y"] * factor
    out["y"] = out["y"] + rng.normal(0.0, noise_scale, size=len(out))
    return out


def y_outliers_one_feature_missing(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    factor: float = 20.0,
    count: int = 8,
    column: str = "x1",
    missing_count: int = 1,
    **params,
) -> pd.DataFrame:
    """少量 y 极端值 + 某个 x 列少量缺失（混合故障，可修）。

    缺失要少到 `raw_missing_rate` 仍低于 `max_missing_rate`（③ 放行，不是单步故障），
    真正的故障是 y 的离群值。回到 data_prepare 开离群值裁剪即可恢复；缺失的行本来就
    会在清洗时被 dropna 丢掉，不影响关系。
    """
    out = frame.copy()
    idx = rng.choice(len(out), size=count, replace=False)
    out.loc[idx, "y"] = out.loc[idx, "y"] * factor
    midx = rng.choice(len(out), size=missing_count, replace=False)
    out.loc[midx, column] = np.nan
    return out


def y_outliers_heavy_noise(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    factor: float = 20.0,
    count: int = 8,
    noise_scale: float = 30.0,
    **params,
) -> pd.DataFrame:
    """少量 y 极端值 + 大幅整列噪声（应转人工）。

    即使去掉极端值，噪声已经破坏特征与目标的关系，r2 仍明显低于 0.8。与
    `y_outliers_with_noise` 的区别只在 `noise_scale` 大小：这里大、修不好。
    """
    out = frame.copy()
    idx = rng.choice(len(out), size=count, replace=False)
    out.loc[idx, "y"] = out.loc[idx, "y"] * factor
    out["y"] = out["y"] + rng.normal(0.0, noise_scale, size=len(out))
    return out


def nonlinear_target(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    feature: str = "x2",
    noise_scale: float = 1.0,
    **params,
) -> pd.DataFrame:
    """y 改成某个特征的非线性函数（平方）再加少量噪声（应转人工）。

    线性模型拟合不了平方关系，白名单里也没有"换模型"的参数，所以任何清洗都救不回来。
    选 `x2`（取值对称分布在 ±10 附近）平方后与特征几乎零线性相关；`x1`（0~10）平方
    后仍近似线性，会误判成可修，所以默认不用 x1。
    """
    out = frame.copy()
    out["y"] = out[feature].astype(float) ** 2 + rng.normal(0.0, noise_scale, size=len(out))
    return out


def y_outliers_partial(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    factor: float = 100.0,
    count: int = 30,
    **params,
) -> pd.DataFrame:
    """极端值很多（超过样本的 30%），裁剪后仍修不好（应转人工）。

    极端值数量多到 IQR 裁剪只能去掉一小部分（Tukey 栅栏对"占四分位数的大簇"天然免疫），
    剩下的离群值继续压低 r2，任何裁剪设置都救不回 r2 ≥ 0.8。
    """
    out = frame.copy()
    idx = rng.choice(len(out), size=count, replace=False)
    out.loc[idx, "y"] = out.loc[idx, "y"] * factor
    return out


# --------------------------------------------------------------------------- #
# T-019 新增：多种修法与陷阱用例。                                              #
# --------------------------------------------------------------------------- #

_EXP_COEF = (0.5, -0.4, 0.3, 1.0)


def exp_target(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    noise_scale: float = 0.05,
    **params,
) -> pd.DataFrame:
    """y 改成特征线性组合的指数（保证 y > 0），加少量乘性噪声。

    线性模型拟合不了指数关系（⑦ r2 低），回到 `compute` 用 `target_transform=log`
    即可恢复。系数 `_EXP_COEF` 与样本数据的线性结构同量级，exp 后 y 跨度足够大，
    保证原始尺度上线性拟合的 r2 明显低于阈值。
    """
    out = frame.copy()
    a, b, c, d = _EXP_COEF
    linear = (
        a * out["x1"].astype(float) + b * out["x2"].astype(float) + c * out["x3"].astype(float) + d
    )
    out["y"] = np.exp(linear) * (1.0 + rng.normal(0.0, noise_scale, size=len(out)))
    return out


def exp_target_with_outliers(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    noise_scale: float = 0.05,
    nonpositive_count: int = 5,
    **params,
) -> pd.DataFrame:
    """`exp_target` 基础上让 y 含少量非正值（应转人工）。

    非正值取一个很小的负数（落在 IQR 栅栏之内），因此 `iqr` 裁剪去不掉它们；而
    `log` 变换又因 y ≤ 0 抛 RF-004。于是任何单一修法（甚至 iqr + log 两次组合）
    都修不好指数关系，正确动作是转人工。
    """
    out = exp_target(frame, rng, noise_scale=noise_scale)
    idx = rng.choice(len(out), size=nonpositive_count, replace=False)
    out.loc[idx, "y"] = -0.5
    return out


def k_choice_trap(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    rows: int = 36,
    n_extreme: int = 4,
    n_mild: int = 3,
    extreme_factor: float = 30.0,
    mild_factor: float = 3.2,
    **params,
) -> pd.DataFrame:
    """样本数接近 `min_rows`，y 含少量极端值、尾部较长（应回到 data_prepare，iqr k=3）。

    重新生成一个 `rows` 行的合成数据集（列与范围同样本数据）：干净的线性关系上加
    少量噪声，再注入 `n_extreme` 个极端值（×`extreme_factor`，落在 3 个 IQR 之外）与
    `n_mild` 个中度值（×`mild_factor`，落在 1.5~3 个 IQR 之间）。

    k=1.5 会连中度值一起误删，导致行数低于 `min_rows`（陷阱）；k=3 只去掉极端值，
    行数达标且 r2 恢复。这里的"尾部较长"来自 y 的自然分布，不是额外注入的噪声。
    """
    n = int(rows)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-10, 10, n)
    x3 = rng.uniform(0, 5, n)
    y = 2.0 * x1 - 1.5 * x2 + 0.6 * x3 - 0.44 + rng.normal(0.0, 0.15, n)

    extreme_idx = rng.choice(n, size=int(n_extreme), replace=False)
    y[extreme_idx] = y[extreme_idx] * extreme_factor
    rest = np.setdiff1d(np.arange(n), extreme_idx)
    mild_idx = rng.choice(rest, size=int(n_mild), replace=False)
    y[mild_idx] = y[mild_idx] * mild_factor

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


# --------------------------------------------------------------------------- #
# 课题故障：`(topic, **params) -> dict`
# --------------------------------------------------------------------------- #


def unknown_algorithm(topic: dict, *, algorithm: str = "ridge", **params) -> dict:
    """数据不动，把课题的 `algorithm` 改为未登记值。"""
    out = deepcopy(topic)
    out["param_set"]["algorithm"] = algorithm
    return out


def empty_retrieval(topic: dict, *, topic_text: str = NO_MATCH_TOPIC, **params) -> dict:
    """把课题改成本地语料完全不命中的题目。"""
    out = deepcopy(topic)
    out["topic"] = topic_text
    return out


DATA_FAULTS: dict[str, Callable[..., pd.DataFrame]] = {
    "clean": clean,
    "duplicate_rows": duplicate_rows,
    "high_missing": high_missing,
    "drop_column": drop_column,
    "rename_column": rename_column,
    "x_out_of_range": x_out_of_range,
    "too_few_rows": too_few_rows,
    "type_pollution": type_pollution,
    "y_outliers": y_outliers,
    "y_outliers_offset": y_outliers_offset,
    "y_noise": y_noise,
    "y_shuffled": y_shuffled,
    "y_outliers_with_noise": y_outliers_with_noise,
    "y_outliers_one_feature_missing": y_outliers_one_feature_missing,
    "y_outliers_heavy_noise": y_outliers_heavy_noise,
    "nonlinear_target": nonlinear_target,
    "y_outliers_partial": y_outliers_partial,
    "exp_target": exp_target,
    "exp_target_with_outliers": exp_target_with_outliers,
    "k_choice_trap": k_choice_trap,
}

TOPIC_FAULTS: dict[str, Callable[..., dict]] = {
    "unknown_algorithm": unknown_algorithm,
    "empty_retrieval": empty_retrieval,
}


def apply_fault(name: str, frame: pd.DataFrame, seed: int, params: dict) -> pd.DataFrame:
    """按故障名变异数据；`seed` 决定 `rng`，保证同种子同输出。"""
    if name not in DATA_FAULTS:
        raise KeyError(f"未登记的数据故障: {name!r}（已登记: {', '.join(sorted(DATA_FAULTS))}）")
    rng = np.random.default_rng(seed)
    return DATA_FAULTS[name](frame, rng, **params)


def apply_topic_fault(name: str, topic: dict, params: dict) -> dict:
    """按故障名变异课题。"""
    if name not in TOPIC_FAULTS:
        raise KeyError(f"未登记的课题故障: {name!r}（已登记: {', '.join(sorted(TOPIC_FAULTS))}）")
    return TOPIC_FAULTS[name](topic, **params)

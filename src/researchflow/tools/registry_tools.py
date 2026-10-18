"""注册表默认工具的入参模型与包装函数。

模型给不出 `pd.DataFrame`，所以这里注册的是**包装函数**：收 `csv_path` 这类引用，
在函数内部解析路径（必须落在 `ToolContext.workspace_root` 之下）、读数据，再调原函数。
原函数签名不动，现有节点继续直接调原函数。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

from ..compute.direct import ALGORITHM, run_direct
from ..core.errors import ConfigError
from ..core.schemas import ComputeResult, LiteratureRef, QualitySummary
from ..core.tools import ToolContext, ToolInputError, ToolPathError, ToolRegistry, ToolSpec
from .clean_strategies import clean_frame
from .quality import precheck
from .rules_engine import AcceptanceRules, load_rules

DEFAULT_RULES_PATH = "configs/rules/data_acceptance.yaml"

# `csv_path` 与 `rules_path` 要求的文件后缀：路径解析（`_resolve_within`）时就拦下
# 明显给错的文件，把"需要什么、收到什么"写进消息，模型才知道怎么改。
CSV_SUFFIXES: tuple[str, ...] = (".csv",)
RULES_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")


class SearchLiteratureArgs(BaseModel):
    """`search_literature` 的入参。"""

    query: str = Field(..., min_length=1, description="检索关键词，非空")
    limit: int = Field(6, ge=1, le=20, description="最多返回几条文献")


class SearchLiteratureOutput(BaseModel):
    """`search_literature` 的输出。"""

    refs: list[LiteratureRef] = Field(default_factory=list)


class CheckDataQualityArgs(BaseModel):
    """`check_data_quality` 的入参。"""

    csv_path: str = Field(..., min_length=1, description="工作区内的 CSV 相对路径或绝对路径")
    rules_path: str = Field(DEFAULT_RULES_PATH, description="工作区内的数据准入规则 YAML")

    @field_validator("csv_path")
    @classmethod
    def _csv_path_non_blank(cls, value: str) -> str:
        """必填参数空白视作未提供，交给入参校验报 invalid_args。"""
        value = value.strip()
        if not value:
            raise ValueError("csv_path 不能为空")
        return value

    @field_validator("rules_path")
    @classmethod
    def _rules_path_default_when_blank(cls, value: str) -> str:
        """空字符串/纯空白视作未提供，落回默认规则路径。"""
        value = value.strip()
        return value or DEFAULT_RULES_PATH


class RunDirectComputeArgs(BaseModel):
    """`run_direct_compute` 的入参。"""

    csv_path: str = Field(..., min_length=1, description="工作区内的 CSV 相对路径或绝对路径")
    target: str = Field(..., description="目标列名")
    features: list[str] = Field(..., min_length=1, description="特征列名，至少一个")
    test_size: float = Field(0.25, gt=0, lt=1, description="测试集比例")
    seed: int = Field(42, description="随机种子，保证可复现")

    @field_validator("csv_path")
    @classmethod
    def _csv_path_non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("csv_path 不能为空")
        return value


def search_literature(args: SearchLiteratureArgs, ctx: ToolContext) -> SearchLiteratureOutput:
    """按关键词检索文献；检索器由 `ToolContext.retriever` 注入，不回退。

    `ctx.retriever` 为 None 时抛 `ConfigError`（RF-001），消息说明"没有配置检索器"，
    不再悄悄改用本地语料：把"没配检索器"伪装成"检索成功"是一个静默失效（D-008）。
    """
    if ctx.retriever is None:
        raise ConfigError("没有配置检索器：ToolContext.retriever 未注入")
    return SearchLiteratureOutput(refs=ctx.retriever.search(args.query, k=args.limit))


def check_data_quality(args: CheckDataQualityArgs, ctx: ToolContext) -> QualitySummary:
    """复现 ③ 的顺序：`clean_frame` → `precheck`，产出数据质量摘要。

    清洗用的列取规则里的 `required_columns`：③ 用的是 `[*features, target]`，
    两者在仓库自带示例数据上相同（见汇报"调查结论"1）。
    """
    csv_path = _resolve_within(args.csv_path, ctx, label="csv_path", suffixes=CSV_SUFFIXES)
    rules_path = _resolve_within(args.rules_path, ctx, label="rules_path", suffixes=RULES_SUFFIXES)
    try:
        rules = load_rules(rules_path, model=AcceptanceRules)
    except yaml.YAMLError as exc:
        raise ToolInputError(f"rules_path 不是可解析的 YAML: {_parse_error_summary(exc)}") from exc

    frame = _read_csv(csv_path)
    cleaned, stats = clean_frame(frame, columns=list(rules.required_columns))
    return precheck(
        cleaned,
        rules,
        duplicates_removed=stats["duplicates_removed"],
        raw_missing_rate=stats["raw_missing_rate"],
        dropped_rows=stats["dropped_rows"],
        rows_before=stats["rows_before"],
    )


def run_direct_compute(args: RunDirectComputeArgs, ctx: ToolContext) -> ComputeResult:
    """在同一进程内跑 DIRECT 线性回归；会消耗算力，标签是 `exec`。

    先按 ③ 的口径清洗（数值化 → 去重 → 丢弃关键列缺失的行），再交给 `run_direct`，
    这样同一份 CSV 走工具与走 ④ 节点得到同样的 metrics。
    """
    csv_path = _resolve_within(args.csv_path, ctx, label="csv_path", suffixes=CSV_SUFFIXES)
    frame = _read_csv(csv_path)
    cleaned, _ = clean_frame(frame, columns=[*args.features, args.target])
    return run_direct(
        cleaned,
        target=args.target,
        features=list(args.features),
        params={"test_size": args.test_size},
        seed=args.seed,
        # 工具入参不暴露 algorithm：DIRECT 工具只登记了线性回归（P1-4 前半）
        algorithm=ALGORITHM,
    )


def build_default_registry() -> ToolRegistry:
    """登记 3 个面向模型的工具：检索、数据质量预检、DIRECT 计算。

    放在 tools 层而不是 `core/tools.py`：默认工具的实现在这里，
    若由 core 提供这个工厂函数，core 就得反向依赖上层（T-003 的 D-005 转入项）。
    """
    registry = ToolRegistry()
    for spec in build_default_specs():
        registry.register(spec)
    return registry


def build_default_specs() -> list[ToolSpec]:
    """默认登记的 3 个工具，按 `build_default_registry().names()` 的顺序返回。"""
    return [
        _spec(
            name="search_literature",
            description=(
                "按关键词检索文献（检索后端由配置决定，默认本地语料）。什么时候用："
                "需要为研究问题找依据时。返回 refs 列表，每项含 ref_id/title/source/"
                "year/summary/score，按相关度降序；一条都没命中时 refs 为空列表。"
            ),
            args_model=SearchLiteratureArgs,
            fn=search_literature,
            side_effect="read",
        ),
        _spec(
            name="check_data_quality",
            description=(
                "按数据准入规则检查一份 CSV 是否可用：清洗后给出行数、列数、缺失率、"
                "移除重复行数与结论备注。什么时候用：拿数据做计算或写结论之前。"
                "规则不满足时返回 execution_error（detail.code 为 RF-003）。"
            ),
            args_model=CheckDataQualityArgs,
            fn=check_data_quality,
            side_effect="read",
        ),
        _spec(
            name="run_direct_compute",
            description=(
                "对一份 CSV 跑单算法线性回归，返回 r2/rmse/mae 等 metrics 与系数。"
                "什么时候用：已确认数据质量、需要量化结论时。会消耗算力，"
                "调用前先跑一次 check_data_quality。"
            ),
            args_model=RunDirectComputeArgs,
            fn=run_direct_compute,
            side_effect="exec",
        ),
    ]


def _spec(
    *,
    name: str,
    description: str,
    args_model: type[BaseModel],
    fn: Callable[[Any, ToolContext], BaseModel],
    side_effect: Any,
    timeout_s: float = 30.0,
) -> ToolSpec:
    """构造 ToolSpec；`fn` 的第一个参数写成 Any，让具体入参模型不必做 cast。"""
    return ToolSpec(
        name=name,
        description=description,
        args_model=args_model,
        fn=fn,
        side_effect=side_effect,
        timeout_s=timeout_s,
    )


def _read_csv(csv_path: Path) -> pd.DataFrame:
    """读 CSV；解析失败（坏内容 / 空文件 / 编码）转成 `ToolInputError`，让模型能改。"""
    try:
        return pd.read_csv(csv_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ToolInputError(f"csv_path 不是可解析的 CSV: {_parse_error_summary(exc)}") from exc


def _parse_error_summary(exc: Exception) -> str:
    """把解析异常压成不超过 200 字符的简短说明，回填给模型。"""
    return str(exc)[:200]


def _resolve_within(
    raw_path: str,
    ctx: ToolContext,
    *,
    label: str,
    kind: Literal["file", "dir"] = "file",
    suffixes: tuple[str, ...] | None = None,
) -> Path:
    """解析路径并确认它落在工作区内、类型与后缀符合要求。

    越界抛 `ToolPathError`（permission_denied）；不存在、类型不对、后缀不符抛
    `ToolInputError`（invalid_args），消息写清"需要什么、收到什么"，方便模型自纠。
    """
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = ctx.workspace_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ctx.workspace_root.resolve()):
        raise ToolPathError(f"{label} 越出工作区: {raw_path}")
    if not resolved.exists():
        raise ToolInputError(f"{label} 不存在: {raw_path}")
    if kind == "file" and resolved.is_dir():
        raise ToolInputError(f"{label} 需要一个文件，收到的是目录: {raw_path}")
    if kind == "dir" and not resolved.is_dir():
        raise ToolInputError(f"{label} 需要一个目录，收到的是文件: {raw_path}")
    if suffixes is not None and resolved.suffix not in suffixes:
        raise ToolInputError(f"{label} 需要后缀为 {'/'.join(suffixes)} 的文件，收到: {raw_path}")
    return resolved

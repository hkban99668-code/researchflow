"""归因 Agent 专用的工具注册表。

只读工具从**当前 state 和 run 目录**读取，不接受任意文件路径参数——归因 Agent
只能"看"这次运行里的数据与结论，不能越出 run 目录去碰别的文件。决策工具
（`propose_backtrack` / `escalate_to_human`）只校验参数形状、返回"已记录"，
真正的决策执行在图里由 `attribute` 节点完成（含 patch 与目标节点匹配性校验）。

注册表每次决策都要按最新 state 重建：回溯之后 `cleaned_data` / `validation_result`
都变了，闭包捕获的是旧快照会让 Agent 看到过期的信息。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression

from ..core.config import RunContext
from ..core.schemas import BacktrackPatch, ComputeResult, ErrorRecord, ValidationReport
from ..core.tools import ToolContext, ToolInputError, ToolRegistry, ToolSpec
from .rules_engine import AcceptanceRules, load_rules


class ReadValidationReportArgs(BaseModel):
    """`read_validation_report` 无入参。"""


class FailedCheck(BaseModel):
    """一条未通过的硬规则检查。"""

    name: str
    detail: str = ""
    owner_step: str


class ReadValidationReportOutput(BaseModel):
    """未通过检查的列表。"""

    failed_checks: list[FailedCheck] = Field(default_factory=list)


class ReadStepErrorArgs(BaseModel):
    """`read_step_error` 无入参。"""


class ReadStepErrorOutput(BaseModel):
    """当前错误记录；没有时为 null。"""

    error: ErrorRecord | None = None


class InspectColumnArgs(BaseModel):
    """`inspect_column` 的入参：只接受列名，不接受路径。"""

    column: str = Field(..., min_length=1, description="要查看统计量的列名")


class InspectColumnOutput(BaseModel):
    """单列统计量：只含汇总数字，不含原始行。"""

    column: str
    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    outliers_k15: int
    outliers_k30: int


class ReadComputeResultArgs(BaseModel):
    """`read_compute_result` 无入参。"""


class ReadComputeResultOutput(BaseModel):
    """④ 的计算结果摘要。"""

    metrics: dict[str, float] = Field(default_factory=dict)
    n_train: float | None = None
    n_test: float | None = None


class FeatureTargetRelationArgs(BaseModel):
    """`feature_target_relation` 无入参：目标列与特征从当前 param_set 取。"""


class FeatureCorrelation(BaseModel):
    """一个特征与目标列的 Pearson 相关系数；零方差特征无法计算时为 null。"""

    feature: str
    pearson: float | None


class RobustRelationStats(BaseModel):
    """按某个 IQR 阈值 k 去掉目标列离群值后重算的关系统计量。"""

    correlations: list[FeatureCorrelation] = Field(default_factory=list)
    r2: float
    rows_removed: int
    # 去掉离群值后剩余的行数（= n_rows - rows_removed），供判断"k 取小了会不会删过头"。
    rows_after: int


class LogTargetStats(BaseModel):
    """对 log(y) 做线性回归的诊断量；y 含 ≤ 0 值时不可用。"""

    applicable: bool
    r2: float | None = None
    note: str = ""


class FeatureTargetRelationOutput(BaseModel):
    """特征与目标列的关系统计量：只含汇总数字，不含原始行。

    r2 / log_target / r2_without_feature 都是**全量数据**上的诊断量，不是 ④ 在
    测试集上的得分，用来比较不同修法的方向，不保证修完一定过线。
    """

    correlations: list[FeatureCorrelation] = Field(default_factory=list)
    r2: float
    n_rows: int
    # ③ 数据准入规则的 min_rows，供判断"去掉离群值后是否还达标"。
    min_rows: int
    # 按两个 IQR 阈值 k 分别给出稳健统计：k 越小去得越多，覆盖 y_outlier_k
    # 允许范围内（1.5~10）常用的两个取值。
    robust_by_k: dict[str, RobustRelationStats] = Field(default_factory=dict)
    log_target: LogTargetStats
    # 分别去掉每一个特征后，用剩下的特征在全量数据上做线性回归的 R²（只剩一个
    # 特征时也照算）；键与当前特征一致。
    r2_without_feature: dict[str, float] = Field(default_factory=dict)
    note: str = ""


class ProposeBacktrackArgs(BaseModel):
    """`propose_backtrack` 的入参：回到哪一步、改什么参数。"""

    target_step: Literal["data_prepare", "compute"]
    patch: BacktrackPatch = Field(default_factory=BacktrackPatch)
    reason: str = ""


class EscalateToHumanArgs(BaseModel):
    """`escalate_to_human` 的入参。

    `suspected_step` 可选：诊断出问题大概在哪一步时填上（"诊断对了但修不了"），
    与"根本没诊断"区分开；没把握就不填。
    """

    reason: str = ""
    suspected_step: str | None = None


class DecisionAck(BaseModel):
    """决策工具的确认输出：决策在图里执行，工具只返回"已记录"。"""

    message: str = "已记录"


def _inspect_column_stats(series: pd.Series, column: str) -> InspectColumnOutput:
    """把一列算成只含汇总数字的输出，不返回任何原始行。"""
    numeric = pd.to_numeric(series, errors="coerce")
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1

    def outliers(k: float) -> int:
        return int(((numeric < q1 - k * iqr) | (numeric > q3 + k * iqr)).sum())

    return InspectColumnOutput(
        column=column,
        count=int(numeric.count()),
        mean=float(numeric.mean()),
        std=float(numeric.std()),
        min=float(numeric.min()),
        q1=q1,
        median=float(numeric.median()),
        q3=q3,
        max=float(numeric.max()),
        outliers_k15=outliers(1.5),
        outliers_k30=outliers(3.0),
    )


def _relation_stats(
    data: pd.DataFrame, features: list[str], target: str
) -> tuple[list[FeatureCorrelation], float, list[str]]:
    """算一组特征与目标列的 Pearson 相关系数与线性回归全量 R²。

    零方差特征无法算相关系数（std=0 会除零得到 NaN），pearson 记为 None，
    特征名收集进返回的列表；R² 是只用 `features` 做线性回归在 `data` 上的得分
    （口径与 ④ 的测试集 r2 不同，见工具描述）。
    """
    correlations: list[FeatureCorrelation] = []
    constant_features: list[str] = []
    for feature in features:
        if float(data[feature].std()) == 0.0:
            constant_features.append(feature)
            correlations.append(FeatureCorrelation(feature=feature, pearson=None))
        else:
            correlations.append(
                FeatureCorrelation(feature=feature, pearson=float(data[feature].corr(data[target])))
            )
    x = data[features].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    r2 = float(model.score(x, y))
    return correlations, r2, constant_features


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """线性回归在全量数据上的 R²（诊断口径，与 `_relation_stats` 一致）。"""
    model = LinearRegression().fit(x, y)
    return float(model.score(x, y))


def _read_min_rows(run_ctx: RunContext) -> int:
    """从 ③ 数据准入规则读取 min_rows。"""
    path = run_ctx.settings.project_root / "configs" / "rules" / "data_acceptance.yaml"
    rules = load_rules(path, model=AcceptanceRules)
    return int(rules.min_rows)


def _robust_relation_stats(
    data: pd.DataFrame, features: list[str], target: str, k: float
) -> RobustRelationStats:
    """按 IQR 阈值 k 去掉目标列离群值后重算关系统计量。

    k 越小，[Q1 − k·IQR, Q3 + k·IQR] 越窄、去得越多；IQR 为 0（目标列几乎恒定）
    时阈值规则不适用，视为没有离群值，稳健值等于原值。
    """
    target_series = data[target]
    q1 = float(target_series.quantile(0.25))
    q3 = float(target_series.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0.0:
        correlations, r2, _ = _relation_stats(data, features, target)
        return RobustRelationStats(
            correlations=correlations, r2=r2, rows_removed=0, rows_after=len(data)
        )
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    robust_data = data[(target_series >= lower) & (target_series <= upper)]
    rows_removed = len(data) - len(robust_data)
    correlations, r2, _ = _relation_stats(robust_data, features, target)
    return RobustRelationStats(
        correlations=correlations, r2=r2, rows_removed=rows_removed, rows_after=len(robust_data)
    )


def build_attribution_registry(state: dict, run_ctx: RunContext) -> ToolRegistry:
    """按当前 state 与 run 目录构建归因工具注册表（每次决策重建）。"""
    registry = ToolRegistry()

    def read_validation_report(
        args: ReadValidationReportArgs, ctx: ToolContext
    ) -> ReadValidationReportOutput:
        report: ValidationReport | None = state.get("validation_result")
        failed = [check for check in (report.checks if report else []) if not check.passed]
        return ReadValidationReportOutput(
            failed_checks=[
                FailedCheck(name=check.name, detail=check.detail, owner_step=check.owner_step)
                for check in failed
            ]
        )

    def read_step_error(args: ReadStepErrorArgs, ctx: ToolContext) -> ReadStepErrorOutput:
        error: ErrorRecord | None = state.get("error")
        return ReadStepErrorOutput(error=error)

    def inspect_column(args: InspectColumnArgs, ctx: ToolContext) -> InspectColumnOutput:
        cleaned = state.get("cleaned_data")
        if cleaned is None:
            raise ToolInputError("没有审核通过的清洗数据（cleaned_data 为空）")
        path = run_ctx.read_verified(cleaned)
        frame = pd.read_csv(path)
        if args.column not in frame.columns:
            raise ToolInputError(f"列不存在: {args.column!r}，可用列: {', '.join(frame.columns)}")
        return _inspect_column_stats(frame[args.column], args.column)

    def read_compute_result(
        args: ReadComputeResultArgs, ctx: ToolContext
    ) -> ReadComputeResultOutput:
        result: ComputeResult | None = state.get("compute_result")
        metrics = dict(result.metrics) if result else {}
        n_train = metrics.get("n_train")
        n_test = metrics.get("n_test")
        return ReadComputeResultOutput(metrics=metrics, n_train=n_train, n_test=n_test)

    def feature_target_relation(
        args: FeatureTargetRelationArgs, ctx: ToolContext
    ) -> FeatureTargetRelationOutput:
        cleaned = state.get("cleaned_data")
        if cleaned is None:
            raise ToolInputError("没有审核通过的清洗数据（cleaned_data 为空）")
        param_set = state.get("param_set")
        if param_set is None:
            raise ToolInputError("param_set 为空")
        params = (
            param_set.params if hasattr(param_set, "params") else (param_set.get("params") or {})
        )
        target = params.get("target")
        features = list(params.get("features") or [])
        if not target or not features:
            raise ToolInputError(
                f"param_set 缺少 target 或 features（target={target!r}, features={features!r}）"
            )

        path = run_ctx.read_verified(cleaned)
        frame = pd.read_csv(path)
        missing = [column for column in [*features, target] if column not in frame.columns]
        if missing:
            raise ToolInputError(
                f"列不存在: {', '.join(missing)}，可用列: {', '.join(frame.columns)}"
            )

        data = frame[[*features, target]].apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            raise ToolInputError("数值化并丢弃缺失后没有可用样本")

        correlations, r2, constant_features = _relation_stats(data, features, target)

        # 稳健统计：按 k=1.5 与 k=3 两个阈值分别给出。k 越小去得越多；这两个值
        # 对应 y_outlier_k 允许范围内（1.5~10）的常用取值，用于区分"有几个极端值、
        # 关系其实还在"和"关系真的没了"（见工具描述）。
        robust_by_k: dict[str, RobustRelationStats] = {}
        for key, k in (("1.5", 1.5), ("3", 3.0)):
            robust_by_k[key] = _robust_relation_stats(data, features, target, k)

        # log 变换的诊断量：y 全部 > 0 时用特征对 log(y) 做线性回归，给出全量 R²；
        # 否则 applicable=false，note 写明有多少个 ≤ 0 的值。
        target_vals = data[target].to_numpy(dtype=float)
        nonpositive = int((target_vals <= 0).sum())
        if nonpositive == 0:
            log_target = LogTargetStats(
                applicable=True,
                r2=_linear_r2(data[features].to_numpy(dtype=float), np.log(target_vals)),
            )
        else:
            log_target = LogTargetStats(
                applicable=False, note=f"{nonpositive} 个 y 值 ≤ 0，不能取对数"
            )

        # 排除每个特征后的 R²：分别去掉每一个特征，用剩下的特征在全量数据上回归
        # （只剩一个特征时也照算；全部排除时无法回归，跳过）。
        r2_without_feature: dict[str, float] = {}
        for feature in features:
            remaining = [name for name in features if name != feature]
            if not remaining:
                continue
            r2_without_feature[feature] = _linear_r2(
                data[remaining].to_numpy(dtype=float), target_vals
            )

        note = (
            f"特征 {', '.join(constant_features)} 方差为零，相关系数记为 null"
            if constant_features
            else ""
        )
        return FeatureTargetRelationOutput(
            correlations=correlations,
            r2=r2,
            n_rows=len(data),
            min_rows=_read_min_rows(run_ctx),
            robust_by_k=robust_by_k,
            log_target=log_target,
            r2_without_feature=r2_without_feature,
            note=note,
        )

    def propose_backtrack(args: ProposeBacktrackArgs, ctx: ToolContext) -> DecisionAck:
        return DecisionAck()

    def escalate_to_human(args: EscalateToHumanArgs, ctx: ToolContext) -> DecisionAck:
        return DecisionAck()

    for spec in _specs(
        [
            (
                "read_validation_report",
                (
                    "读当前硬规则校验里未通过的检查。什么时候用：需要判断哪里没通过、"
                    "该回到哪一步时。返回 failed_checks 列表，每项含 name/detail/owner_step。"
                ),
                ReadValidationReportArgs,
                read_validation_report,
                "read",
            ),
            (
                "read_step_error",
                "读当前节点的错误记录。什么时候用：链路报错后需要看错误码与回溯建议时。"
                "没有错误时返回 error=null。",
                ReadStepErrorArgs,
                read_step_error,
                "read",
            ),
            (
                "inspect_column",
                (
                    "对审核通过的清洗数据里的某一列返回统计量（count/mean/std/min/Q1/median/"
                    "Q3/max，以及 k=1.5 与 k=3 两种口径的 IQR 外点数）。只返回汇总数字，"
                    "不返回原始行。什么时候用：怀疑某列有离群值、需要量化偏离程度时。"
                ),
                InspectColumnArgs,
                inspect_column,
                "read",
            ),
            (
                "read_compute_result",
                "读 ④ 的计算结果（metrics、n_train、n_test）。什么时候用：需要对照 r2 等"
                "指标判断是否满足阈值时。",
                ReadComputeResultArgs,
                read_compute_result,
                "read",
            ),
            (
                "feature_target_relation",
                (
                    "对审核通过的清洗数据，返回每个特征与目标列的 Pearson 相关系数，以及"
                    "只用这些特征做线性回归的 R²。这里的 R² 衡量的是"
                    "「数据本身是否还存在可解释的线性关系」，与 ④ 在测试集上的 r2 口径不同"
                    "（那是这次随机拆分上的得分）。"
                    "r2 是**全量数据**上的值，可能被少量极端值拉低；robust_by_k 是去掉目标列"
                    "落在 [Q1 − k·IQR, Q3 + k·IQR] 之外的行后重算的统计量，按 k=1.5 与 k=3 "
                    '两个阈值分别给出（键 "1.5" / "3"），每一项含 correlations、r2、'
                    "rows_removed、rows_after（去掉离群值后剩余行数）。k 越小阈值越窄、去得"
                    "越多；这两个 k 对应 y_outlier_k 允许范围内（1.5~10）的常用取值。若 k=1.5 "
                    "的 r2 明显高于 k=3 的 r2，说明极端值集中在 1.5~3 个 IQR 之间、关系其实"
                    "还在（回 ③ 开离群值处理、k 取小就能修）；若两者都低且接近全量 r2，说明"
                    "关系真的被破坏了，清洗救不回来。"
                    "min_rows 是 ③ 数据准入规则的样本数下限，用来判断「去掉离群值后是否还"
                    "达标」：某档 k 的 rows_after 低于 min_rows，说明这个 k 会删过头、不能用。"
                    "log_target 是「对 log(y) 做线性回归」的诊断量：y 全部 > 0 时 applicable=true"
                    "并给出全量 R²（指数关系在 log 后 R² 会明显变高）；否则 applicable=false，"
                    "note 写明有多少个 ≤ 0 的值。r2_without_feature 是分别去掉每一个特征后、"
                    "用剩下特征在全量数据上回归的 R²，键与当前特征一致（只剩一个特征时也照算）。"
                    "注意：r2 / log_target.r2 / r2_without_feature 都是**全量数据上的诊断量**，"
                    "不是 ④ 在测试集上的得分，用来比较不同修法的方向，不保证修完一定过线。"
                    "什么时候用：需要判断目标列与特征之间是否还有关系时——若相关系数都接近 0、"
                    "R² 也很低，说明关系已被破坏，清洗参数救不回来，应当转人工。只返回汇总数字，"
                    "不返回原始行。"
                ),
                FeatureTargetRelationArgs,
                feature_target_relation,
                "read",
            ),
            (
                "propose_backtrack",
                (
                    "提交回溯决策：回到 target_step（data_prepare 或 compute）并用 patch "
                    "修改白名单参数。什么时候用：已经查清根因、且确定某个可调参数能修复时。"
                    '注意 patch 字段必须与 target_step 匹配：y_outlier_strategy（"none"/"iqr"）'
                    "与 y_outlier_k（1.5~10）只能配 data_prepare；target_transform"
                    '（"none"/"log"）与 exclude_features（要从建模中排除的特征列表，'
                    "必须是当前特征的真子集且至少保留 1 个）只能配 compute。"
                    "test_size 与 seed 已从白名单移除，不能通过改变随机划分来通过校验。"
                ),
                ProposeBacktrackArgs,
                propose_backtrack,
                "write",
            ),
            (
                "escalate_to_human",
                (
                    "转人工：诊断出问题在哪一步、但没有可调参数能修复（或数据关系已被破坏）。"
                    "什么时候用：确认无法用白名单参数自动修复时。若诊断出了问题大概在哪一步，"
                    "用 suspected_step 填上（只诊断、修不了），没把握就留空。"
                ),
                EscalateToHumanArgs,
                escalate_to_human,
                "write",
            ),
        ]
    ):
        registry.register(spec)
    return registry


def _specs(
    entries: list[tuple[str, str, type[BaseModel], Any, Literal["read", "write"]]],
) -> list[ToolSpec]:
    """把 (name, description, args_model, fn, side_effect) 列表转成 ToolSpec 列表。"""
    return [
        ToolSpec(
            name=name,
            description=description,
            args_model=args_model,
            fn=fn,
            side_effect=side_effect,
        )
        for name, description, args_model, fn, side_effect in entries
    ]

"""Gate 1 数据契约。

Real 与 Fake 两种 LLM 适配器共用这里的 Pydantic 模型，
保证 e2e（Fake）与真实冒烟（Real）的语义一致。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# `ref_id` 的形状：**来源前缀** + 该来源内的稳定 ID。
# 前缀区分来源（local / openalex / arxiv），避免不同来源撞号；
# 后半段由 `tools/literature.py::make_ref_id` 之类的函数生成，必须可复现。
REF_ID_PATTERN = r"^[a-z]+:[A-Za-z0-9._/-]+$"
REF_ID_RE = re.compile(REF_ID_PATTERN)


def ensure_relative_path(value: str) -> str:
    """校验产物路径必须是**相对 run_dir 的 POSIX 相对路径**。

    绝对路径会泄露用户目录名、换机器或进容器后即失效；`..` 会越出 run 目录；
    反斜杠是 Windows 分隔符，在 POSIX 下却是合法文件名字符，必须单独拦。
    三者都在这里挡掉，state 里的路径就只有一个解析入口（`RunContext`）。
    """
    if not value:
        raise ValueError("产物路径不能为空")
    if "\\" in value:
        raise ValueError(f"产物路径必须用 POSIX 分隔符 /: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"产物路径必须是相对 run_dir 的相对路径: {value!r}")
    if ".." in path.parts:
        raise ValueError(f"产物路径不允许包含 ..: {value!r}")
    return value


class LiteratureRef(BaseModel):
    """① 检索产出的文献引用摘要。

    `ref_id` 是文献的稳定 ID：同一篇文献每次运行必须得到同一个 ID，否则产物
    checksum 不可复现，以后检索评测（Recall@k）的人工标注也无处对照。
    """

    ref_id: str
    title: str
    source: str = ""
    year: int | None = None
    url: str = ""
    summary: str = ""
    score: float = 0.0
    # 产出它的检索器名（`Retriever.name`），换检索器后能追溯文献来源
    retriever: str = ""
    # T-030 chunk 级引用：命中块由混合重排器填入（`"{ref_id}#c{序号}"`）
    chunk_id: str | None = None
    # 本次结果来自缓存还是网络：本地语料恒为 None，真实 API 检索器填 True/False
    cached: bool | None = None

    @field_validator("ref_id")
    @classmethod
    def _ref_id_must_be_prefixed(cls, value: str) -> str:
        if not REF_ID_RE.match(value):
            raise ValueError(f"ref_id 必须匹配 {REF_ID_PATTERN}: {value!r}")
        return value


class Citation(BaseModel):
    """⑥ 草稿里的一条引用：模型声称"某句话由某篇文献支撑"。

    程序只信 `ref_id`：它必须出现在本次 ① 检索到的 `literature_refs` 里，否则
    ⑦ 的 `citations_subset` 判失败（"引用必须来自检索结果"，同 D-014"模型输出
    必须由程序校验"）。`claim` 是被引用支撑的那句话，长度上限 120 字，防止把整段
    正文塞进来（也避免 strict 模式下模型用超长句子绕过约束）。

    `chunk_id` 是 T-030 的块级引用（`"{ref_id}#c{序号}"`）：非空时 ⑦ 会校验该块
    属于所引文献且在本次索引块集合里；缺省 None 的旧数据行为不变。
    """

    ref_id: str
    claim: str = Field(max_length=120, description="该引用支撑的一句话（≤120 字）")
    # T-030 的 chunk 级引用；旧数据为 None，校验跳过
    chunk_id: str | None = None


class ResearchQuestion(BaseModel):
    """② 问题提炼的结构化输出。"""

    question: str = Field(description="可被数据检验的具体研究问题")
    variables: list[str] = Field(
        default_factory=list, description="参与分析的变量名，只能从数据集列中选择"
    )
    hypothesis: str = Field(default="", description="可证伪的假设")
    method: str = Field(default="", description="验证方法的简述")
    confidence: float = Field(default=0.0, description="0 到 1 之间的置信度")


class QualitySummary(BaseModel):
    """③ 数据质量摘要。

    ③ 拆成 prepare/review 两个节点后，审核节点只有 `cleaned_candidate` 可用，
    审核载荷里的 `raw_missing_rate` / `dropped_rows`（T-P0-4：审核人要看清洗前的
    原始数字）只能从这里取，所以这两个清洗口径的统计进了摘要，由预检统一填写。
    """

    rows: int = 0
    columns: int = 0
    missing_rate: float = 0.0
    duplicates_removed: int = 0
    # 清洗**前**的缺失率（判定口径）与被丢弃的行数，见类文档字符串
    raw_missing_rate: float = 0.0
    dropped_rows: int = 0
    notes: list[str] = Field(default_factory=list)


class CleanedDataRef(BaseModel):
    """③ 清洗后数据的引用（大文件只传引用、摘要与 checksum）。

    `path` 是相对 `run_dir` 的 POSIX 相对路径，读取时经
    `RunContext.artifact_path()` 解析。

    `raw_checksum` 是**原始输入 CSV** 的 sha256，不是本文件的：审核节点靠它确认
    "人审的是哪份原始数据"，暂停期间原始文件被改就能立刻发现。
    """

    path: str
    checksum: str = ""
    rows: int = 0
    columns: list[str] = Field(default_factory=list)
    quality: QualitySummary = Field(default_factory=QualitySummary)
    raw_checksum: str = ""

    @field_validator("path")
    @classmethod
    def _path_must_be_relative(cls, value: str) -> str:
        return ensure_relative_path(value)


class ParamSet(BaseModel):
    """④ 计算参数（Gate 1 为固定参数，不做参数流水线）。"""

    algorithm: str = "linear_regression"
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42


class ReviewRecord(BaseModel):
    """③ 审核记录，追加写入 review_log。

    `subject_checksum` 是被审候选文件的 sha256、`raw_checksum` 是原始 CSV 的：
    两者把审核结论绑定到具体字节上，事后才能回答"人审的到底是哪份数据"。
    审计日志只增不减，但每条带 `attempt`，回溯重审后能区分第几次审核。
    """

    stage: str
    decision: str
    reason: str = ""
    reviewer: str = "cli"
    # 被审文件相对 run_dir 的路径（P1-1）。打回时候选文件留在 run 目录里，
    # 只靠 checksum 无法在磁盘上定位它，manifest 的 review_log 必须能指回文件。
    subject_path: str = ""
    subject_checksum: str = ""
    raw_checksum: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    # 该审核由 data_review 第几次执行产出，由 @step 注入的 attempt 填入（P1-3）
    attempt: int = 1


class ComputeResult(BaseModel):
    """④ DIRECT 计算结果。"""

    algorithm: str
    metrics: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class ArtifactRef(BaseModel):
    """产物引用，进入 manifest。

    `path` 是相对 `run_dir` 的 POSIX 相对路径：manifest 因此可以在别的机器、
    别的目录、容器里原样复用，跨进程恢复（P1-1）也不会拿到失效的旧路径。

    `attempt` 是该产物由哪个节点第几次执行产出（P1-3）：state 里 `artifacts`
    按 `kind` 覆盖，回溯重跑后同一种产物只剩最新一条，旧的那条路径已被覆盖、
    checksum 已失效，必须能通过 attempt 区分。
    """

    kind: str
    path: str
    checksum: str = ""
    produced_by: str = ""
    attempt: int = 1

    @field_validator("path")
    @classmethod
    def _path_must_be_relative(cls, value: str) -> str:
        return ensure_relative_path(value)


class CheckResult(BaseModel):
    """⑦ 单条硬规则结果。

    `owner_step` 是这条检查归哪个节点负责（P1-8）：归属关系写在规则 YAML 的
    `owner_steps` 里，而不是写在代码里。Gate 2 的归因 Agent 据此判断失败后
    该回到哪一步。
    """

    name: str
    passed: bool
    detail: str = ""
    owner_step: str


class CitationStats(BaseModel):
    """⑦ 的引用统计，随报告进 `validation_report.json` 与 manifest（A6）。

    `total` 是草稿里 citation 条数（含重复 `ref_id` 各算一条），`invalid` 是
    `ref_id` 不在本次检索结果里的条数，`hallucination_rate = invalid / total`
    （`total == 0` 时为 0.0，不除零）。这是**校验前**的幻觉引用率：⑦ 拦住的是
    这些非法引用，但比例本身是模型的表现指标。
    """

    total: int = 0
    invalid: int = 0
    hallucination_rate: float = 0.0


class ValidationReport(BaseModel):
    """⑦ 硬规则校验报告。"""

    passed: bool = False
    checks: list[CheckResult] = Field(default_factory=list)
    summary: str = ""
    citation_stats: CitationStats = Field(default_factory=CitationStats)


class DraftContent(BaseModel):
    """⑥ LLM 生成的正文内容，由 Jinja2 渲染为 Markdown。

    `citations` 默认空列表（`default_factory`），旧 checkpoint 与旧 Fake 构造的
    `DraftContent` 反序列化/校验后落回空列表，向后兼容（A1）。不要写成"可为 None
    且默认值非 None"的字段（D-006）。
    """

    background: str = Field(description="研究背景与问题来源")
    findings: str = Field(description="基于计算结果的关键发现")
    limitation: str = Field(description="局限与待扩展之处")
    citations: list[Citation] = Field(
        default_factory=list, description="正文用到的文献引用（ref_id 必须来自本次检索结果）"
    )


class ErrorRecord(BaseModel):
    """统一的错误记录：进入 state 与 manifest，供 Gate 2 归因与回溯使用。

    `retryable` / `backtrack_to` 由 `@step` 装饰器从异常类的类属性填入（P1-8）：
    前者回答"原地重试有没有可能成功"，后者回答"换起点重跑默认回到哪一步"。
    """

    code: str = "RF-000"
    step: str = ""
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 1
    retryable: bool = False
    backtrack_to: str | None = None


class Termination(BaseModel):
    """链路终止语义，取代原先的字符串 terminal_reason。"""

    kind: Literal["rejected", "empty", "error", "budget_exhausted", "escalated"]
    step: str = ""
    reason: str = ""
    error: ErrorRecord | None = None


class ReviewDecision(BaseModel):
    """③ 人工审核决策，仅接受 approve / reject。"""

    decision: Literal["approve", "reject"]
    reason: str = ""
    reviewer: str = "cli"


class LLMUsage(BaseModel):
    """单次或汇总的 LLM 用量。

    `total_tokens` 恒等于 `prompt_tokens + completion_tokens`，由校验器强制，
    调用方只需给出前两项。`attempts` 为实际发起的请求次数（含结构化重试）。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 1

    @model_validator(mode="after")
    def _sync_total_tokens(self) -> LLMUsage:
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class LLMCallRecord(BaseModel):
    """一次 `complete()` 的用量记录，追加进 state 的 llm_calls。

    `duration_ms` 是这次调用（含结构化重试的全部尝试）的墙钟耗时；`attempt`
    是所属节点第几次执行（与 `StepRecord.attempt` 同源，由 `@step` 注入）。
    两者都是 T-025 第 2 轮新增，旧 checkpoint 里没有，缺失时为 None：trace
    生成按 `(step, attempt)` 挂接，缺 `attempt` 的旧记录退回按出现顺序。
    """

    step: str
    response_model: str
    usage: LLMUsage
    # 结构化的取值方式："json_schema" | "json_object"；chat() 记 None。
    # 供以后按模式统计"首轮合法率"（attempts == 1 的比例）。
    structured_output: str | None = None
    duration_ms: int | None = None
    attempt: int | None = None


class LLMTurnTrace(BaseModel):
    """Agent 工具回路里一次 `chat()` 的追溯记录，进入 `AttributionRecord` /
    `RetrievalTrace.llm_turns`。

    归因 / 检索 Agent 的模型调用不在 `llm_calls` 里（回路本身不写 state，见 D-006），
    而是由 `run_tool_loop` 逐轮记录下来交给节点写入所属记录。`turn` 从 1 起，
    与同轮触发的工具调用（`ToolCallTrace.turn`）配对，trace 里按发生顺序交错。
    """

    turn: int
    usage: LLMUsage
    duration_ms: int | None = None


class BacktrackPatch(BaseModel):
    """回溯时的参数 patch（白名单）。

    `extra="forbid"`：Agent 只能改这里列出的参数，未知字段一律拒绝，防止它把
    任意键写进 `param_set`。字段与目标节点的匹配关系由 `validate_for()` 强制：
    `y_outlier_*` 只能用于 `data_prepare`，`target_transform` / `exclude_features`
    只能用于 `compute`。

    `test_size` / `seed` 已从白名单移除（靠换随机划分过线是凑分，不是修复）。
    它们仍保留为字段，只为了让 `validate_for` 能给出"不允许通过改变随机划分"这条
    针对性报错——对任何 `target_step` 都会被拒绝。
    """

    model_config = {"extra": "forbid"}

    y_outlier_strategy: Literal["none", "iqr"] | None = None  # 只允许 target=data_prepare
    y_outlier_k: float | None = Field(default=None, ge=1.5, le=10)  # 只允许 target=data_prepare
    target_transform: Literal["none", "log"] | None = None  # 只允许 target=compute
    exclude_features: list[str] | None = None  # 只允许 target=compute
    # 已从白名单移除：任何 target 都不接受，validate_for 给出针对性报错。
    test_size: float | None = Field(default=None, gt=0, lt=0.5)
    seed: int | None = None

    def validate_for(self, target_step: str, *, features: list[str] | None = None) -> None:
        """校验 patch 字段是否都允许用于 `target_step`，不匹配抛 `ValueError`。

        `features` 用于校验 `exclude_features` 必须是当前特征的真子集（至少保留
        1 个特征）；不传则跳过子集校验（仍校验字段与目标节点的归属）。
        """
        data_prepare_fields = ("y_outlier_strategy", "y_outlier_k")
        compute_fields = ("target_transform", "exclude_features")
        removed_fields = ("test_size", "seed")
        # 与"不改"等价的取值视为未提供：strict 模式下模型必须填满所有字段，常会把
        # 不相关的字段填成 "none" / []（真实评测 dev 集上 2/13 条因此被误判为越权）。
        no_op = {"y_outlier_strategy": "none", "target_transform": "none", "exclude_features": []}
        given = {
            name
            for name, value in self.model_dump().items()
            if value is not None and not (name in no_op and value == no_op[name])
        }

        # 已从白名单移除的字段：任何 target 都拒绝，并给出针对性报错。
        removed = sorted(given & set(removed_fields))
        if removed:
            raise ValueError(
                "不允许通过改变随机划分来通过校验（test_size/seed 已从白名单移除）: "
                + ", ".join(removed)
            )

        if target_step == "data_prepare":
            allowed = set(data_prepare_fields)
        elif target_step == "compute":
            allowed = set(compute_fields)
        else:
            raise ValueError(
                f"未知的 target_step: {target_step!r}（只能是 data_prepare 或 compute）"
            )
        forbidden = sorted(given - allowed)
        if forbidden:
            raise ValueError(f"target_step={target_step} 不允许以下字段: {', '.join(forbidden)}")

        if target_step == "compute" and self.exclude_features is not None and features is not None:
            self._validate_exclude_features(features)

    def _validate_exclude_features(self, features: list[str]) -> None:
        """`exclude_features` 必须是 `features` 的真子集（至少保留 1 个特征）。"""
        feature_set = set(features)
        excluded = list(self.exclude_features or [])
        missing = [name for name in excluded if name not in feature_set]
        if missing:
            raise ValueError(
                f"exclude_features 含不存在的特征列: {', '.join(missing)}"
                f"（当前特征: {', '.join(features)}）"
            )
        if not feature_set - set(excluded):
            raise ValueError("exclude_features 不能排除全部特征（至少保留 1 个）")


class AttributionDecision(BaseModel):
    """一次归因决策：回到哪一步、改什么，或转人工。"""

    action: Literal["backtrack", "escalate"]
    # backtrack 时必填，且只能是 data_prepare 或 compute；
    # escalate 时可选，表示"诊断出问题在哪一步"（但没有可调的参数）
    target_step: str | None = None
    patch: BacktrackPatch = Field(default_factory=BacktrackPatch)
    reason: str
    # 谁做的决定：rule（规则层）或 agent（归因 Agent）
    source: Literal["rule", "agent"]


class ToolCallTrace(BaseModel):
    """一次工具调用的追溯记录，进入 `AttributionRecord` 与 `RetrievalTrace`。

    由 `run_tool_loop` 从 `LoopResult.invocations` 生成：`arguments_digest` 是参数
    JSON 的前 200 字符（路径已脱敏），`duration_ms` 取自 `ToolResult.duration_ms`。
    `turn` 是触发这次调用的 `chat()` 轮次（1 起），让 trace 能把 llm 与 tool 按
    发生顺序交错。不含任何 Key 与业务空间 ID，可安全进 manifest 与 trace.json。
    """

    name: str
    arguments_digest: str = ""
    ok: bool
    error_code: str | None = None
    duration_ms: int = 0
    turn: int = 1


class AttributionRecord(BaseModel):
    """每次进入 `attribute` 节点追加一条的归因审计记录。

    `tool_calls` 是本次 Agent 回路的逐次工具调用追溯（T-025）；原来的调用次数
    直接取 `len(tool_calls)`，不再单独维护计数字段。`llm_turns` 是同一回路里每次
    `chat()` 的用量与耗时（T-025 第 2 轮）：归因 Agent 的模型调用不在 `llm_calls`，
    成本与 token 统计必须靠它。
    """

    attempt: int
    trigger: Literal["error", "validation_failed"]
    failed_checks: list[str] = Field(default_factory=list)
    error_code: str | None = None
    decision: AttributionDecision
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    llm_turns: list[LLMTurnTrace] = Field(default_factory=list)
    usage: LLMUsage | None = None
    # 本次决策是否发生过"Agent 调用失败降级"（T-027）：归因评测据此把失败用例
    # 单列并排除出归因准确率 / 恢复率的分母。
    agent_failed: bool = False
    # 脱敏后的失败原因（如 "LLMError: 真实 LLM 调用失败: ..."）：不含 Key 与绝对
    # 路径，只有错误类型名与一句消息，让事后能看见"为什么 agent_error"。
    failure_reason: str | None = None


class StepRecord(BaseModel):
    """单步执行记录：Gate 2 计时、重试计数与 span 的挂载点。

    每次节点执行产出一条，追加进 state 的 `step_records`（列表追加，回溯重跑
    会多一条且 `attempt` 递增）；`step_status` 只保留每个节点的最新结论，两者
    语义不同且写入方唯一（`@step` 装饰器），不会互相漂移。

    放在 `LLMUsage` 之后定义，因为 `llm_usage` 引用它。
    """

    step: str
    # ok | error | rejected 等，与 step_status 的值一致
    status: str
    # 该节点第几次执行，从 1 开始
    attempt: int = 1
    # ISO 8601，毫秒级精度，带时区（T-025 第 2 轮起）
    started_at: str
    duration_ms: int
    error: ErrorRecord | None = None
    # 本次执行内全部 LLM 调用的合计；没调用为 None
    llm_usage: LLMUsage | None = None


class ToolCall(BaseModel):
    """模型提出的一次工具调用意图。

    `arguments` 原样保留模型给的 JSON 字符串，解析交给 `ToolRegistry.invoke`——
    参数不合法也要以 tool message 回填给模型，不能在解析阶段就抛异常。
    """

    id: str
    name: str
    arguments: str


class AssistantTurn(BaseModel):
    """一次 `chat()` 的模型回复：正文与工具调用意图可能同时出现。

    OpenAI 在有 `tool_calls` 时 `content` 常为 `null`，所以它是 `str | None`。
    """

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)


class RetrievalDecision(BaseModel):
    """一次检索 Agent 决策：选中的文献、实际用过的查询、停止原因与用量。

    `stop_reason` 取值：
    - `agent`：模型调用了 `finish_retrieval` 并成功选中文献（或没调 finish 就结束、
      一次都没尝试过 finish）；
    - `budget_exhausted`：回路预算耗尽，`refs` 回退为已检索结果里排在最前的若干条；
    - `agent_error`：Agent 自身出错（如 `LLMError`），`refs` 回退为一次课题原文检索；
    - `invalid_selection`（T-027）：模型调用过 `finish_retrieval` 但选的 ref_id 全不
      在已检索集合里、最终没能选出任何文献——不再伪装成"Agent 判断不选"。
    """

    refs: list[LiteratureRef] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    stop_reason: Literal["agent", "budget_exhausted", "agent_error", "invalid_selection"]
    usage: LLMUsage = Field(default_factory=LLMUsage)


class RerankTrace(BaseModel):
    """在线重排（T-034）的追溯：候选、查询集合、重排前后顺序与合并动作。

    `reason` 非 None 表示"配置开启但这次没有执行重排"（无 Agent、Agent 降级、
    没有候选），此时 `before_ids` / `after_ids` 只记录最终沿用的顺序。
    `duration_ms` / `prompt_tokens` / `embedding_model` 供 tracing 生成 rerank span
    并按嵌入模型查价格表，不改变重排本身的排序逻辑。
    """

    enabled: bool = False
    candidates: int = 0
    queries: list[str] = Field(default_factory=list)
    before_ids: list[str] = Field(default_factory=list)
    after_ids: list[str] = Field(default_factory=list)
    agent_pick_kept: bool = False
    replaced_ref_id: str | None = None
    reason: str | None = None
    duration_ms: int | None = None
    prompt_tokens: int = 0
    embedding_model: str | None = None


class RetrievalTrace(BaseModel):
    """检索 Agent 的运行轨迹，进入 state 与 manifest。

    不含 API Key 与业务空间 ID：只记录查询词、停止原因、选中的 ref_id 与用量，
    都是可审计、可复现、不泄露凭证的信息。
    """

    queries: list[str] = Field(default_factory=list)
    # no_agent（T-034）：没有注入检索 Agent 时，为了记录"重排为何没执行"仍产出一条
    # trace；不开启在线重排时该路径仍然不产出 trace（trace=None，旧行为不变）。
    stop_reason: Literal[
        "agent", "budget_exhausted", "agent_error", "invalid_selection", "no_agent"
    ]
    selected_ref_ids: list[str] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    # 本次 Agent 回路的逐次工具调用追溯（T-025），由 run_tool_loop 生成
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    # 本次 Agent 回路每轮 chat() 的用量与耗时（T-025 第 2 轮），与 tool_calls 按 turn 交错
    llm_turns: list[LLMTurnTrace] = Field(default_factory=list)
    # 脱敏后的失败原因（T-027）：stop_reason 为 agent_error / invalid_selection 时
    # 记录"为什么"，不含 Key 与绝对路径。
    failure_reason: str | None = None
    # 被注入启发式标记为 suspicious 的 ref_id（T-033）：只标记不丢弃，供注入评测
    # 统计召回与误报。默认空列表：旧 checkpoint 缺该字段时反序列化落回空列表（A6）。
    suspicious_ref_ids: list[str] = Field(default_factory=list)
    # 在线重排的追溯（T-034）。None = 未开启，旧 checkpoint 缺该字段时同样为 None。
    rerank: RerankTrace | None = None

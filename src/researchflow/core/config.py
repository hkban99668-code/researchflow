"""配置加载：configs/settings.yaml + 环境变量（.env）。

优先级：环境变量 > settings.yaml > 默认值。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

from .errors import ArtifactError, ConfigError
from .schemas import ArtifactRef, CleanedDataRef, ensure_relative_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings.yaml"

# 结构化输出的两种取值方式：json_schema 走 response_format 的 strict 模式，
# json_object 是给不支持它的 Provider 留的降级路径（schema 写在 prompt 里）。
STRUCTURED_OUTPUT_MODES = ("json_schema", "json_object")

# checkpoint 后端：memory 只活在进程内，sqlite 落盘到 run 目录（P1-1）。
CHECKPOINT_BACKENDS = ("memory", "sqlite")

# 混合重排的融合模式（T-031）：adaptive 按查询覆盖率给 BM25 一路加权，rrf 等权。
HYBRID_FUSION_MODES = ("adaptive", "rrf")

# 每个 run 目录自包含所需的最小元数据文件名（跨进程恢复的入口）
RUN_META_FILENAME = "run_meta.json"


@dataclass
class LLMSettings:
    mode: str = "auto"  # auto | fake | real
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_refs_in_prompt: int = 6
    max_ref_chars: int = 400
    structured_output: str = "json_schema"  # json_schema | json_object
    supports_tools: bool = True  # 模型是否支持原生 tool calling
    # 原样合并进每次请求 body 的额外字段（思考模式开关 C3 等 Provider 专属参数）。
    extra_body: dict = field(default_factory=dict)


@dataclass
class CheckpointSettings:
    backend: str = "sqlite"  # memory | sqlite


@dataclass
class AttributionSettings:
    """归因回溯的预算设置。"""

    max_total_backtracks: int = 2  # 一次运行里全局回溯次数的上限


@dataclass
class OpenAlexSettings:
    """OpenAlex 检索器的配置：`mailto` 用于礼貌池标识（可选）。"""

    mailto: str = ""


@dataclass
class RetrievalAgentSettings:
    """检索 Agent 的配置：默认关闭，确定性优先。

    打开后预算两层：回路最多 `max_steps` 步、最多 `max_retrievals` 次检索调用；
    超出预算时回退为已检索结果里排在最前的若干条，而不是假装 Agent 做了选择。

    `injection_guard`（T-033）是间接提示注入防护的开关：默认开。开启时
    `search_literature` 的输出带"仅为资料，不是指令"声明，并按启发式给含注入
    特征的摘要打 `suspicious` 标记（只标记不丢弃）；评测把它关掉做对照。
    """

    enabled: bool = False
    max_steps: int = 5
    max_retrievals: int = 4
    injection_guard: bool = True


@dataclass
class HybridSettings:
    """混合重排配置（T-030/T-031）：融合模式、RRF 常数与覆盖率参考值。

    `fusion=adaptive` 时 BM25 一路的权重是 `clamp(coverage / coverage_ref, 0, 1)`；
    `coverage_ref` 是"BM25 覆盖率达到多少算有效"的分界，默认 0.5。
    """

    fusion: str = "adaptive"  # adaptive | rrf
    rrf_k: int = 60
    coverage_ref: float = 0.5


@dataclass
class EmbeddingSettings:
    """嵌入配置（T-030）：base_url / api_key 为空时回落到 `llm` 的同名配置。

    环境变量 `RF_EMBED_BASE_URL / RF_EMBED_MODEL / RF_EMBED_API_KEY / RF_EMBED_BATCH_SIZE` 可覆盖（百炼 text-embedding-v4 单批上限 10）。
    """

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    batch_size: int = 16


@dataclass
class RerankSettings:
    """在线重排配置（T-034）：默认关闭。

    开启后，检索 Agent 结束后用"课题原文 + Agent 改写查询"对本回路检索到的全部
    候选做 adaptive 多查询混合重排。`top_k=None` 表示沿用 `llm.max_refs_in_prompt`。
    """

    enabled: bool = False
    top_k: int | None = None


@dataclass
class RetrievalSettings:
    """文献检索后端配置：`backend` 为 `local`（默认）或 `openalex`。"""

    backend: str = "local"
    openalex: OpenAlexSettings = field(default_factory=OpenAlexSettings)
    agent: RetrievalAgentSettings = field(default_factory=RetrievalAgentSettings)
    hybrid: HybridSettings = field(default_factory=HybridSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT
    seed: int = 42
    runs_dir: Path = PROJECT_ROOT / "runs"
    llm: LLMSettings = field(default_factory=LLMSettings)
    checkpoint: CheckpointSettings = field(default_factory=CheckpointSettings)
    attribution: AttributionSettings = field(default_factory=AttributionSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    raw: dict = field(default_factory=dict)

    def resolve_path(self, relative: str | Path) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (self.project_root / path)

    def resolve_llm_mode(self) -> str:
        """把 auto 解析为 fake 或 real；real 缺少 Key 时直接报错。"""
        if self.llm.mode == "fake":
            return "fake"
        if self.llm.mode == "real":
            if not self.llm.api_key:
                raise ConfigError("RF_LLM_MODE=real 但未配置 RF_LLM_API_KEY")
            return "real"
        return "real" if self.llm.api_key else "fake"


def relative_to_root(path: Path, root: Path) -> str | None:
    """把 `root` 下的路径转成相对 `root` 的 POSIX 字符串；不在 `root` 下时返回 None。

    用于两类"要进产物但不该泄露安装位置"的路径：manifest 的 `run_dir`
    （返回 None 就写 null）和 ③ 审核载荷里的 `dataset` / `rules`
    （返回 None 说明是用户给的外部路径，保留原样）。
    """
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def sha256_file(path: str | Path) -> str:
    """计算文件 sha256：产物 checksum 的唯一算法。

    放在 core 而不是 `tools/quality.py`，是因为 `RunContext.read_verified`（P2-2）
    要用它，而 core 不能反向依赖上层；上层统一从这里导入，避免两份实现漂移。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunContext:
    """单次运行的上下文：不可序列化，经图的 `context_schema` 逐次调用注入。

    它**不进 checkpoint**（P1-1）：LLM 客户端里带着 API Key，检索器与 run_dir 也
    都能从配置重建。恢复时由 `from_run_id()` 读 `run_meta.json` 现造一份。
    """

    settings: Settings
    run_dir: Path
    llm: Any = None
    # 文献检索器（`core/retrieval.py::Retriever`）；类型写成 Any，和 llm 一样不做耦合
    retriever: Any = None
    # 归因 Agent（`core/attribution.py::Attributor`）；为 None 时 attribute 节点对
    # validation_failed 一律转人工。由 main.py 构造，不进 checkpoint。
    attributor: Any = None
    # 检索 Agent（`core/retrieval_agent.py::RetrievalAgent`）；为 None 时 ① 节点走
    # 原文一次检索。由 main.py 按配置构造，不进 checkpoint。
    retrieval_agent: Any = None
    # 在线重排器（`tools/hybrid.py::HybridReranker`）；为 None 时 ① 节点不重排。
    # 由 main.py 在 `retrieval.rerank.enabled=true` 时构造，不进 checkpoint。
    reranker: Any = None

    @classmethod
    def from_run_id(cls, run_id: str, settings: Settings, *, mode: str | None = None) -> RunContext:
        """只凭 run_id 与磁盘文件重建上下文（跨进程恢复的入口）。

        - run 目录或 `run_meta.json` 不存在 → `ConfigError`（不是可恢复的运行）。
        - 默认用 `run_meta.json` 里记录的 `llm_mode` 恢复：Fake 运行就用 Fake 恢复，
          不受 `.env` 影响（B17）。显式给 `mode` 且与记录不一致 → `ConfigError`：
          同一个 run 前后用两种模型会让产物无法解释，宁可停也不要悄悄换。
        - 模型名、结构化输出方式的一致性（T-009）只在 real 模式下校验；Fake 不校验。
        - API Key **只从当前进程的环境变量读**，不落盘也不从文件读。
        """
        run_dir = settings.runs_dir / run_id
        meta_path = run_dir / RUN_META_FILENAME
        if not meta_path.exists():
            raise ConfigError(f"无法恢复 run {run_id}：{meta_path} 不存在")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded_mode = meta.get("llm_mode")
        if mode is not None and mode != recorded_mode:
            raise ConfigError(
                f"无法恢复 run {run_id}：--mode 与首次运行模式不一致"
                f"（首次运行为 {recorded_mode!r}，--mode 为 {mode!r}）"
            )
        effective_mode = mode if mode is not None else recorded_mode

        meta_llm = meta.get("settings", {}).get("llm", {})
        if effective_mode == "real":
            if settings.llm.model != meta_llm.get("model"):
                raise ConfigError(
                    f"无法恢复 run {run_id}：llm.model 不一致"
                    f"（首次运行为 {meta_llm.get('model')!r}，当前为 {settings.llm.model!r}）"
                )
            if settings.llm.structured_output != meta_llm.get("structured_output"):
                raise ConfigError(
                    f"无法恢复 run {run_id}：llm.structured_output 不一致"
                    f"（首次运行为 {meta_llm.get('structured_output')!r}，"
                    f"当前为 {settings.llm.structured_output!r}）"
                )

        # 把生效的模式写回 settings，让 build_llm_client 构造出正确的适配器
        # （Fake 恢复不依赖 .env 里的真实模式配置）。
        settings.llm.mode = effective_mode

        # 延迟导入：`core/llm.py` 与 `tools/literature.py` 都依赖本模块，
        # 放在模块级 import 会成环。
        from ..tools.literature import build_retriever
        from .llm import build_llm_client

        return cls(
            settings=settings,
            run_dir=run_dir,
            llm=build_llm_client(settings),
            retriever=build_retriever(settings),
        )

    def artifact_path(self, rel: str) -> Path:
        """把相对 `run_dir` 的路径解析成绝对路径。

        这是 state 里路径的**唯一解析入口**：节点读写 run 目录下的文件都必须
        经过它，以后加 checksum 校验（P2-2）也只改这一处。
        """
        try:
            ensure_relative_path(rel)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        return self.run_dir / rel

    def read_verified(self, ref: CleanedDataRef | ArtifactRef) -> Path:
        """解析产物路径**并校验 checksum**，一致才返回（P2-2）。

        跨进程恢复（P1-1）之后，state 里的引用可能指向已被覆盖或改动的文件。
        静默读取会让"审核过的"和"实际用的"脱钩，所以这里一律先比 sha256：
        不一致抛 `ArtifactError`（RF-005），`detail` 带期望值与实际值。

        `ref.checksum` 为空串时也拒绝：那是没写全的引用，不是"免校验"。

        文件不存在时同样抛 RF-005（而不是让 `FileNotFoundError` 冒出去）：
        跨进程恢复后暂停几天、期间产物被移走是现实情况，它属于"产物已变化"。
        """
        path = self.artifact_path(ref.path)
        if not path.exists():
            raise ArtifactError(
                f"产物文件不存在，视为已变化: {ref.path}", detail={"path": ref.path}
            )
        if not ref.checksum:
            raise ArtifactError(
                f"产物引用缺少 checksum，拒绝读取: {ref.path}", detail={"path": ref.path}
            )
        actual = sha256_file(path)
        if actual != ref.checksum:
            raise ArtifactError(
                f"产物已被改动，拒绝读取: {ref.path}",
                detail={"path": ref.path, "expected": ref.checksum, "actual": actual},
            )
        return path

    def relative_to_run(self, path: Path) -> str:
        """把 `run_dir` 下的绝对路径转成相对 `run_dir` 的 POSIX 字符串。"""
        try:
            return Path(path).resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise ArtifactError(f"路径不在 run 目录下: {path}") from exc


# extra_body 里键名含这些词（不区分大小写）的项视为敏感，落盘前替换成 ***。
SENSITIVE_EXTRA_BODY_TERMS = ("key", "token", "secret", "authorization")


def sanitize_extra_body(extra_body: dict[str, Any]) -> dict[str, Any]:
    """把 `extra_body` 里键名含敏感词的项替换成 `***`。

    与 `base_url` 脱敏同级：`extra_body` 会随 `asdict(settings.llm)` 写进
    `run_meta.json`（属于"这次运行用什么配置"，复现需要它），但若有人往里放
    `api_key` / `token` 之类的字段就不能原样落盘。
    """
    return {
        key: ("***" if any(term in key.lower() for term in SENSITIVE_EXTRA_BODY_TERMS) else value)
        for key, value in extra_body.items()
    }


def sanitize_base_url(url: str) -> str:
    """把 base_url 脱敏成 `scheme://*.<主域>`：去 userinfo、去路径与端口，第一段子域名换成 *。

    专属地址的第一段子域名常是业务空间 ID，能定位到账号（D-012），所以一并遮掉；
    `from_run_id` 本来也不读 base_url，脱敏不影响跨进程恢复。
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) >= 2:
        parts[0] = "*"
        host = ".".join(parts)
    return f"{parsed.scheme or 'https'}://{host}"


def _env_flag(name: str, default: Any) -> bool:
    """读布尔型环境变量；`0` / `false` / `no` / 空串一律视为 False。"""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _git_commit(root: Path) -> str | None:
    """当前 HEAD；不在 git 仓库里或 git 不可用时返回 None。"""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = done.stdout.strip()
    return commit or None


def run_meta_payload(
    *,
    run_id: str,
    topic_path: Path,
    llm_mode: str,
    settings: Settings,
    checkpoint_backend: str,
) -> dict[str, Any]:
    """组装 `run_meta.json`：重建上下文所需的最小信息。

    两条硬约束（D-009）：**不含 API Key**，**不含项目根目录的绝对路径**。
    `runs_dir` 与 `topic_path` 都先转成相对项目根目录的路径，转不了（在根目录
    之外，是用户给的外部路径）`topic_path` 才保留原值并只在终端显示。
    """
    root = settings.project_root
    # 去掉 api_key：Key 只从恢复进程的环境变量读，绝不落盘；base_url 脱敏后写（D-012）。
    llm_meta = {key: value for key, value in asdict(settings.llm).items() if key != "api_key"}
    llm_meta["base_url"] = sanitize_base_url(settings.llm.base_url)
    llm_meta["extra_body"] = sanitize_extra_body(llm_meta.get("extra_body") or {})
    return {
        "run_id": run_id,
        "topic_path": relative_to_root(topic_path, root) or str(topic_path),
        "llm_mode": llm_mode,
        "settings": {
            "seed": settings.seed,
            "runs_dir": relative_to_root(settings.runs_dir, root),
            "llm": llm_meta,
            "paths": settings.raw.get("paths") or {},
        },
        "checkpoint_backend": checkpoint_backend,
        "git_commit": _git_commit(root),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_run_meta(run_dir: Path, payload: dict[str, Any]) -> Path:
    """把 `run_meta.json` 写进 run 目录，返回文件路径。"""
    path = run_dir / RUN_META_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_settings(
    project_root: Path | None = None,
    settings_path: Path | None = None,
    load_env_file: bool | None = None,
) -> Settings:
    root = project_root or PROJECT_ROOT
    # 测试与 .env 隔离（T-009）：默认仍读 .env；显式传 load_env_file 或设
    # RF_LOAD_DOTENV=0/false/no 时跳过，避免测试被开发者 shell 或仓库根目录的
    # 真实 Key 影响。
    if load_env_file is None:
        load_env_file = _env_flag("RF_LOAD_DOTENV", default=True)
    if load_env_file:
        load_dotenv(root / ".env")

    path = settings_path or root / "configs" / "settings.yaml"
    if not path.exists():
        raise ConfigError(f"缺少配置文件: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    llm_raw = raw.get("llm") or {}
    paths_raw = raw.get("paths") or {}

    extra_body = llm_raw.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        raise ConfigError(f"settings.yaml 的 llm.extra_body 必须是对象: {extra_body!r}")
    env_extra = os.getenv("RF_LLM_EXTRA_BODY")
    if env_extra:
        try:
            parsed_extra = json.loads(env_extra)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"RF_LLM_EXTRA_BODY 不是合法 JSON: {exc.msg}") from exc
        if not isinstance(parsed_extra, dict):
            raise ConfigError(
                f"RF_LLM_EXTRA_BODY 必须是 JSON 对象，收到 {type(parsed_extra).__name__}"
            )
        extra_body = {**extra_body, **parsed_extra}

    llm = LLMSettings(
        mode=str(os.getenv("RF_LLM_MODE", llm_raw.get("mode", "auto"))),
        base_url=str(
            os.getenv("RF_LLM_BASE_URL", llm_raw.get("base_url", "https://api.openai.com/v1"))
        ),
        api_key=os.getenv("RF_LLM_API_KEY") or llm_raw.get("api_key") or None,
        model=os.getenv("RF_LLM_MODEL") or llm_raw.get("model") or None,
        timeout_seconds=float(
            os.getenv("RF_LLM_TIMEOUT_SECONDS", llm_raw.get("timeout_seconds", 60))
        ),
        temperature=float(llm_raw.get("temperature", 0.0)),
        max_refs_in_prompt=int(llm_raw.get("max_refs_in_prompt", 6)),
        max_ref_chars=int(llm_raw.get("max_ref_chars", 400)),
        structured_output=str(
            os.getenv("RF_LLM_STRUCTURED_OUTPUT", llm_raw.get("structured_output", "json_schema"))
        ),
        supports_tools=_env_flag("RF_LLM_SUPPORTS_TOOLS", llm_raw.get("supports_tools", True)),
        extra_body=extra_body,
    )
    if llm.structured_output not in STRUCTURED_OUTPUT_MODES:
        raise ConfigError(
            f"RF_LLM_STRUCTURED_OUTPUT 取值非法: {llm.structured_output!r}"
            f"（只能是 {' / '.join(STRUCTURED_OUTPUT_MODES)}）"
        )

    backend = str(
        os.getenv("RF_CHECKPOINT_BACKEND", raw.get("checkpoint", {}).get("backend", "sqlite"))
    )
    if backend not in CHECKPOINT_BACKENDS:
        raise ConfigError(
            f"checkpoint.backend 取值非法: {backend!r}（只能是 {' / '.join(CHECKPOINT_BACKENDS)}）"
        )

    runs_dir = Path(paths_raw.get("runs_dir", "runs"))
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir

    attribution_raw = raw.get("attribution") or {}
    retrieval_raw = raw.get("retrieval") or {}
    openalex_raw = retrieval_raw.get("openalex") or {}
    retrieval_agent_raw = retrieval_raw.get("agent") or {}
    rerank_raw = retrieval_raw.get("rerank") or {}
    hybrid_raw = retrieval_raw.get("hybrid") or {}
    embedding_raw = raw.get("embedding") or {}
    hybrid_fusion = str(hybrid_raw.get("fusion", "adaptive"))
    if hybrid_fusion not in HYBRID_FUSION_MODES:
        raise ConfigError(
            f"retrieval.hybrid.fusion 取值非法: {hybrid_fusion!r}"
            f"（只能是 {' / '.join(HYBRID_FUSION_MODES)}）"
        )
    hybrid_coverage_ref = float(hybrid_raw.get("coverage_ref", 0.5))
    if hybrid_coverage_ref <= 0:
        raise ConfigError(f"retrieval.hybrid.coverage_ref 必须大于 0: {hybrid_coverage_ref!r}")
    rerank_top_k_raw = rerank_raw.get("top_k")
    rerank_top_k = int(rerank_top_k_raw) if rerank_top_k_raw is not None else None
    if rerank_top_k is not None and rerank_top_k <= 0:
        raise ConfigError(f"retrieval.rerank.top_k 必须大于 0: {rerank_top_k!r}")
    return Settings(
        project_root=root,
        seed=int(raw.get("project", {}).get("seed", 42)),
        runs_dir=runs_dir,
        llm=llm,
        checkpoint=CheckpointSettings(backend=backend),
        attribution=AttributionSettings(
            max_total_backtracks=int(attribution_raw.get("max_total_backtracks", 2))
        ),
        retrieval=RetrievalSettings(
            backend=str(os.getenv("RF_RETRIEVER", retrieval_raw.get("backend", "local"))),
            openalex=OpenAlexSettings(mailto=str(openalex_raw.get("mailto") or "")),
            agent=RetrievalAgentSettings(
                enabled=bool(retrieval_agent_raw.get("enabled", False)),
                max_steps=int(retrieval_agent_raw.get("max_steps", 5)),
                max_retrievals=int(retrieval_agent_raw.get("max_retrievals", 4)),
                injection_guard=bool(retrieval_agent_raw.get("injection_guard", True)),
            ),
            hybrid=HybridSettings(
                fusion=hybrid_fusion,
                rrf_k=int(hybrid_raw.get("rrf_k", 60)),
                coverage_ref=hybrid_coverage_ref,
            ),
            rerank=RerankSettings(
                enabled=bool(rerank_raw.get("enabled", False)),
                top_k=rerank_top_k,
            ),
        ),
        embedding=EmbeddingSettings(
            base_url=str(os.getenv("RF_EMBED_BASE_URL", embedding_raw.get("base_url", ""))),
            model=str(os.getenv("RF_EMBED_MODEL", embedding_raw.get("model", ""))),
            api_key=os.getenv("RF_EMBED_API_KEY") or embedding_raw.get("api_key") or "",
            batch_size=int(os.getenv("RF_EMBED_BATCH_SIZE", embedding_raw.get("batch_size", 16))),
        ),
        raw=raw,
    )

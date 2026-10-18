# ResearchFlow Gate 1 框架评审：优化项与 Gate 3 扩展性

- 评审范围：`researchflow/` 全部 `src/`（约 1500 行）、README §1/§3.5–5/§9/§10、`configs/`、设计方案中 Gate 2/3 边界条目
- 方式：只读，未运行测试，未改代码
- 依赖版本（`uv.lock`）：langgraph 1.2.12、langgraph-checkpoint 4.2.0、pydantic 2.13.5
- 行号均相对 `src/researchflow/`

## 结论

骨架方向是对的：单向分层、Pydantic 契约、产物 + checksum、Fake/Real 同契约，这些都能直接带进 Gate 3。**但是有三类问题会在 Gate 2/3 引入持久化、回溯和 MODULE 时变成结构性返工**，建议在 Gate 1 收尾时先把接口改好（只改接口和数据形状，不做功能）：

1. **失败路径**：异常绕过图和 manifest，终止语义只覆盖 ③。回溯的前提是"错误在 state 里"，现在不在。
2. **状态形状**：`append_list` 用在了一旦重跑就会重复的字段上，而回溯就意味着重跑。
3. **硬编码耦合**：checkpointer、计算算法、图表回算公式、检索实现都在调用点写死，没有可替换的接缝。

另有三处会导致**静默误判**的 Gate 1 缺陷（P0-2/3/4），建议在打 `v0.1.0-mvp` 标签前修掉。

---

## P0：Gate 1 收尾前应修（正确性 / 静默误判）

### P0-1 业务异常绕过图与 manifest，失败 run 没有可定位记录

- **证据**：`main.py:123` 的 `graph.invoke` 抛出 `ResearchFlowError` 后直接跳到 `main.py:166-168`，`_write_manifest`（`main.py:131`）不会执行。③ 的 `precheck` 失败在 interrupt 之前就抛出（`tools/quality.py:53/57/63/69`），④ 抛出于 `compute/direct.py:29/33`。
- **问题**：失败的 run 只留下一个 stderr JSON，`runs/<run_id>/` 里没有 manifest、没有 step_status，与设计方案"DIRECT 计算失败：记录异常、输入摘要和节点状态"（方案 793 行）和 README §9.5 的要求不符。更关键的是，Gate 2 的归因/回溯需要"错误是 state 的一部分"，现在错误只存在于 Python 调用栈里。
- **建议**（两层，推荐两层都做）：
  1. CLI 兜底：`run()` 用 `try/finally`，失败时从 `graph.get_state(config).values` 取当前快照 + `exc.as_dict()` 写 manifest（`status: "error"`）。
  2. 节点层收敛：加一个 `@step("compute")` 装饰器，把 `ResearchFlowError` 转成 state 更新而不是抛出：
     ```python
     def step(name):
         def wrap(fn):
             def node(state, runtime):
                 t0 = now()
                 try:
                     out = fn(state, runtime)
                     return {**out, "steps": {name: StepRecord(status="ok", ...)}}
                 except ResearchFlowError as e:
                     err = ErrorRecord(code=e.code, step=name, message=e.message, detail=e.detail)
                     return {"steps": {name: StepRecord(status="error", error=err, ...)},
                             "termination": Termination(kind="error", step=name, error=err)}
             return node
         return wrap
     ```
     然后每个节点后都走同一个路由（见 P0-2）。这个装饰器同时是 Gate 2 计时、Langfuse span、重试计数的唯一挂载点。

### P0-2 终止语义只在 ③ 生效，`terminal_reason` 一律映射为"审核拒绝"

- **证据**：`nodes/retrieval.py:21-22` 设置了 `terminal_reason`，但 `graphs/orchestrator.py:66` 是无条件边，链路会继续进入 ② 和 ③ 并触发人工审核；`route_after_review`（`orchestrator.py:49-51`）只挂在 data_pipeline 后。`main.py:135-137` 把任何 `terminal_reason` 都返回 `EXIT_REVIEW_REJECTED=2`。
- **问题**：目前因为 `search_local` 有"全不命中就返回全部语料"的兜底（`tools/literature.py:86`），这个分支永远走不到，属于潜伏缺陷；Gate 2 换成真实检索后就会触发。字符串型的 `terminal_reason` 也无法区分"拒绝 / 空检索 / 错误 / 校验不过"，回溯没法据此做决策。
- **建议**：
  - `terminal_reason: str` → `termination: Termination | None`，`Termination{kind: Literal["rejected","empty","error","budget_exhausted"], step, reason, error: ErrorRecord | None}`。
  - 统一路由函数 `route_next(next_step)`，每个节点后都用条件边：`termination` 不为空则 `END`，否则 `next_step`。
  - `main.py` 按 `termination.kind` 映射退出码（rejected→2，error→3，其余→1），不再靠字段是否存在来判断。

### P0-3 规则文件缺失时校验静默通过

- **证据**：`tools/rules_engine.py:13-15` 在文件不存在时返回 `{}`。⑦ 用空规则时只剩 `draft_min_chars(0)` 和 `forbidden_tokens([])` 两条，必然 `passed=True`（`rules_engine.py:52-87`）。③ 用空规则时 `required_columns=[]`，`precheck` 全部放行（`tools/quality.py:50-67`）。
- **问题**：路径配错或文件被误删时，退出码是 0，"硬规则校验通过"，这是验收链路里最不该出现的静默假阳性。
- **建议**：`load_rules(path, *, required=True)`，文件不存在时抛 `ConfigError`；再加一个最小 schema（Pydantic `SubmissionRules` / `AcceptanceRules`），解析失败也抛错。规则一旦变成 Pydantic 模型，Gate 3 的软评审规则可以作为同一模型的扩展字段加进来。

### P0-4 缺失率阈值实际不生效：在已经 dropna 的数据上计算

- **证据**：`tools/clean_strategies.py:28` 先删除关键列含缺失的行，`clean_strategies.py:32` 和 `tools/quality.py:60` 再在删除后的数据上算 missing_rate，结果恒为 0；`data_acceptance.yaml` 的 `max_missing_rate: 0.05` 永远不会触发。人工审核看到的"缺失率: 0.0"（`main.py:86-88`）对审核人有误导。
- **问题**：原始数据缺失 40% 也能通过，只是行数变少；如果没低于 `min_rows`，审核人根本不知道丢了多少数据。这是算法层面的缺陷，不是展示问题。
- **建议**：清洗函数返回 `raw_missing_rate`（删除前）和 `dropped_ratio = dropped_rows / rows_before`；`precheck` 判定 `raw_missing_rate <= max_missing_rate` 且 `dropped_ratio <= max_dropped_ratio`（新增规则项）；审核 payload 同时展示 raw 与 cleaned 两组数字。另外 `drop_duplicates` 发生在数值化之前（`clean_strategies.py:20` 与 `:23-25`），"1" 和 "1.0" 不会被判为重复，建议先数值化再去重。

### P0-5 HITL 输入默认放行

- **证据**：`main.py:97-98`：`approved = not answer.startswith("reject")`，空回车、`rejcet`、`no` 都会被当作 approve。
- **问题**：人工审核是 Gate 1 唯一的人类关口，默认值应该是"不通过"而不是"通过"。
- **建议**：严格白名单 `{approve, reject}`，其他输入重新询问；节点侧 `data_pipeline.py:51` 的宽松集合（`y`、`yes`）也收紧，并把 `decision` 改为 `Literal["approve","reject"]` 的 Pydantic 模型 `ReviewDecision`，resume 时做 `model_validate`。

### P0-6 LLM 结构化输出无重试，与设计方案不一致

- **证据**：设计方案 788 行规定"LLM 结构化输出失败：最多重试 2 次"；`core/llm.py:123-141` 失败一次就抛 `LLMError`。
- **问题**：真实冒烟（Gate 1 的阻断条件 D-07）最容易在这里失败。
- **建议**：`OpenAICompatClient.complete` 内部做至多 `max_retries` 次：HTTP 5xx/超时退避重试；`ValidationError` 时把错误摘要追加进下一轮 user message（"上次输出未通过校验：…，请修正"）。重试次数与最后一次错误写进 `LLMError.detail`，供 P0-1 的 ErrorRecord 记录。

---

## P1：Gate 2/3 扩展的结构性接缝（建议 Gate 1 收尾时先改接口）

### P1-1 跨进程 checkpoint：图、checkpointer、运行上下文三者需要解耦

- **证据**：
  - `graphs/orchestrator.py:76-79` 在 `build_graph()` 内写死 `InMemorySaver`。
  - `main.py:118` 把不可序列化的 `RunContext`（含 LLM 客户端）放进 `config["configurable"]`；`core/config.py:64` 的类型是 `llm: Any`。
  - `main.py:125-129` 的 resume 依赖同一进程里的 `result` 变量。
- **问题**：换成 SqliteSaver 后，新进程拿着 `thread_id` 恢复时，需要能**仅凭 run_id 重建**运行上下文（run_dir、llm 模式、配置快照），但现在这些信息只在第一次进程的内存里。
- **建议**：
  1. `build_graph(checkpointer: BaseCheckpointSaver | None = None)`，由 `make_checkpointer(settings)` 根据 `checkpoint.backend: memory | sqlite` 构造；序列化器配置放进这个工厂。
  2. 改用 langgraph 1.x 的 `context_schema`：`StateGraph(GraphState, context_schema=RunContext)`，节点签名 `def node(state, runtime: Runtime[RunContext])`，调用时 `graph.invoke(input, config, context=ctx)`。这样 ctx 不会混进 `configurable`，类型也明确。
  3. 首次运行时把"重建上下文所需的最小信息"落盘：`runs/<run_id>/run_meta.json = {run_id, topic_path, settings_snapshot, llm_mode, git_commit, created_at}`。`RunContext.from_run_id(run_id)` 从这里重建。
  4. CLI 增加 `researchflow resume --run-id X [--decision approve|reject --reason ...]`；主循环统一改为读 `graph.get_state(config).interrupts`，不依赖上一次 `invoke` 的返回值。这样同一套代码在 Gate 3 FastAPI 里就是 `POST /runs/{id}/resume`。
  5. 序列化白名单（`orchestrator.py:33-46`）目前是正确的：jsonplus 对 pydantic v2 只以顶层类重建，嵌套模型按 dict 存，所以漏登记 `QualitySummary`/`CheckResult` 不会出错。但这个列表是手工维护的，建议改为从 `schemas` 模块自动收集所有 `BaseModel` 子类，并加一条单测：构造一个完整 GraphState，经 serde `dumps_typed`→`loads_typed` 往返后相等。

### P1-2 ③ 把副作用和 interrupt 放在同一节点，跨进程后"审的"和"用的"可能不是同一份数据

- **证据**：`nodes/data_pipeline.py:32-35` 在 interrupt 之前读取 CSV、清洗、预检；resume 时整个节点从头重跑（LangGraph 语义），`data_pipeline.py:68-71` 在 interrupt 之后才落盘。审核 payload（`:37-48`）不含任何 checksum。
- **问题**：进程内这确实是幂等的（README §5.3 的说法成立）。但持久化之后，暂停到恢复之间可能隔几小时甚至几天，期间原始 CSV 被改了，resume 重跑会用新数据，审核记录却对应旧数据。这是 HITL 审计链的断点。
- **建议**：拆成两个节点：
  - `data_prepare`：清洗 + 预检 + 写 `cleaned_data.candidate.csv` + 计算 checksum，结果进 state（`cleaned_candidate: CleanedDataRef`，含 `raw_checksum`）。
  - `data_review`：只做 `interrupt(payload)`，payload 带 `raw_checksum`/`cleaned_checksum`；resume 后校验文件 checksum 与 state 一致，不一致就以 `Termination(kind="error", code=RF-003)` 结束。`ReviewRecord` 增加 `subject_checksum` 字段。
  - 这样 interrupt 节点没有任何副作用，重跑是严格幂等的；Gate 2 的"审核队列"也只需要处理 `data_review` 这一类挂起点。

### P1-3 追加型 reducer 与回溯不兼容

- **证据**：`core/state.py:44/48/53` 的 `literature_refs`、`review_log`、`artifacts` 都是 `append_list`。
- **问题**：Gate 2 回溯 = 从某个节点重跑。重跑 ① 会让文献重复；重跑 ④ 会让 `compute_result` 产物出现两条（旧的路径已被覆盖、checksum 已失效）。`validation.py:26` 只看 kind 是否存在，会把旧的失效记录也当作"有"。manifest（`main.py:69`）会出现重复和失效条目。
- **建议**：按语义分三类 reducer：
  - `literature_refs`：**替换**语义（每次检索就是一个完整结果集），去掉 reducer 或用 `replace`。
  - `artifacts`：**按 key 覆盖**，`Annotated[dict[str, ArtifactRef], merge_dict]`，key 为 `kind`（MODULE 有多产物时用 `f"{kind}:{name}"`）。`ArtifactRef` 增加 `attempt: int`。
  - `review_log`：保留**追加**（审计日志本来就该只增不减），但每条记录带 `attempt`。
  - 新增 `attempt: Annotated[dict[str, int], merge_dict]` 或放进 `StepRecord`，回溯路由用它做上限判断。

### P1-4 计算层没有 Mode/Backend 抽象，`algorithm` 参数被静默忽略

- **证据**：`nodes/compute.py:23` 直接调用 `run_direct`；`compute/direct.py:13/43` 写死 `LinearRegression`，完全不读 `ParamSet.algorithm`（`core/schemas.py:59`）。课题 YAML 写 `algorithm: ridge` 也会跑线性回归，产物里却写着 `linear_regression`。设计方案 724 行要求 `ComputeMode` 只定义 DIRECT/MODULE，Gate 1 仅允许 DIRECT，代码里没有这个枚举。
- **问题**：静默忽略用户参数本身是个缺陷；而且 MODULE 进来时只能在 `compute_node` 里写 if/else。
- **建议**（只定义接口，Gate 1 只注册一个实现）：
  ```python
  class ComputeMode(StrEnum): DIRECT = "direct"; MODULE = "module"

  class ComputeSpec(BaseModel):
      mode: ComputeMode = ComputeMode.DIRECT
      algorithm: str
      hyperparams: dict[str, Any] = {}
      seed: int
      module: ModuleSpec | None = None      # MODULE 时：入口、镜像、对齐映射

  class ComputeBackend(Protocol):
      def run(self, spec: ComputeSpec, data: CleanedDataRef, out_dir: Path) -> ComputeResult: ...

  BACKENDS: dict[ComputeMode, ComputeBackend]   # DIRECT → DirectBackend(registry={"linear_regression": ...})
  ```
  未注册的 `(mode, algorithm)` 抛 `ComputeError`。MODULE 的 Docker sandbox 作为 `ModuleBackend` 的执行器注入（与设计方案"sandbox 是 MODULE 的执行后端"一致），数据交换只通过 `out_dir` 里的文件 + 一个 `result.json` 契约，进程边界天然就有了。

### P1-5 ⑤ 图表依赖线性模型系数回算，MODULE/非线性模型下无法工作

- **证据**：`nodes/chart.py:16-27` 从 `compute_result.parameters` 取 `coefficients`/`intercept`，用线性公式在全量数据上重算预测值。
- **问题**：
  1. 换成任何非线性算法或外部 MODULE，这段代码都不能用。
  2. 指标是在测试集上算的（`direct.py:45-53`），图却画训练集加测试集，图和指标描述的不是同一批样本。
  3. ⑤ 重复做了 ④ 的计算，属于职责泄漏。
- **建议**：④ 输出一个预测产物 `predictions.csv`（`row_id, split, y_true, y_pred`），作为 `ArtifactRef(kind="predictions")` 登记，`ComputeResult` 增加 `predictions_ref`。⑤ 只读这个文件，按 `split` 着色，不再知道模型是什么。这也是 MODULE 契约里要求外部模块必须产出的文件之一。

### P1-6 数据集描述和超参数混在 `ParamSet.params` 里

- **证据**：`main.py:39-46` 把 `dataset_path/target/features` 塞进 `ParamSet.params`；`data_pipeline.py:22-24`、`extraction.py:20`、`compute.py:20-26` 都从 `params` 里取数据集字段。
- **问题**：Gate 2 的"正式参数流水线"产出的是超参数，不应该覆盖数据集定义；两类信息的审核人和来源也不同。
- **建议**：拆出 `DatasetSpec{path, target, features, checksum}`，作为 state 的独立字段；`ParamSet` → `ComputeSpec`（见 P1-4），只放算法和超参。

### P1-7 检索没有接口层，RAG 无处挂载；⑥ 不读文献

- **证据**：`nodes/retrieval.py:8/16` 直接 import 调用 `search_local`；`LiteratureRef`（`core/schemas.py:15-23`）没有稳定 ID；`nodes/writing.py:46-57` 完全不使用 `literature_refs`。
- **问题**：Gate 3 的 RAG 和"引用插入"都需要：可替换的检索器、可引用的文献 ID，以及写作节点能拿到检索结果。
- **建议**：
  - `core` 中定义 `Retriever` Protocol：`search(query: str, k: int) -> list[LiteratureRef]`，通过 `RunContext.retriever` 注入；Gate 1 是 `LocalCorpusRetriever`，Gate 3 是 `VectorRetriever`（chunk 级）或 `HybridRetriever`。
  - `LiteratureRef` 增加 `ref_id`（doi 或内容哈希）、`chunk_id: str | None`、`retriever: str`。
  - `DraftContent` 增加 `citations: list[str]`（ref_id 列表）；⑦ 增加规则"citations 必须都是 literature_refs 中的 ref_id"，这样就能防止 RAG 阶段的幻觉引用。
  - 检索查询目前只用 `topic`，Gate 3 建议在 ② 之后再做一次基于 `ResearchQuestion.variables` 的二次检索（query expansion），图上表现为 `extraction → retrieval_refine` 的可选边。

### P1-8 错误回溯：需要的是"可路由的错误"，不是更多异常类

- **证据**：`core/errors.py` 只有 6 个扁平的异常类；`step_status` 是 `dict[str, str]`（`core/state.py:55`）；⑦ 失败后直接 END（`orchestrator.py:74`）。
- **建议**（Gate 1 只放数据结构，Gate 2 再实现路由）：
  - `ErrorRecord{code, step, message, detail, retryable: bool, attempt}`；异常类增加类属性 `retryable` 和 `default_backtrack_to: str | None`（例如 RF-003 → `data_prepare`，RF-004 → `compute`）。
  - ⑦ 的每个 `CheckResult` 增加 `owner_step`（`metric:r2` → `compute`，`section:*` → `writing`，`artifact:chart` → `chart`）。在规则 YAML 中声明，而不是写在代码里。
  - Gate 2 新增 `attribute` 节点：输入 `termination` 或失败的 `CheckResult`，输出 `BacktrackPlan{target_step, patch: dict, reason}`，用 `Command(goto=target_step, update=patch)` 跳转；用 `attempt` 上限加上 `max_total_backtracks` 防止死循环，超限时以 `Termination(kind="budget_exhausted")` 结束。
  - 有了 P0-1、P1-3、P1-8 的数据形状，回溯就只是加节点和条件边，不用改现有节点。

---

## P2：值得做的改进（可放 Gate 2）

| # | 证据 | 问题 | 建议 |
| --- | --- | --- | --- |
| P2-1 | `nodes/*` 所有 `ArtifactRef.path=str(abs_path)`；`writing.py:74` 把绝对路径写进 draft | manifest 和初稿里都是本机绝对路径：不可跨机复现，也会泄露用户目录名（README §9.3 关心的点）；Gate 3 容器里路径不同 | state 和 manifest 中存相对 `run_dir` 的路径，由 `ctx.resolve(rel)` 解析；draft 里写 `chart.png` |
| P2-2 | `compute.py:22`、`chart.py:22` 读取 `cleaned_data.path` 时不校验 checksum | 跨进程/跨机器恢复后无法确认读到的是审核过的那份数据 | 提供 `read_verified(ref)` 工具函数，checksum 不匹配抛 `ArtifactError` |
| P2-3 | `validation.py:26` 只检查产物 kind 是否在列表里 | 文件被删或被改也会通过 | `artifact:*` 规则改为校验"文件存在且 sha256 与登记一致" |
| P2-4 | `direct.py:38-53`：61 行 × 0.25 ≈ 15 个测试样本上算 r2，阈值 0.8（`submission_checks.yaml`） | 单次划分的 r2 方差很大，换 seed 就可能跨过阈值，README 也承认了这个风险（R-20） | 指标改用 `KFold(n_splits=5, shuffle=True, random_state=seed)` 的均值 ± 标准差，阈值判定用均值，标准差写进报告；保留单次划分用于出图 |
| P2-5 | 无结构化日志 | 失败时只有 stderr 一行 | P0-1 的装饰器同时写 `runs/<id>/events.jsonl`（step、status、耗时、输入/输出 digest）；Gate 2 接 Langfuse 时把同一个钩子换成 span |
| P2-6 | `ReviewRecord.timestamp` 用本地时间且无时区（`schemas.py:71`），`main.py:65` 同理 | 跨时区/服务化后排序和审计会出问题 | 统一 `datetime.now(UTC).isoformat()` |
| P2-7 | `writing.py:12-37` 模板内联在代码中，`extraction` 的 prompt 却在 configs 里 | 不一致；Gate 3 ⑧⑨ 修订/校稿也要模板 | 模板统一放 `configs/templates/`，用 Jinja2 `FileSystemLoader`；② 的 `str.replace` 渲染（`extraction.py:15`）也换成 Jinja2 |
| P2-8 | `settings.yaml` 的 `paths.dataset` 是死配置（README §10 已自认） | 误导 | 删除，或作为课题 YAML 缺省值真正读取 |

---

## 建议的落地顺序

1. **Gate 1 打标签前**：P0-1 ~ P0-6。改动都局限在单个文件或单个函数内，其中 P0-3、P0-4、P0-5 应各配一条反例单测（缺失规则文件、高缺失率原始数据、非法审核输入）。
2. **Gate 1 → Gate 2 过渡（只改形状，不加功能）**：P1-3（reducer）、P1-6（DatasetSpec）、P1-1 的第 1/2/3 步（checkpointer 注入、`context_schema`、run_meta）、P1-2（③ 拆节点）。这些越晚改，受影响的节点和测试越多。
3. **Gate 2 功能期**：P1-4/P1-5（Backend + predictions 契约，进而接入 MODULE/sandbox）、P1-8（attribute 节点 + BacktrackPlan）、P1-1 的第 4 步（resume CLI）、P2 项。
4. **Gate 3**：P1-7（Retriever + 引用校验）在 Gate 2 已定好接口的前提下，只需新增实现类；FastAPI 层复用 P1-1 的 `start / get_state / resume` 三个操作即可。

## 未验证事项

- 未运行测试，以上所有结论均来自读代码。P0-4（缺失率恒为 0）和 P0-3（空规则时通过）是根据代码逻辑推断的，建议各写一条单测确认。
- P1-1 中关于 jsonplus 嵌套模型的结论，依据是 `.venv/.../langgraph/checkpoint/serde/jsonplus.py:305-316` 的 `_msgpack_default` 实现（顶层 `model_dump()`），未实际做往返测试。

---

# 附录 A：测试达标标准

测试分三轮，每一轮都有明确的"达标"判据。用例 ID 与正文意见一一对应（T-P0-3 对应 P0-3）。

- **第 1 轮（现在，代码未修改）**：跑基线测试，再加"确认型"用例，目的是证明问题真实存在。确认型用例在这一轮**预期失败**，用 `@pytest.mark.xfail(strict=True, reason="P0-x 待修")` 标记。修复之后它会变成 XPASS，`strict=True` 会让该用例报错，从而提醒你把 xfail 标记去掉。
- **第 2 轮（P0 修完）**：去掉 xfail，所有 T-P0 用例都必须通过。
- **第 3 轮（P1 接口改完）**：T-P1 形状用例必须通过，同时第 1、2 轮的用例不得退化。

## A.1 基线（每一轮都必须满足）

| ID | 命令（仓库根目录执行） | 达标标准 |
| --- | --- | --- |
| B-1 | `uv lock --check && uv sync --locked` | 退出码 0 |
| B-2 | `uv run ruff check . && uv run ruff format --check .` | 退出码 0，并贴出 `# noqa` 的数量（`grep -rn noqa src \| wc -l`） |
| B-3 | `uv run mypy` | 0 error；贴出检查的文件数 |
| B-4 | `uv run pytest -rxXs`（`addopts` 已含 `-q`，不要再叠加） | 0 failed、0 error。贴出最后的汇总行（含 passed/xfailed/xpassed 计数）。第 1 轮的 xfailed 数应等于本轮新增确认型用例的数量 |
| B-5 | `RF_LLM_MODE=fake uv run researchflow run --topic examples/demo_topic.yaml --auto-approve`，连续跑两次 | 两次退出码都为 0；`cleaned_data.csv`、`compute_result.json`、`chart.png`、`validation_report.json` 四个文件的 sha256 两次完全一致（`draft.md` 含 run_id，预期不同）。贴出两次的 run_id 与四个 checksum 对照 |
| B-6 | 同上，但交互输入 `reject` 并填写原因 | 退出码 2；`runs/<id>/` 下没有 `cleaned_data.csv`；manifest 的 `review_log[0].decision == "reject"` |

## A.2 第 1 轮确认型用例（预期失败，用来证明问题存在）

| ID | 建议位置 | 构造 | 第 1 轮预期 | 第 2 轮达标标准 |
| --- | --- | --- | --- | --- |
| T-P0-1 | `tests/e2e/test_gate1_failure_paths.py` | 把 `data_acceptance.yaml` 的 `min_rows` 改成 1000（用 tmp 副本 + `settings.raw` 注入），调用 `main.main([...])` | 退出码 3，但 `runs/<id>/manifest.json` **不存在** → xfail | 退出码 3；manifest 存在，且 `status=="error"`、`error.code=="RF-003"`、`error.step=="data_pipeline"`（或拆分后的 `data_prepare`）、`step_status` 中含该步骤 |
| T-P0-2 | `tests/unit/test_routing.py` | 对编译后的图做结构断言：除 `validation` 外，每个节点的出边都是条件边；再 monkeypatch `search_local` 返回 `[]` 并跑图 | 空检索仍然进入 ② 并触发 interrupt → xfail | 空检索直接 END，不产生 interrupt；`termination.kind=="empty"`；CLI 退出码 **不是** 2 |
| T-P0-3 | `tests/unit/test_rules_engine.py` | `load_rules("不存在.yaml")`；另外用 `{}` 规则跑 `evaluate_submission` | 返回 `{}`，空规则报告 `passed=True` → xfail | 文件不存在抛 `ConfigError`（RF-001）；规则缺少必填段时 schema 校验失败 |
| T-P0-4 | `tests/unit/test_quality.py` | 构造 100 行数据，其中 40 行的 `x1` 为 NaN，依次走 `clean_frame` → `precheck`（**注意必须走组合路径**，现有的 `test_precheck_rejects_missing_rate_above_threshold` 只单测了 `precheck`，所以没能发现这个问题） | 不抛异常，`missing_rate==0.0` → xfail | 抛 `DataValidationError`，消息里带 raw 缺失率 0.4；审核 payload 同时含 `raw_missing_rate` 与 `dropped_rows` |
| T-P0-4b | 同上 | 两行内容分别为 `"1"` 与 `"1.0"`，其他列相同 | `duplicates_removed==0` → xfail | `duplicates_removed==1` |
| T-P0-5 | `tests/unit/test_cli_review.py` | monkeypatch `input` 依次返回 `""`、`"rejcet"`、`"no"` | 三种输入都得到 approve → xfail | 非法输入会被重新询问（第二次返回 `reject` 时结果为 reject）；节点侧 `ReviewDecision.model_validate({"decision": "yes"})` 抛 `ValidationError` |
| T-P0-6 | `tests/unit/test_llm_retry.py`（用已声明但未使用的 `respx`） | 打桩：第一次返回不合 schema 的 JSON，第二次返回合法 JSON；另设一组连续 3 次都不合法 | 第一次就抛 `LLMError` → xfail | 第一组成功，且请求计数为 2，第二次请求的 messages 中含上一次的校验错误摘要；第二组请求计数为 3，`LLMError.detail.attempts==3` |

## A.2b 第 2 轮：确认型用例的收紧要求（第 1 轮评审后追加）

第 1 轮写出的 7 条用例已足以证明问题存在。但如果第 2 轮只是去掉 xfail 标记，其中几条要么会**卡死**，要么**断言不足**，或者与建议的数据形状**冲突**。所以第 2 轮修 P0 时，要按下表同时改测试：

| ID | 现状问题 | 第 2 轮要求 |
| --- | --- | --- |
| T-P0-5 | `monkeypatch input` 用 `lambda: ""` 恒定返回空串。按 P0-5 修成"非法输入重新询问"之后会**无限循环**，测试挂死；`"no"` 未覆盖；`ReviewDecision` 只断言存在 | 用迭代器 `["", "no", "rejcet", "reject", "原因"]`，断言最终为 reject，且 `input` 被调用 5 次；`_ask_decision` 增加 `max_attempts`（建议 5），超限抛 `ConfigError` 并加一条用例覆盖；补 `ReviewDecision.model_validate({"decision": "yes"})` 抛 `ValidationError` |
| T-P0-2 | 用 `termination["kind"]` 下标访问，而 P0-2 建议 `Termination` 是 Pydantic 模型，取值方式不一致；缺少 CLI 退出码断言 | 改成 `termination.kind == "empty"`；补一条 CLI 级断言：空检索时 `main.main(...)` 的退出码 `!= 2`（建议为 1） |
| T-P0-1 | 缺 `error.step` 断言 | 补 `manifest["error"]["step"] == "data_pipeline"`（拆分后改为 `data_prepare`），并断言 `run_id` 与目录名一致 |
| T-P0-6 | 未断言"重试时会把校验错误回传给模型" | 补：`route.calls[1].request` 的 body 中，最后一条 user message 含上一次的校验错误摘要 |
| T-P0-4 | 只测了 `precheck` 抛错，没测审核人看到的数字 | 在 e2e 中补：审核 payload 含 `raw_missing_rate` 与 `dropped_rows`，且 `raw_missing_rate` 不再恒为 0 |
| T-P0-3 | `evaluate_submission({})` 必须返回 `passed=False`，这把实现锁死在"dict 规则"上 | 改为二选一都算达标：`evaluate_submission` 拒绝空规则（抛 `ConfigError`），或返回 `passed=False`；另补"规则缺必填段时 `load_rules` 抛错" |

另外，第 2 轮修 P0-1/P0-2 时，节点仍从 `configurable["ctx"]` 取上下文，这一点允许保留。P1-1 改为 `context_schema` 是第 3 轮的事，届时 T-P0-1/T-P0-2 的构造代码需要一起更新。

## A.3 第 3 轮：Gate 2/3 接口形状用例

这一轮不测功能，只锁定数据形状和接口，防止后续实现时回退。

| ID | 对应意见 | 达标标准 |
| --- | --- | --- |
| T-P1-1a | P1-1 序列化 | 构造一个**字段填满**的 GraphState，经 checkpointer 的 `serde.dumps_typed` → `loads_typed` 往返后，与原值 `==`；白名单从 schemas 模块自动收集，新增模型时无需改 orchestrator |
| T-P1-1b | P1-1 注入 | `build_graph(checkpointer=X)` 使用传入的 X；节点签名使用 `Runtime[RunContext]`，`grep -rn 'configurable' src/researchflow/nodes` 结果为空 |
| T-P1-1c | P1-1 跨进程（Gate 1 用 `InMemorySaver` 模拟） | 在 interrupt 处只保留 checkpointer 与 run_id，丢弃 graph 和 ctx 对象，用 `RunContext.from_run_id(run_id)` 加新 `build_graph(同一 saver)` 进行 resume，链路能跑完，产物 checksum 与 B-5 一致 |
| T-P1-2 | P1-2 ③ 拆分 | 在 interrupt 期间改写原始 CSV 的一个字节，然后 resume，应以 RF-003 终止，且 `review_log` 记录的 `subject_checksum` 与改写前一致；`data_review` 节点内无任何文件写操作（可用 monkeypatch `Path.write_text`/`to_csv` 计数为 0） |
| T-P1-3 | P1-3 reducer | 用 `Command(goto="compute")` 或 `update_state` 让 ④ 重跑一次，`artifacts` 中 `compute_result` 只有 1 条，且 `attempt==2`；`literature_refs` 在 ① 重跑后长度不变；`review_log` 长度 +1 |
| T-P1-4 | P1-4 Backend | `ComputeSpec(algorithm="ridge")` 抛 `ComputeError`（不再静默跑线性回归）；`ComputeMode.MODULE` 未注册时报出可读错误 |
| T-P1-5 | P1-5 predictions | ④ 产出 `predictions.csv`，含 `split` 列；⑤ 模块内不再出现 `coefficients`（`grep` 为空）；图上测试集点数 == `metrics.n_test` |
| T-P1-6 | P1-6 DatasetSpec | `ParamSet`/`ComputeSpec` 中没有 `dataset_path`/`target`/`features` 字段 |
| T-P1-8 | P1-8 错误形状 | 每个 `ResearchFlowError` 子类都声明 `retryable`；`submission_checks.yaml` 中每条规则都有 `owner_step`，且它的值是图中真实存在的节点名（用 `graph.get_graph().nodes` 校验） |

P1-7（Retriever）不设第 3 轮用例。它属于 Gate 3 的范围，只需在 Gate 2 结束前确认 `Retriever` Protocol 存在，且 ① 通过 ctx 调用它。

---

# 附录 B：测试结果返回方式

**存放位置**：仓库内 `docs/reviews/gate1/`（代码实现 agent 的工作目录就是仓库根，放在这里它才读得到）

```
researchflow/docs/reviews/gate1/
├── gate1-framework-review.md   ← 本评审文件（标准来源）
├── TEMPLATE.md                 ← 结果模板
├── round1-YYYYMMDD.md          ← 每轮一份，复制模板填写
├── round2-YYYYMMDD.md
└── artifacts/roundN/           ← 可选：完整 pytest 输出、两次运行的 manifest.json（建议加入 .gitignore）
```

**返回流程**：

1. 让代码实现 agent 按 `TEMPLATE.md` 填好 `roundN-日期.md`。命令输出只贴汇总行和失败用例片段；完整输出放在 `artifacts/roundN/` 下。
2. 在本评审线程里发一句"第 N 轮结果好了"即可，不需要粘贴内容。我在你的机器上可以直接读这个目录。
3. 我对照附录 A 逐条判定 达标 / 未达标 / 与预期不符，并把结论写回同一份结果文件末尾的「评审结论」一节，再在线程里给出摘要。

**判定规则**：

- 第 1 轮中，如果某条确认型用例**意外通过**（XPASS），说明我的推断有误，我会撤回对应意见并在评审文件里标注。
- 任何一轮中，基线 B-1~B-6 只要有一项失败，该轮就判为未达标，其他用例的结论仅供参考。

# ResearchFlow 简历导向改造方案（对照 2026 Agent 开发岗位要求）

- 日期：2026-10-02
- 依据：仓库 `src/`（约 1900 行，commit `6431c02` 之后）、`docs/reviews/gate1/` 全部评审与第 2 轮结果、上级目录《全流程科研Agent项目设计与实现方案》§0/§3/§8、《Agent开发实习岗位技术栈详解》，以及联网检索的 2026 年招聘资料（来源见文末）
- 性质：只做框架设计和建议，不改代码；不提交

---

## 0. 结论先行

1. **现在的项目在面试官眼里是"带 LLM 调用的数据流水线"，不是 Agent。** 图是固定的 7 节点 DAG，LLM 只在 ②提炼、⑥写作被调用两次，且都是"把 JSON Schema 写进 prompt + `response_format=json_object`"（`core/llm.py:121-128,168`）。没有工具调用、没有自主决策回路、检索是对 7 条硬编码语料做关键词匹配（`tools/literature.py`）。简历写"科研 Agent"会在第一个追问（"你的 Agent 自己决定了什么？"）上失分。
2. **已有部分里最值钱的是工程底座，要保留并讲出来**：LangGraph `interrupt` + checkpointer 的 HITL、Pydantic 契约 + 结构化输出校验失败回喂重试、`termination` 统一路由、产物 checksum 可复现、错误进 manifest、36 条测试 + CI。这些正好是 JD 里"生产级 / 可靠性"那一半，大多数学生项目没有。
3. **改造方向不是推翻重写，而是"确定性工作流外壳 + 有边界的 Agent 内核"。** 这正是 Anthropic《Building Effective Agents》推荐的形态：确定的步骤用 workflow，开放的决策才交给 agent。具体落在三处天然需要自主决策的地方：
   - **归因回溯 Agent**（Gate 2 本来就要做的 `attribute` 节点）→ 改成带工具调用的有界 ReAct 回路；
   - **文献检索 Agent**（① 换成真实学术 API + 多轮检索/改写）→ Agentic RAG；
   - **评审-修订回路**（Gate 3 的软评审 + ⑧）→ Evaluator-Optimizer。
4. **评测（Eval）是这次改造里投入产出比最高的一项，必须前置到 Gate 2 开头。** 2026 年 JD 中评测/Tracing 出现频率很高（检索到的一份 21 份 JD 抽样中 14 份提到），而且简历上所有"可量化"数字都只能从评测集里来。没有 eval，下面所有简历写法都只能写功能，不能写数字。
5. **对正在进行的 Gate 1 第 2 轮没有阻断影响**：P0 已全部达标，可以照常判定、打 `v0.1.0-mvp` 标签。影响在第 3 轮：建议新增 4 个接口形状项（工具注册表、LLM 用量回传、事件日志、Retriever 提前），并把原计划的 Streamlit 剔除。详见 §4。

---

## 1. 岗位要求 × 项目现状对照表

"覆盖度"列：✅ 已有且能讲 / 🟡 有雏形但简历上撑不住 / ❌ 没有。

| # | 岗位能力（2026 JD 高频） | 项目现状（证据） | 覆盖度 | 简历上的问题 |
| --- | --- | --- | --- | --- |
| 1 | Tool / Function Calling（Schema 设计、并行调用、错误回传、重试） | 无。`LLMClient.complete` 只有结构化输出，没有 `tools` 参数 | ❌ | 最核心的一项空白 |
| 2 | Agent 范式：ReAct / Plan-and-Execute / Reflection | 固定 DAG，无模型自主分支 | ❌ | 会被定性为 pipeline |
| 3 | 结构化输出 + 校验 | Pydantic 契约 + 校验失败把错误摘要回喂重试（P0-6，已测） | ✅ | 能讲，但缺"首轮合法率"等数字 |
| 4 | LangGraph（State/Reducer/条件边/interrupt/checkpointer） | 全部用到；termination 路由、reducer 语义问题已在评审中分析 | ✅ | 跨进程恢复还没做（P1-1） |
| 5 | Human-in-the-loop | CLI + `interrupt`，白名单决策、审计记录 | ✅ | 审核对象未带 checksum（P1-2） |
| 6 | RAG（切分、Embedding、混合检索、Rerank、引用溯源） | 本地 7 条语料关键词匹配；⑥不读文献 | ❌ | JD 几乎必提，抽样中 21/21 提到 RAG/检索 |
| 7 | 向量数据库 | 无 | ❌ | 同上 |
| 8 | MCP | 无 | ❌ | 2026 大厂 JD 已写入；"自己写过 MCP Server"是公认加分项 |
| 9 | 评测 Eval（黄金集、自动评测、回归检测、LLM-as-Judge） | 只有功能测试，无质量指标 | ❌ | 没有任何可量化结果 |
| 10 | 可观测性（trace/span、token、成本、延迟） | `run_id` + manifest；无 trace、无 token 统计 | 🟡 | 说不出 p95 延迟、单次成本 |
| 11 | 记忆（短期上下文管理 / 长期经验） | 无；设计中有"失败经验库" | ❌ | 可以小成本做出亮点 |
| 12 | 多 Agent | 无 | ❌ | 不建议硬凑，见 §3.3 |
| 13 | Guardrails / 安全（prompt 注入、工具权限、沙箱） | 硬规则校验 ⑦；Docker 沙箱在设计中 | 🟡 | 没有针对 LLM 的防护 |
| 14 | 服务化 + 流式（FastAPI、SSE、异步、部署） | CLI only；Gate 3 规划 FastAPI | ❌ | 规划中有，但 SSE 未提 |
| 15 | 上下文工程（Context Engineering） | `max_refs_in_prompt`/`max_ref_chars` 截断 | 🟡 | 太薄，但 Agentic RAG 做完自然变厚 |
| 16 | 工程素养（类型、测试、CI、可复现） | ruff + mypy + 36 测试 + 锁依赖 + checksum 复现 | ✅ | 优势项，要量化写 |

---

## 2. 改造后的目标架构

```text
接入层     CLI  │  FastAPI（REST + SSE 流式事件）  │  MCP Server（对外暴露工具）
             └──────────── start / get_state / resume 三个操作（P1-1）────────────┘
编排层     LangGraph 主图（确定性工作流外壳，保留现有 ①→⑦ 与 termination 路由）
             ├─ ① 文献检索子图 ── Agentic RAG：检索 Agent（工具调用回路，有预算）
             ├─ ③ data_prepare → data_review(interrupt)        （P1-2）
             ├─ ⑦ 校验失败 / 节点错误 ──► 归因回溯 Agent（ReAct + 工具）
             │                              └─ Command(goto=目标节点, update=patch)
             └─ ⑥ 写作 ⇄ 软评审（Evaluator-Optimizer，有界迭代）
能力层     Tool Registry：每个工具 = Pydantic 入参 + 描述 + 实现 + 权限标签
             同一份定义同时导出为 ① LLM tools schema ② MCP tool ③ 普通 Python 调用
基础层     SQLite checkpointer │ 向量库（Chroma/Qdrant）│ 失败经验库（episodic memory）
           Tracing（Langfuse / OTel span，挂在 @step 装饰器上）│ Eval Harness（CI + 手动真实模型）
```

关键设计判断（面试要能讲清楚为什么）：

- **外壳为什么保持确定性**：科研流程的步骤顺序是领域知识，不需要模型来决定；把它交给模型只会降低可复现性和可审计性。只有"出错后改哪里""要不要再检索""稿子够不够好"这类开放决策才交给 Agent。
- **每个 Agent 回路都有预算**：`max_steps`、`max_tool_calls`、`max_tokens`、`max_total_backtracks`，超限以 `Termination(kind="budget_exhausted")` 结束并升级人工。这条线已经在 P1-8 的数据形状里预留了。
- **工具定义只写一份**：避免"LLM 工具一套、MCP 一套、测试一套"的三份漂移。这是 MCP 面试题里很好的答点。

---

## 3. 保留 / 新增 / 剔除

### 3.1 保留（只改形状，不加功能）

| 模块 | 保留理由 | 需要配合的现有评审项 |
| --- | --- | --- |
| LangGraph 主图 + `route_next` 终止路由 | 是 Agent 内核的外壳；条件边已统一 | P1-8（attribute 节点接入点） |
| HITL（interrupt + 白名单 + ReviewRecord） | 企业级 Agent 必备范式，学生项目少见 | P1-1 跨进程、P1-2 拆节点 |
| Pydantic 契约 + 结构化输出重试 | 直接迁移为工具入参校验和工具错误回传 | 新增 R-1（见 §3.2） |
| 产物 checksum + manifest + 可复现 | Eval 回归检测的地基 | P2-1 相对路径 |
| DIRECT 计算 + ComputeBackend 接口 | 作为 Agent 可调用的工具 | P1-4、P1-5 |
| Fake/Real 双实现 | Eval 的离线回归层依赖它 | Fake 需要支持"脚本化工具调用序列"（见 R-1） |
| 测试与 CI | 优势项 | 新增 eval 冒烟作为 CI 一步 |

### 3.2 新增（按投入产出比排序）

每项给出：做什么 / 落在哪个 Gate / 最小可交付 / 简历可量化指标 / 面试可讲的点。指标里的 X、Y 必须来自实际评测，**不要编数字**。

#### R-1 原生 Tool Calling + 工具注册表（Gate 1 第 3 轮定接口，Gate 2 实现）

- **做什么**：
  - `core/tools.py`：`ToolSpec{name, description, args_model: type[BaseModel], fn, side_effect: Literal["read","write","exec"], timeout_s}`，`registry.as_openai_tools()` / `registry.as_mcp_tools()` / `registry.invoke(name, raw_args)`。
  - `LLMClient` 增加 `chat(messages, tools=None, tool_choice=None) -> AssistantTurn{content, tool_calls, usage}`，保留现有 `complete()` 作为结构化输出快捷方式。
  - 结构化输出改为优先用 `response_format={"type":"json_schema","strict":true}`（Provider 支持时），保留 `json_object` + prompt 内 schema 作为降级路径。
  - 工具执行统一做：Pydantic 校验入参 → 超时 → 异常转成 `ToolError{code, message, retryable}` 以 tool message 回填给模型（而不是抛出），让模型自我纠错。
  - FakeLLM 增加"脚本模式"：按预设序列返回 tool_calls，用于确定性测试 Agent 回路。
- **简历指标**：结构化输出首轮合法率（`json_object` vs `json_schema strict` vs tool calling 三种方式对比）、工具调用参数合法率、平均每任务工具调用次数。
- **面试点**：Function Calling 闭环（模型只产生意图，宿主执行）；为什么工具错误要回填而不是抛出；工具描述怎样影响选择准确率；`side_effect` 标签如何支撑权限控制和 HITL（写/执行类工具需要审批）。

#### R-2 Eval Harness：故障注入基准 + 指标看板（Gate 2 第一件事）

- **做什么**：
  - `evals/cases/*.yaml`：30~50 条用例，每条 = 课题 + 数据集 + **注入的故障** + 期望结果。故障类型覆盖：高缺失率、列名错误、类型污染、离群点、重复行、规则阈值不达标（r2 低）、检索为空、LLM 输出不合 schema、引用了不存在的文献。故障可以由脚本从 `sample_dataset.csv` 自动变异生成，每条记录种子。
  - `evals/run.py`：批量运行，输出 `evals/reports/<date>.json` + Markdown 表格。指标：
    - 端到端成功率（含 HITL 自动审批模式）
    - 故障归因准确率（Agent 给出的 `target_step` 与期望一致的比例）
    - 自动恢复率（回溯后最终通过 ⑦）、平均回溯次数
    - 结构化输出首轮合法率、工具调用参数合法率
    - 引用忠实度（citations ⊆ 已检索 ref_id；Gate 3 加 LLM-as-Judge 判断"引文是否支持该句"）
    - 单次运行 token、成本、p50/p95 延迟（来自 R-3）
  - 两层运行：Fake 层进 CI 做回归（数字固定，任何变动即失败）；真实模型层手动/夜间跑，结果归档。
- **简历指标**：用例数、各指标数值、"引入 X 后指标从 A 提升到 B"的对比（这是最有说服力的一类句子）。
- **面试点**：黄金集怎么构造、为什么用故障注入而不是人工标注（可自动扩充、标签天然正确）、Fake 回归和真实评测的分层、指标如何防止"刷分"（例如归因准确率要和恢复率一起看）。

#### R-3 Tracing 与成本可观测（Gate 1 第 3 轮定钩子，Gate 2 接 Langfuse）

- **做什么**：评审 P0-1 建议的 `@step` 装饰器作为唯一挂载点：每个节点、每次 LLM 调用、每次工具调用各一个 span，属性含 token（R-1 的 `usage`）、模型名、耗时、输入/输出 digest、attempt。Gate 1 先写 `runs/<id>/events.jsonl`（即原 P2-5，提升优先级），Gate 2 把同一钩子换成 Langfuse（或 OTel 导出），eval 分数作为 score 挂到 trace 上。
- **简历指标**：单次运行平均 token / 成本、各节点 p95 延迟、失败 run 定位时间（可以说"从日志定位到 trace 定位"）。
- **面试点**：trace 与 span 的区别；为什么把 eval 分数挂到 trace（从回归指标反查是哪次改动导致）；采样与脱敏。

#### R-4 归因回溯 Agent（Gate 2 核心，替代原 P1-8 的规则式 attribute 节点）

- **做什么**：⑦ 失败或节点 `termination.kind=="error"` 时进入 `attribute` 子图，一个有界 ReAct 回路：
  - 只读工具：`read_validation_report`、`read_step_error`、`inspect_dataset(summary/column/sample)`、`read_artifact(kind)`、`search_failure_memory(error_signature)`（R-7）。
  - 决策工具（终止回路）：`propose_backtrack(target_step, patch, reason)`、`escalate_to_human(reason)`。
  - `target_step` 必须是图中真实节点（P1-8 已要求），`patch` 用 Pydantic 校验（例如只允许改 `ComputeSpec.hyperparams`、清洗策略参数、检索 query）。
  - 规则先行：规则 YAML 中 `owner_step` 能直接确定归因的，跳过 LLM（便宜、确定）；只有多条规则同时失败或错误原因不明确时才调用 Agent。这个"规则优先、模型兜底"的分层要在面试里主动讲。
  - 每次回溯 `attempt+1`，超过 `max_total_backtracks` 以 `budget_exhausted` 结束并转人工。
- **简历指标**：归因准确率、自动恢复率、平均回溯轮数、"规则直接命中 / 需要 LLM"的占比、相对"纯规则归因"基线的提升。
- **面试点**：为什么 Agent 只给只读工具 + 一个决策工具（最小权限、可审计）；如何防止死循环；与 LangGraph `Command(goto=...)` 的结合；为什么不是让模型直接改数据。

#### R-5 真实文献检索 + Agentic RAG（检索部分 Gate 2，RAG 部分 Gate 3）

- **做什么**：
  - Gate 2：`Retriever` Protocol（原 P1-7，从 Gate 3 提前）+ 真实学术 API 适配器作为工具（建议 OpenAlex 或 arXiv，免 key、限流友好；Semantic Scholar 作为第二源），带本地缓存，现有本地语料降级为 fixture。① 变为检索 Agent：工具 `search(query, year_range, k)`、`get_abstract(ref_id)`；模型可以改写 query、多轮检索，自己判断"够了"；预算约束同上。
  - Gate 3：对摘要/开放获取全文做切分 → Embedding → 向量库（本地 Chroma 即可，面试时讲清 Qdrant/Milvus/pgvector 的取舍）；BM25 + 向量混合检索，RRF 融合，可选 reranker；⑥ 写作时按 chunk 引用，`DraftContent.citations` 记录 `ref_id/chunk_id`；⑦ 校验 citations ⊆ 已检索集合（P1-7 已设计）。
  - 检索评测：从真实领域选 20~30 个查询，人工标注相关文献，算 Recall@k / MRR；对比"单次关键词检索 / 查询改写 / 混合检索 / +rerank"四档。
- **简历指标**：Recall@10、MRR 的分档提升；引用忠实度；幻觉引用率（降到 0 依赖 ⑦ 的硬校验）。
- **面试点**：RAG 全链路；切分策略的取舍；混合检索为什么在学术文本上有效（专有名词、缩写）；Agentic RAG 与固定 RAG 的区别和成本；如何把"引用不能编造"从 prompt 约束升级为程序校验。

#### R-6 MCP Server（Gate 2 末，成本低、加分高）

- **做什么**：用官方 Python SDK 的 FastMCP，把 R-1 注册表里只读和低风险工具导出：`search_literature`、`check_data_quality(csv_path)`、`run_direct_compute(spec)`、`get_run_status(run_id)`、`start_run(topic_yaml)`；写/执行类工具需要显式开关。stdio 传输先做，HTTP 传输放 Gate 3 与 FastAPI 一起。验证方式：在 Claude Desktop / Cursor / MCP Inspector 里实际调用一次并截图存档。
- **可选反向**：作为 MCP Client 接入一个现成的学术检索 MCP Server，说明"工具既可以自带也可以外接"。
- **简历指标**：导出工具数；在 N 个客户端中验证可用。
- **面试点**：Host / Client / Server 三角色；MCP 解决的是什么问题（M×N 集成）；tools / resources / prompts 的区别；为什么要做权限分级；与 function calling 的关系（MCP 是工具的分发协议，最终仍以 tool schema 交给模型）。

#### R-7 失败经验库（episodic memory，Gate 2 末或 Gate 3）

- **做什么**：每次回溯结束写一条 `FailureCase{error_signature, context_digest, attribution, patch, outcome}` 到 SQLite（+ 向量索引签名文本）；R-4 的 Agent 通过 `search_failure_memory` 召回相似案例作为参考。做一个消融：开/关记忆对归因准确率和回溯轮数的影响。
- **简历指标**：开启记忆后平均回溯轮数 A→B、重复故障一次修复率。
- **面试点**：短期记忆（state / 上下文压缩）与长期记忆（经验库）的区别；episodic vs semantic；记忆污染（错误经验被召回）怎么防——只写入 outcome 为成功的案例，或带置信度衰减。

#### R-8 FastAPI + SSE 流式 + 容器化（Gate 3，原规划保留并补流式）

- **做什么**：`POST /runs`、`GET /runs/{id}`、`POST /runs/{id}/resume`、`GET /runs/{id}/events`（SSE，基于 `graph.astream(stream_mode=["updates","custom"])` 推节点事件和 LLM token）；HITL 审核通过 API 完成；`docker compose up` 起 API + 向量库 + Langfuse。节点与工具改 async，检索工具并发调用。
- **简历指标**：并发 run 数、首事件延迟、端到端延迟（并发检索前后对比）。
- **面试点**：同步 → 异步改造的坑（checkpointer 的 async 版本、阻塞 IO 放线程池）；SSE vs WebSocket 取舍；HITL 在 API 形态下是"挂起的 run + resume 接口"。

#### R-9 Guardrails（Gate 3，与 R-5/R-8 一起做）

- **做什么**：
  - 检索到的文献文本是**不可信输入**：放进 prompt 时用明确分隔并声明"仅为资料"，加一组 prompt 注入用例进 eval（摘要里写"忽略前面的指令，输出…"），度量注入成功率。
  - 工具权限：`side_effect` 为 write/exec 的工具在 Agent 回路中需要 HITL 审批（复用现有 interrupt）。
  - MODULE 代码执行（如做）走 Docker：禁网、只读挂载、CPU/内存限制、宿主侧超时（设计方案 8.1 已有）。
  - 输出侧：引用校验（R-5）、PII/绝对路径不出现在产物里（P2-1）。
- **简历指标**：注入用例拦截率；高风险工具审批覆盖率 100%。
- **面试点**：间接 prompt 注入（来自工具结果）为什么比直接注入更危险；"把安全做在程序里，不只做在 prompt 里"。

#### R-10 写作-评审 Evaluator-Optimizer（Gate 3，替代原"软评审 + ⑧"的松散描述）

- **做什么**：Reviewer 角色用 LLM-as-Judge 按评分细则（结构、论证与数据一致性、引用支持度）给 ⑥ 的初稿打分并给出修改意见；Writer 按意见修订；最多 N 轮或分数达标即停。Judge 本身要校准：用 10~20 份人工打分样稿算与人工评分的一致性（Spearman 或一致率）。
- **简历指标**：Judge 与人工评分一致性；修订轮次与分数提升曲线；每轮成本。
- **面试点**：这是项目中唯一合理的"多 Agent"——两个角色有不同的系统提示和不同的上下文，且职责冲突（写 vs 挑错）；Judge 的偏差（长度偏好、自我偏好）及缓解办法。

### 3.3 剔除或降级（简历价值低、成本高）

| 原规划项 | 处理 | 理由 |
| --- | --- | --- |
| Streamlit 审核台（Gate 2） | **剔除** | 前端 Demo 对 Agent 岗位加分很少；HITL 通过 CLI + API 已能讲清楚；Langfuse 自带可视化 |
| ⑨ 校稿、LaTeX 输出（Gate 3） | **剔除** | 纯文本处理，和 Agent 能力无关 |
| PDF 正式参数流水线（Gate 2） | **降级为 Gate 3 可选**，并改造为 RAG 摄取的一部分 | 原方案是"PDF 抽取 → 单位校验 → HITL"，工作量大，技术点与 R-1 结构化输出、R-5 文档切分重复 |
| MODULE + LLM 生成对齐代码 + AST guard + Docker（Gate 2） | **降级为 Gate 3 可选**，如做则做成 Agent 的一个 `exec` 工具 | 成本最高的一项；"代码执行沙箱"有面试价值，但优先级低于 Eval / 工具调用 / RAG；接口（P1-4 ComputeBackend）照常在第 3 轮定好，保证以后能接 |
| 为凑"多 Agent"而加 Supervisor / 研究员-程序员-评审员 角色分工 | **不做** | 固定流程上加 Supervisor 只会增加成本和不确定性；面试时"知道什么时候不用多 Agent"本身就是答点 |
| P2-4 KFold、P2-7 模板迁 Jinja2、Git revert 演练等 | **保留但不投入额外时间** | 工程正确性项，做了不加分、不做会被问；放在顺手时做 |
| 第二算法、可插拔清洗策略库 | **不做** | 数据科学方向的扩展，不是 Agent 岗位关注点 |

---

## 4. 与 Gate 1/2/3 规划对齐，以及对第 2 轮的影响

### 4.1 对已完成的 Gate 1 第 2 轮（P0 修复）

- **没有阻断影响，不需要返工**。第 2 轮 36 条用例全部通过，P0-1~P0-6 的修复方向与本方案一致：
  - P0-1 的错误进 manifest / `ErrorRecord` → 正是 R-3 trace 和 R-4 归因的输入；
  - P0-2 的 `Termination` → R-4 的入口条件和 `budget_exhausted` 出口；
  - P0-6 的"校验错误回喂重试" → R-1 工具错误回填的同一模式，面试时可以连着讲。
- 第 2 轮第 4 节"偏离"中的第 1 条（`step_status` 仍是 `dict[str,str]`）在本方案下**变得更重要**：R-3 的 span 和 R-2 的指标都需要 `StepRecord{status, started_at, duration_ms, attempt, usage}`。建议在第 3 轮完成，而不是留到 P1-8 一起。
- 第 2 轮的"评审结论"一节尚未填写；按原流程判定即可，本方案不改变判定标准。
- **Gate 1 收尾仍缺一项硬条件**：设计方案的 D-07"至少一次真实 LLM 冒烟"。这一条对简历很关键（所有真实模型指标都从这里起步），建议在打 `v0.1.0-mvp` 标签前完成。

### 4.2 第 3 轮（原"P1 接口形状"）新增的 4 项

原有 T-P1-1 ~ T-P1-8 全部保留。新增以下形状用例，只锁接口，不要求功能：

| ID | 对应 | 达标标准 |
| --- | --- | --- |
| T-R1 | R-1 工具注册表 | `core/tools.py` 存在 `ToolSpec` 与 registry；至少把 `search_local`、`precheck`、`run_direct` 三个函数登记为工具；`registry.as_openai_tools()` 产出的每个 schema 都能被 `jsonschema` 校验为合法 JSON Schema；非法入参调用 `registry.invoke` 返回 `ToolError` 而不是抛出 |
| T-R1b | R-1 LLM 用量 | `LLMClient` 的返回（或 `last_usage` 属性）含 `prompt_tokens/completion_tokens`；Fake 返回确定的伪造值；manifest 中每个调用 LLM 的步骤有 usage 汇总 |
| T-R3 | R-3 事件日志（原 P2-5 提前） | 每次运行生成 `runs/<id>/events.jsonl`；每个已执行节点恰好一条 `end` 事件，含 `step/status/duration_ms/attempt`；失败 run 的最后一条事件含 `error.code`；`StepRecord` 进入 `GraphState` |
| T-R5 | R-5 Retriever（原 P1-7 提前） | `Retriever` Protocol 存在；① 通过 ctx/runtime 调用它，`grep -n search_local nodes/` 为空；`LiteratureRef` 含 `ref_id` |

### 4.3 调整后的 Gate 路线

| Gate | 原范围 | 调整后范围 | 完成标志（可直接转成简历数字） |
| --- | --- | --- | --- |
| **Gate 1**（收尾） | ①→⑦ MVP | 不变 + 真实 LLM 冒烟 + 第 3 轮接口（含 T-R1/R1b/R3/R5） | 打 `v0.1.0-mvp`；测试数、复现 checksum 一致 |
| **Gate 2**（审核回溯闭环 → **Agentic 闭环 + 评测**） | SQLite、参数流水线、MODULE、Docker、归因回溯、Langfuse、Streamlit | ① SQLite checkpointer + resume CLI（P1-1）② Eval Harness v1（R-2）③ 原生 tool calling（R-1）④ 归因回溯 Agent（R-4）⑤ Tracing/成本（R-3）⑥ 真实文献检索 Agent（R-5 前半）⑦ MCP Server stdio（R-6）⑧ 失败经验库（R-7，时间允许）| 故障注入集上：归因准确率、自动恢复率、单次成本/延迟；MCP 在一个客户端验证可用 |
| **Gate 3**（完整交付） | ⑧⑨、RAG、软评审、FastAPI、Compose | ① RAG：向量库 + 混合检索 + 引用校验（R-5 后半）② Evaluator-Optimizer 写作评审（R-10）③ FastAPI + SSE + Compose（R-8）④ Guardrails 与注入评测（R-9）⑤ 可选：MODULE 沙箱工具、PDF 参数抽取 | Recall@k/MRR 分档提升；Judge 一致性；注入拦截率；`docker compose up` 复现 |

Gate 2 内部顺序建议：**② Eval 先于 ④ Agent**。先用"纯规则归因"跑出基线数字，再上 Agent，才有"从 A 提升到 B"的对比；反过来做就只能写绝对值。

---

## 5. 简历写法建议

### 5.1 项目标题与一句话定位

> **ResearchFlow：可审计的科研流程 Agent（LangGraph / Tool Calling / Agentic RAG / MCP / Eval）**
> 以确定性工作流编排科研全流程（检索 → 问题提炼 → 数据清洗与人工审核 → 计算 → 图表 → 写作 → 校验），在归因回溯、文献检索、写作评审三处引入有预算约束的工具调用 Agent，配套故障注入评测集与全链路 Tracing。

### 5.2 要点模板（X/Y/A/B 用实测值替换；没测到的项不要写）

1. **编排与 HITL**：基于 LangGraph 设计 7 节点状态机，统一 `termination` 条件路由与错误状态化；用 `interrupt` + SQLite checkpointer 实现跨进程可恢复的人工审核，审核记录绑定数据 checksum，保证"审的就是用的"。
2. **工具调用 Agent**：设计统一工具注册表（Pydantic 入参 + 权限标签），同一份定义导出为 LLM tool schema 与 MCP tool；实现有界 ReAct 归因回溯 Agent（只读工具 + 决策工具，规则优先、模型兜底），在 **N** 条故障注入用例上归因准确率 **X%**（纯规则基线 **A%**），自动恢复率 **Y%**，平均回溯 **Z** 轮。
3. **结构化输出可靠性**：Pydantic 契约 + 校验错误回喂重试；对比 json_object / json_schema strict / tool calling 三种方式，首轮合法率 **A% / B% / C%**，最终合法率 **100%**（≤2 次重试）。
4. **Agentic RAG**：接入 OpenAlex/arXiv 检索工具，Agent 自主改写查询与多轮检索；BM25 + 向量混合检索 + RRF，Recall@10 从 **A** 提升至 **B**；引用由程序校验必须来自检索结果，幻觉引用率 **0**。
5. **评测与可观测**：构建 **N** 条故障注入评测集，Fake 层接入 CI 做回归、真实模型层定期评测；基于 Langfuse 记录节点 / LLM / 工具三级 span，单次运行平均 **T** tokens、成本 **¥C**、p95 延迟 **L s**。
6. **MCP 与服务化**：将 **K** 个工具封装为 MCP Server 并在 Claude Desktop/Cursor 中验证；FastAPI + SSE 推送节点事件，Docker Compose 一键复现。
7. **工程质量**（现在就能写）：**36+** 条单元/端到端测试、ruff + mypy 严格检查接入 CI；依赖锁定与产物 sha256 校验，同一输入两次运行四类产物逐字节一致。

写作原则：一条要点 = 动作 + 技术 + 可验证结果；数字写清口径（多少条用例、哪个模型）；面试官会追问的数字必须能在 `evals/reports/` 里找到对应文件。

### 5.3 面试可讲的"深挖点"清单（按项目里的真实决策组织）

| 追问 | 可以讲的真实决策 | 证据位置 |
| --- | --- | --- |
| 这到底是 Agent 还是 workflow？ | 外壳确定性（可复现、可审计），三处开放决策交给 Agent；引用 Anthropic 的 workflow/agent 划分 | §2 |
| LangGraph reducer 有什么坑？ | 追加型 reducer 在回溯重跑时会重复，按语义拆成替换 / 按 key 覆盖 / 追加审计三类 | gate1 评审 P1-3 |
| interrupt 恢复时节点会重跑，怎么办？ | 把副作用与 interrupt 拆成两个节点；审核对象带 checksum | P1-2 |
| 模型输出不合 schema 怎么办？ | 错误摘要回喂、有界重试、首轮合法率数据 | P0-6、R-1 |
| Agent 死循环怎么防？ | attempt 计数、全局回溯预算、`budget_exhausted` 转人工 | R-4 |
| 你怎么知道改动变好了？ | 故障注入评测集 + 基线对比 + CI 回归 | R-2 |
| 为什么用 / 不用多 Agent？ | 只在写作-评审这种职责冲突处用；其他地方单 Agent + 好工具更稳 | R-10、§3.3 |
| MCP 和 function calling 什么关系？ | 注册表一份定义两种导出；MCP 是分发协议 | R-1、R-6 |
| prompt 注入怎么防？ | 检索内容视为不可信、程序化引用校验、高风险工具审批 | R-9 |
| 发现过什么隐蔽 bug？ | dropna 后算缺失率恒为 0、规则文件缺失时静默通过、审核输入默认放行——都是"静默假阳性"，用"先写失败用例再修"的方式锁住 | gate1 评审 P0-3/4/5 |

最后一行很重要：评审里发现并修复的 P0 问题本身就是很好的面试素材，体现"会验证、会找静默错误"，比功能堆叠更能区分候选人。

---

## 6. 需要你决定的事

1. **Streamlit 是否剔除**：建议剔除（理由见 §3.3）。如果你有展示需求（比如答辩），可以改为 FastAPI 自带的 `/docs` + Langfuse 界面。
2. **MODULE + Docker 沙箱是否降级到 Gate 3 可选**：建议降级。它是原设计里 Gate 2 的重头，降级后 Gate 2 时间主要给 Eval 和 Agent。
3. **真实模型选择**：Eval 的真实层需要一个固定模型和预算上限；建议选一个支持 `json_schema strict` 和原生 tool calling 的 OpenAI 兼容 Provider，否则 R-1 的三方式对比只能做两档。

---

## 未验证事项

- 招聘数据来自检索结果摘要（部分页面无法直接打开全文），频次数字（如 21 份抽样中 14 份提到评测）引自二手汇总，只作方向参考。
- 各 R 项的工作量没有给出时间估计；第 3 轮接口项都是局部改动，其余项建议每项开始前再出一份类似 gate1 评审的接口与测试标准。
- 文中 Provider 对 `json_schema strict` / tool calling 的支持情况取决于你选的模型，未实测。

## 参考来源

- [AI Agent 开发学习与面试指南：招聘需求与能力模型报告（todayforai）](https://todayforai.com/zh/guides/20260722-guide-agent-engineer-00-recruitment-requirements-report)
- [AI Agent 研发实习生（2026届）· 传音控股（牛客）](https://www.nowcoder.com/jobs/detail/444729)
- [如何成为一个 AI Agent 工程师？2026 版学习路线（牛客）](https://www.nowcoder.com/discuss/879367411529506816)
- [Agent 岗位 7 个方向（牛客）](https://www.nowcoder.com/discuss/879082734507810816)
- [ai-agent-engineer-handbook：JD requirements（GitHub）](https://github.com/harrisliangsu/ai-agent-engineer-handbook/blob/main/interview-prep/jd-requirements.md)
- [2026 年第 30 周 Agent＋Java 岗位观察（掘金）](https://juejin.cn/post/7667140623372255242)
- [MCP and Tool-Use Architecture Careers (topgenaijobs)](https://www.topgenaijobs.com/blog/mcp-tool-use-architecture-careers)
- [AI Developer Hiring 2026: Skills That Actually Matter (digitalapplied)](https://www.digitalapplied.com/blog/ai-developer-hiring-skills-that-matter-2026)
- [The Complete AI Agent Engineer Skills Stack 2026 (Great Learning)](https://www.mygreatlearning.com/blog/the-complete-ai-agent-engineer-skills-stack-you-need-in-2026/)
- Anthropic, *Building Effective Agents*（workflow 与 agent 的划分、Evaluator-Optimizer 模式）

# STATUS（researchflow-agent）

最后更新：2026-10-17（Claude）

## 现在
- main = origin/main，Gate 2 完成，Gate 3 进行中。已合并：T-025 Tracing、T-026 FastAPI+SSE、T-027 价格按模型计币种/评测单列 Agent 失败、T-028 检索 Agent 摩擦修复、T-029 有据写作与引用硬校验、T-030 切块+混合重排、T-031 adaptive 融合与共用 .env 隔离、T-032 引用支持度双评委、T-033 检索 Agent 注入评测与防护、T-034 在线多查询混合重排（默认关）、T-035 容器化（`docker compose up` 零密钥可跑，CI 有容器冒烟）。测试 545 条。
- 关键数字（单次运行、模型评委，口径见各报告）：
  - 归因回溯 Agent：test 集可修复 13/13、应转人工 11/11、无效回溯 0（`docs/reviews/gate2/attribution-eval-20261010.md`）。
  - 检索 Agent：困难课题 Precision@5 0.31→0.93（18 胜 2 负），简单课题 0.83→0.87 不显著（`docs/reviews/gate2/retrieval-eval-20261013.md` 末节）。
  - 引用：Fake 注入/编造拦截率 100%、误杀 0%；真实 20 课题校验前幻觉引用率 0/118；引用支持度（双评委）严格 90.9%、宽松 100%、κ 0.88（`docs/reviews/gate3/citation-eval-20261016.md`）。
  - 离线重排（困难课题）：纯向量/adaptive P@5 0.84，等权 RRF 0.45，BM25 0.34；**adaptive + Agent 改写的多查询 P@5 0.96、Recall@10 0.93**（`docs/reviews/gate3/rerank-eval-20261017.md`）。

  - 注入：deepseek 对 12 种攻击严格成功率 0/60（有无防护都是），防护使被注入文献被选中 53%→25%（`docs/reviews/gate3/injection-eval-20261017.md`）。

## 下次入口
1. 读 `CLAUDE.md` → 本文件 → `.collab/board.md` → 最新的 `.collab/reports/`、`reviews/`。
2. 真实调用的配置：聊天用 `.env.commandcode`（`env $(grep -v '^#' .env.commandcode | xargs) ...`）；嵌入用百炼 `text-embedding-v4`（`RF_EMBED_BASE_URL/API_KEY/MODEL`，`RF_EMBED_BATCH_SIZE=10`）；百炼聊天接口已 403。检索建池加 `RF_RETRIEVER=openalex NO_PROXY=api.openalex.org`。
3. CodeBuddy 派发：`nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' caffeinate -i .collab/dispatch.sh T-xxx [continue]`；被取消（499）时先 `git add -A && git commit -m wip` 再 continue。

## 待办
- 在线重排已接入（`retrieval.rerank.enabled`），需要在 20+20 课题上做开/关的在线对比评测，再决定是否默认开启。
- 注入评测加强：攻击文本放进与课题无关的文献；写作评审 Evaluator-Optimizer。

## 待 gu 决定
- 无（gu 2026-10-16 授权合并 main、push；只在改方向/简历口径、较大花费、改写历史、删数据时问）。

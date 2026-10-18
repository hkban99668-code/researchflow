# researchflow-agent

可审计的科研流程 Agent（LangGraph / Tool Calling / Agentic RAG / MCP / Eval）。Claude 只审查和写任务卡，CodeBuddy 实现（`.collab/`）。

接手顺序：`STATUS.md` → `.claude/memory/MEMORY.md`（尤其 resume-pivot-direction）、`docs/reviews/issues-log.md`、`.collab/PROTOCOL.md`、`.collab/decisions.md`。

硬规则：
- 真实模型调用必须带 `--confirm-real-calls`，CodeBuddy 不运行任何可能发真实请求的命令；Key 只在被忽略的 `.env*`。
- 合并 main、push、删除前问 gu；评测数字要写样本量和口径。
- 跑测试用 `set -o pipefail`；自测在只装主依赖的干净目录里再跑一遍。
- 检索评测建池必须设 `RF_RETRIEVER=openalex`；Agent 得分异常先查有没有静默降级。
- 给 gu 的文件路径一律用完整路径（/Users/gu/CodeBuddy/20260921235524/...）。

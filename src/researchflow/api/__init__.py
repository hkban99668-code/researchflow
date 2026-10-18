"""FastAPI 服务层（T-026）：用 HTTP + SSE 暴露运行与 API 形态的人工审核。

设计要点（Claude 在任务卡里定，本层照做）：

- 图与节点保持同步，API 只用线程池跑 `researchflow.main.drive_run`，不做 async 改造。
- SSE 的数据源是 `runs/<id>/events.jsonl`（T-004 就为推送预留的同一份流），不是内存
  队列：文件是持久的，进程重启、客户端断线重连都能续上。
- HITL 是"挂起的 run + resume 接口"：停在审核处线程就退出，`POST /runs/{id}/resume`
  起新线程接着跑；状态在 checkpoint 里，所以 API 进程重启后照样能 resume。
- run 状态不存内存，从 `events.jsonl` 与 `manifest.json` 推出来。
"""

from .app import create_app

__all__ = ["create_app"]

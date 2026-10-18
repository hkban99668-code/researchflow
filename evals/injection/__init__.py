"""T-033：检索 Agent 的间接提示注入评测与防护（R-9）。

包结构：
- `attacks.yaml`：三类攻击 × 中英两种措辞 × 注入位置 1/3，共 12 种攻击文本；
- `injecting_retriever.py`：包装任意 `Retriever`，确定性地在指定位置的结果摘要前
  插入攻击文本，并记录被注入的 ref_id（A1）；
- `rules.py`：三类攻击的成功判定规则（可程序判定，不靠人工读）与 suspicious
  标记的召回/误报统计（A2）；
- `fake_clients.py`：模拟"被骗"与"没被骗"两种模型输出的 Fake LLM 客户端
  （Fake 层专用，不调真实模型）；
- `run.py`：批量运行入口，`--layer fake` 验证判定与报告，`--layer real` 需
  `--confirm-real-calls`（不带只预演）。

本包不发送任何真实网络请求：检索固定用 `evals/retrieval/pool.json` 构造的本地
候选池检索器；`--layer real` 只让检索 Agent 用真实模型。
"""

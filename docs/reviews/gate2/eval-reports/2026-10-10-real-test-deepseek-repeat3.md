# Eval 报告（真实层，只让归因 Agent 用真实模型）

- 生成时间：2026-10-10T15:10:49+00:00
- git_commit：6ef6daa9315a0f4a8f7d79ab0a84ffb55809fd07
- cases_sha256：a9c7fc05faf925297fa3006b5bcf849ec4df4ae073e2290cb299b4e135b623ce
- split：test / model：deepseek-v4-flash-0731 / extra_body：{}
- repeat：3（每次运行是单次结果，repeat_correct 是 N 次中判对次数）
- 预算：max_total_tokens=300000

## 汇总

| 组 | 用例数 | Agent 归因准确率 | 规则基线归因准确率 | recovered |
|------|------|------|------|------|
| 总体 | 28 | 100.00% | 72.73% | 25.00% |
| clean | 4 | — | — | 100.00% |
| 单步故障 | 18 | 100.00% | 100.00% | 0.00% |
| 跨步骤故障 | 3 | 100.00% | 0.00% | 100.00% |
| 应转人工 | 3 | 100.00% | 0.00% | 0.00% |

- 转人工精确率：100.00%，召回率：100.00%（正例是 human 组，只看 Agent 的决策）
- 无效工具参数率：0.00%
- token：平均 5020.5，合计 140574

## 逐条用例

| id | 组 | split | 实际终止 | decision(action/target_step) | 工具调用数 | agent_tokens | agent_latency_ms | repeat_correct | baseline_ok | attribution_ok | recovered |
|------|------|------|------|------|------|------|------|------|------|------|------|
| F-CLEAN-01 | clean | test | ok | / | 0 | 0 | 0 | 0/3 | False | False | True |
| F-CLEAN-02 | clean | test | ok | / | 0 | 0 | 0 | 0/3 | False | False | True |
| F-DUP-01 | clean | test | ok | / | 0 | 0 | 0 | 0/3 | False | False | True |
| F-DUP-02 | clean | test | ok | / | 0 | 0 | 0 | 0/3 | False | False | True |
| F-MISS-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-MISS-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-MISS-03 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-DROP-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-DROP-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-RENAME-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-RENAME-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-RANGE-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-RANGE-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-RANGE-03 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-ROWS-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-ROWS-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-TYPE-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-TYPE-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 3/3 | True | True | False |
| F-ALGO-01 | single_step | test | escalated | escalate/compute | 0 | 0 | 0 | 3/3 | True | True | False |
| F-ALGO-02 | single_step | test | escalated | escalate/compute | 0 | 0 | 0 | 3/3 | True | True | False |
| F-EMPTY-01 | single_step | test | empty | / | 0 | 0 | 0 | 0/3 | n/a | n/a | False |
| F-EMPTY-02 | single_step | test | empty | / | 0 | 0 | 0 | 0/3 | n/a | n/a | False |
| F-OUT-02 | cross_step | test | ok | backtrack/data_prepare | 6 | 23435 | 57503 | 3/3 | False | True | True |
| F-OUT-04 | cross_step | test | ok | backtrack/data_prepare | 6 | 23214 | 53397 | 3/3 | False | True | True |
| F-OUT-06 | cross_step | test | ok | backtrack/data_prepare | 6 | 23203 | 51905 | 3/3 | False | True | True |
| F-NOISE-02 | human | test | escalated | escalate/data_prepare | 6 | 24023 | 65438 | 3/3 | False | True | False |
| F-SHUFFLE-01 | human | test | escalated | escalate/data_prepare | 6 | 23302 | 59111 | 3/3 | False | True | False |
| F-SHUFFLE-03 | human | test | escalated | escalate/data_prepare | 6 | 23397 | 60096 | 3/3 | False | True | False |

## 说明

- 单次运行的结果标注：真实模型每次输出可能不同。`decision` / `tool_calls` / `actual_termination` / `recovered` 是第 1 次运行的单次快照；`repeat_correct`/`n_repeats` 给出 N 次中判对的次数。`agent_tokens` / `agent_latency_ms` 是 N 次累计。
- `attribution_ok` 口径与 Fake 层一致（T-014）：期望 `target_step == "human"` 的用例，决策是 `escalate` 即算对；否则看 `target_step` 是否等于期望。
- 规则基线是同一批用例上用 `fake_attributor` 跑出来的确定性结果；`baseline_ok` 为 `n/a` 表示该用例不进 `attribute`（检索为空）。

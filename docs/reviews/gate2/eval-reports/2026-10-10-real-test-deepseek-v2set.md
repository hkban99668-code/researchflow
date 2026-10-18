# Eval 报告（真实层，只让归因 Agent 用真实模型）

- 生成时间：2026-10-10T17:40:20+00:00
- git_commit：631a1928b534b490eaf5a2f7f37b88abadb05ca4
- cases_sha256：783f776636587408ff46b66db7afc9aa98e6ededdecea24c9d847e25b4dcba19
- split：test / model：deepseek-v4-flash-0731 / extra_body：{}
- repeat：1（每次运行是单次结果，repeat_correct 是 N 次中判对次数）
- 预算：max_total_tokens=300000

## 汇总

| 组 | 用例数 | Agent 归因准确率 | 朴素规则归因 | 聪明规则归因 | recovered |
|------|------|------|------|------|------|
| 总体 | 40 | 100.00% | 47.06% | 100.00% | 32.50% |
| clean | 4 | — | — | — | 100.00% |
| 单步故障 | 18 | 100.00% | 100.00% | 100.00% | 0.00% |
| 跨步骤故障 | 9 | 100.00% | 0.00% | 100.00% | 100.00% |
| 应转人工 | 9 | 100.00% | 0.00% | 100.00% | 0.00% |

- 转人工精确率：100.00%，召回率：100.00%（正例是 human 组，只看 Agent 的决策）
- 无效工具参数率：0.00%
- 无效回溯率：朴素规则 100.00%，聪明规则 50.00%
- token：平均 3791.2，合计 151649

## 逐条用例

| id | 组 | split | 实际终止 | decision(action/target_step) | 工具调用数 | agent_tokens | agent_latency_ms | repeat_correct | baseline_ok | smart_baseline_ok | attribution_ok | recovered |
|------|------|------|------|------|------|------|------|------|------|------|------|------|
| F-CLEAN-01 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | True |
| F-CLEAN-02 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | True |
| F-DUP-01 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | True |
| F-DUP-02 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | True |
| F-MISS-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-MISS-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-MISS-03 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-DROP-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-DROP-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-RENAME-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-RENAME-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-RANGE-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-RANGE-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-RANGE-03 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-ROWS-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-ROWS-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-TYPE-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-TYPE-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-ALGO-01 | single_step | test | escalated | escalate/compute | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-ALGO-02 | single_step | test | escalated | escalate/compute | 0 | 0 | 0 | 1/1 | True | True | True | False |
| F-EMPTY-01 | single_step | test | empty | / | 0 | 0 | 0 | 0/1 | n/a | n/a | n/a | False |
| F-EMPTY-02 | single_step | test | empty | / | 0 | 0 | 0 | 0/1 | n/a | n/a | n/a | False |
| F-OUT-02 | cross_step | test | ok | backtrack/data_prepare | 6 | 8106 | 20814 | 1/1 | False | True | True | True |
| F-OUT-04 | cross_step | test | ok | backtrack/data_prepare | 6 | 7527 | 16247 | 1/1 | False | True | True | True |
| F-OUT-06 | cross_step | test | ok | backtrack/data_prepare | 5 | 7716 | 18713 | 1/1 | False | True | True | True |
| F-NOISE-02 | human | test | escalated | escalate/data_prepare | 6 | 8029 | 23078 | 1/1 | False | True | True | False |
| F-SHUFFLE-01 | human | test | escalated | escalate/data_prepare | 6 | 10675 | 18007 | 1/1 | False | True | True | False |
| F-SHUFFLE-03 | human | test | escalated | escalate/data_prepare | 5 | 7403 | 18280 | 1/1 | False | True | True | False |
| F-OUT-NOISE-02 | cross_step | test | ok | backtrack/data_prepare | 6 | 7752 | 18928 | 1/1 | False | True | True | True |
| F-OUT-NOISE-04 | cross_step | test | ok | backtrack/data_prepare | 6 | 12045 | 30695 | 1/1 | False | True | True | True |
| F-OUT-NOISE-06 | cross_step | test | ok | backtrack/data_prepare | 5 | 7911 | 20337 | 1/1 | False | True | True | True |
| F-OUT-MISS-02 | cross_step | test | ok | backtrack/data_prepare | 6 | 8539 | 26888 | 1/1 | False | True | True | True |
| F-OUT-MISS-04 | cross_step | test | ok | backtrack/data_prepare | 6 | 7566 | 16700 | 1/1 | False | True | True | True |
| F-OUT-MISS-06 | cross_step | test | ok | backtrack/data_prepare | 6 | 7692 | 18293 | 1/1 | False | True | True | True |
| F-YNOISE-02 | human | test | escalated | escalate/data_prepare | 5 | 7669 | 20727 | 1/1 | False | True | True | False |
| F-YNOISE-04 | human | test | escalated | escalate/data generation | 6 | 8455 | 28339 | 1/1 | False | True | True | False |
| F-YNONLIN-02 | human | test | escalated | escalate/data_prepare | 5 | 7909 | 22921 | 1/1 | False | True | True | False |
| F-YNONLIN-04 | human | test | escalated | escalate/data_prepare | 6 | 7821 | 20541 | 1/1 | False | True | True | False |
| F-YPART-02 | human | test | escalated | escalate/ | 6 | 10983 | 22373 | 1/1 | False | True | True | False |
| F-YPART-04 | human | test | escalated | escalate/null | 5 | 7851 | 21106 | 1/1 | False | True | True | False |

## 说明

- 单次运行的结果标注：真实模型每次输出可能不同。`decision` / `tool_calls` / `actual_termination` / `recovered` 是第 1 次运行的单次快照；`repeat_correct`/`n_repeats` 给出 N 次中判对的次数。`agent_tokens` / `agent_latency_ms` 是 N 次累计。
- `attribution_ok` 口径与 Fake 层一致（T-014）：期望 `target_step == "human"` 的用例，决策是 `escalate` 即算对；否则看 `target_step` 是否等于期望。
- 规则基线是同一批用例上用 `fake_attributor`（朴素）与 `smart_rule_attributor`（聪明） 跑出来的确定性结果；`baseline_ok` / `smart_baseline_ok` 为 `n/a` 表示该用例不进 `attribute`（检索为空）。

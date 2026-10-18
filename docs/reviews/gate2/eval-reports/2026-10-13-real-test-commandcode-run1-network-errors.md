# Eval 报告（真实层，只让归因 Agent 用真实模型）

- 生成时间：2026-10-13T15:05:56+00:00
- git_commit：0cdc7374d93da0e8b2ceb5fb744463f114d24b13
- cases_sha256：514446287ac341a557d577018974a3dd690ab8b25c89a0f660af114c0e2ad068
- split：test / model：deepseek/deepseek-v4-flash / extra_body：{}
- repeat：1（每次运行是单次结果，repeat_correct 是 N 次中判对次数）
- 预算：max_total_tokens=600000

## 汇总

| 组 | 用例数 | Agent 归因准确率 | 朴素规则归因 | 聪明规则归因 | 搜索规则归因 | recovered |
|------|------|------|------|------|------|------|
| 总体 | 46 | 87.50% | 45.00% | 72.50% | 92.50% | 26.09% |
| clean | 4 | — | — | — | — | 100.00% |
| 单步故障 | 18 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| 跨步骤故障 | 13 | 61.54% | 15.38% | 100.00% | 76.92% | 61.54% |
| 应转人工 | 11 | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% |

- 转人工精确率：68.75%，召回率：100.00%（正例是 human 组，只看 Agent 的决策）
- 无效工具参数率：0.00%
- 无效回溯率：朴素规则 100.00%，聪明规则 64.86%，搜索规则 0.00%
- 修法正确率（Agent）：100.00%
- token：平均 2012.3，合计 92568

## 逐条用例

| id | 组 | split | 实际终止 | decision(action/target_step) | 工具调用数 | agent_tokens | agent_latency_ms | repeat_correct | baseline_ok | smart_baseline_ok | search_baseline_ok | attribution_ok | recovered |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| F-CLEAN-01 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | False | True |
| F-CLEAN-02 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | False | True |
| F-DUP-01 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | False | True |
| F-DUP-02 | clean | test | ok | / | 0 | 0 | 0 | 0/1 | False | False | False | False | True |
| F-MISS-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-MISS-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-MISS-03 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-DROP-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-DROP-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-RENAME-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-RENAME-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-RANGE-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-RANGE-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-RANGE-03 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-ROWS-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-ROWS-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-TYPE-01 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-TYPE-02 | single_step | test | escalated | escalate/data_prepare | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-ALGO-01 | single_step | test | escalated | escalate/compute | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-ALGO-02 | single_step | test | escalated | escalate/compute | 0 | 0 | 0 | 1/1 | True | True | True | True | False |
| F-EMPTY-01 | single_step | test | empty | / | 0 | 0 | 0 | 0/1 | n/a | n/a | n/a | n/a | False |
| F-EMPTY-02 | single_step | test | empty | / | 0 | 0 | 0 | 0/1 | n/a | n/a | n/a | n/a | False |
| F-OUT-02 | cross_step | test | ok | backtrack/data_prepare | 6 | 8356 | 13810 | 1/1 | False | True | True | True | True |
| F-OUT-04 | cross_step | test | ok | backtrack/data_prepare | 6 | 8011 | 21387 | 1/1 | False | True | True | True | True |
| F-OUT-06 | cross_step | test | ok | backtrack/data_prepare | 6 | 8353 | 13382 | 1/1 | False | True | True | True | True |
| F-NOISE-02 | human | test | escalated | escalate/data_prepare | 5 | 8305 | 13717 | 1/1 | False | False | True | True | False |
| F-SHUFFLE-01 | human | test | escalated | escalate/data_prepare | 5 | 7912 | 12415 | 1/1 | False | False | True | True | False |
| F-SHUFFLE-03 | human | test | escalated | escalate/data_prepare | 6 | 8280 | 47226 | 1/1 | False | False | True | True | False |
| F-OUT-NOISE-02 | cross_step | test | ok | backtrack/data_prepare | 6 | 11313 | 90043 | 1/1 | False | True | True | True | True |
| F-OUT-NOISE-04 | cross_step | test | ok | backtrack/data_prepare | 5 | 8076 | 46983 | 1/1 | False | True | False | True | True |
| F-OUT-NOISE-06 | cross_step | test | ok | backtrack/data_prepare | 5 | 7835 | 11572 | 1/1 | False | True | True | True | True |
| F-OUT-MISS-02 | cross_step | test | ok | backtrack/data_prepare | 5 | 8091 | 13449 | 1/1 | False | True | False | True | True |
| F-OUT-MISS-04 | cross_step | test | ok | backtrack/data_prepare | 6 | 8036 | 26385 | 1/1 | False | True | True | True | True |
| F-OUT-MISS-06 | cross_step | test | escalated | escalate/ | 0 | 0 | 0 | 0/1 | False | True | True | False | False |
| F-YNOISE-02 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |
| F-YNOISE-04 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |
| F-YNONLIN-02 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |
| F-YNONLIN-04 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |
| F-YPART-02 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |
| F-YPART-04 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |
| F-XEXP-02 | cross_step | test | escalated | escalate/ | 0 | 0 | 0 | 0/1 | True | True | True | False | False |
| F-XEXP-04 | cross_step | test | escalated | escalate/ | 0 | 0 | 0 | 0/1 | True | True | True | False | False |
| F-XKTRAP-02 | cross_step | test | escalated | escalate/ | 0 | 0 | 0 | 0/1 | False | True | True | False | False |
| F-XKTRAP-04 | cross_step | test | escalated | escalate/ | 0 | 0 | 0 | 0/1 | False | True | False | False | False |
| F-ZEXP-02 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |
| F-ZEXP-04 | human | test | escalated | escalate/ | 0 | 0 | 0 | 1/1 | False | False | True | True | False |

## 说明

- 单次运行的结果标注：真实模型每次输出可能不同。`decision` / `tool_calls` / `actual_termination` / `recovered` 是第 1 次运行的单次快照；`repeat_correct`/`n_repeats` 给出 N 次中判对的次数。`agent_tokens` / `agent_latency_ms` 是 N 次累计。
- `attribution_ok` 口径与 Fake 层一致（T-014）：期望 `target_step == "human"` 的用例，决策是 `escalate` 即算对；否则看 `target_step` 是否等于期望。
- 规则基线是同一批用例上用 `fake_attributor`（朴素）、`smart_rule_attributor`（聪明） 与 `search_rule_attributor`（搜索）跑出来的确定性结果；`baseline_ok` / `smart_baseline_ok` / `search_baseline_ok` 为 `n/a` 表示该用例不进 `attribute`（检索为空）。

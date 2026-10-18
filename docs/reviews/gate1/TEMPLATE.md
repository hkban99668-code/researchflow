# Gate 1 测试结果 · 第 N 轮

- 日期：
- 执行者：（代码实现 agent / 本人）
- Git commit：`git rev-parse --short HEAD` 的输出
- 本轮相对上轮的改动：（对应评审意见编号，如 P0-3、P0-4）
- 标准来源：docs/reviews/gate1/gate1-framework-review.md 附录 A

## 1. 基线 B-1 ~ B-6

| ID | 退出码 | 关键输出（只贴汇总行） | 自评 达标/未达标 |
| --- | --- | --- | --- |
| B-1 | | | |
| B-2 | | noqa 数量： | |
| B-3 | | 检查文件数： | |
| B-4 | | `N passed, N xfailed, N xpassed ...` | |
| B-5 | | run_id_1= / run_id_2= | |
| B-6 | | | |

B-5 checksum 对照：

| 文件 | 第 1 次 sha256 | 第 2 次 sha256 | 一致 |
| --- | --- | --- | --- |
| cleaned_data.csv | | | |
| compute_result.json | | | |
| chart.png | | | |
| validation_report.json | | | |

## 2. 用例结果

| ID | 用例位置（文件::函数） | 结果 PASS/FAIL/XFAIL/XPASS/未实现 | 与附录 A 预期是否一致 | 备注 |
| --- | --- | --- | --- | --- |
| T-P0-1 | | | | |
| T-P0-2 | | | | |
| T-P0-3 | | | | |
| T-P0-4 | | | | |
| T-P0-4b | | | | |
| T-P0-5 | | | | |
| T-P0-6 | | | | |

（第 3 轮再追加 T-P1-* 行）

## 3. 失败或不符合预期的用例详情

每条贴：用例 ID、断言失败信息（不超过 20 行）、你认为的原因。完整输出放 `artifacts/roundN/`。

## 4. 实现中偏离评审建议的地方

（如有：哪条意见、实际怎么做、为什么）

## 5. 评审结论（由评审方填写）

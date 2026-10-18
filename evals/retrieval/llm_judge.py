"""用两个模型独立给候选池打相关性标签（模型当评委），代替人工标注。

用法（会真实调用模型，必须显式确认）：

    uv run python -m evals.retrieval.llm_judge --confirm-real-calls

设计要点：
- **评委不能是被评测的模型。** 检索 Agent 用的是 deepseek，评委用两个 qwen 型号，
  避免"自己评自己"的偏好。两个评委同属 qwen 家族，独立性有限，这一点在报告里写明。
- 评委只看课题、标题、摘要，和人工标注的规则一样："做这个课题写综述时，会不会愿意
  点开读一读"。评委看不到文献是哪种检索方法找到的。
- 结果写到新文件 `labels_judge.csv`，不改动冻结的 `labels_todo.csv`。每行给出两个评委
  的标签、是否一致，以及两者都同意时的共识标签；不一致的行 `consensus` 留空，
  可以交给人工复核，也可以在算分时排除。
- 同一行的判断结果按 (模型, ref_id, 课题) 缓存，重跑不会重复花钱。
"""

from __future__ import annotations

from .. import _env

# 与 .env 隔离（T-031 起统一到 evals._env）：必须在 import researchflow 之前调用；
# 真实层用 _env.load_real_settings() 让命令行 RF_LLM_* 优先于 .env 并打印生效配置。
_env.isolate_dotenv()

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

from researchflow.core.errors import LLMError  # noqa: E402
from researchflow.core.llm import OpenAICompatClient  # noqa: E402

RETRIEVAL_DIR = Path(__file__).resolve().parent
LABELS_TODO = RETRIEVAL_DIR / "labels_todo.csv"
LABELS_JUDGE = RETRIEVAL_DIR / "labels_judge.csv"
CACHE_DIR = RETRIEVAL_DIR.parents[1] / ".cache" / "llm_judge"
JUDGES = ("qwen3.6-flash-2026-04-16", "qwen3.5-flash-2026-02-23")
# 关闭思考模式：判断相关性用不到长推理；开着时实测每次约 4500 token，关掉后成本约降一个数量级。
# 写进缓存键，开关不同的结果不会混在一起。
EXTRA_BODY = {"enable_thinking": False}

PROMPT = """你是严谨的科研文献评审。判断下面这篇文献对这个研究课题是否"相关"。

判断标准只有一条：如果一个研究者正在做这个课题、准备写文献综述，他会不会愿意点开这篇文献读一读？会就是相关，不会就是不相关。
- 只根据下面给出的标题和摘要判断，不要依赖你对这篇文献的其他记忆。
- 只是关键词字面相同、但研究的问题不同（例如同样写 quality control，一个是数据标注质量，一个是工厂产品质检），算不相关。
- 摘要为空、仅凭标题无法判断与课题有关时，算不相关。

课题：{topic}
文献标题：{title}
文献摘要：{abstract}
"""


class Judgment(BaseModel):
    relevant: bool = Field(description="是否相关")
    reason: str = Field(description="一句话理由，不超过 40 字")


def _cache_path(model: str, extra_body: dict | None, ref_id: str, topic: str) -> Path:
    key = hashlib.sha256(f"{model}|{extra_body}|{ref_id}|{topic}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def judge_row(
    client: OpenAICompatClient, row: dict[str, str], extra_body: dict | None = EXTRA_BODY
) -> Judgment:
    path = _cache_path(client.model, extra_body, row["ref_id"], row["topic"])
    if path.exists():
        return Judgment.model_validate_json(path.read_text(encoding="utf-8"))
    prompt = PROMPT.format(
        topic=row["topic"], title=row["title"], abstract=row["abstract"] or "（摘要为空）"
    )
    # 网络偶发断开（SSL EOF、连接被拒）时重试；缓存保证已完成的行不会重复调用。
    for attempt in range(1, 4):
        try:
            result = client.complete(prompt, Judgment)
            break
        except LLMError:
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.retrieval.llm_judge")
    parser.add_argument("--confirm-real-calls", action="store_true")
    parser.add_argument(
        "--judges", nargs=2, default=list(JUDGES), help="两个评委模型名（默认两个 qwen）"
    )
    parser.add_argument("--labels", type=Path, default=LABELS_TODO, help="待判断的标注表")
    parser.add_argument("--out", type=Path, default=LABELS_JUDGE, help="输出 CSV 路径")
    parser.add_argument(
        "--no-extra-body",
        action="store_true",
        help="不发送 enable_thinking 参数（非百炼服务商用）",
    )
    args = parser.parse_args(argv)
    judges = tuple(args.judges)
    extra_body = None if args.no_extra_body else EXTRA_BODY

    rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    if not args.confirm_real_calls:
        print(f"预演：{len(rows)} 行 × {len(judges)} 个评委 = {len(rows) * len(judges)} 次调用")
        print("加上 --confirm-real-calls 才会真正调用模型")
        return 0

    # 真实层：命令行 RF_LLM_* 优先于 .env，生效配置与来源由 load_real_settings 打印。
    settings = _env.load_real_settings()
    if not settings.llm.api_key:
        print("缺少 RF_LLM_API_KEY", file=sys.stderr)
        return 2
    clients = [
        OpenAICompatClient(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            model=model,
            timeout_seconds=settings.llm.timeout_seconds,
            structured_output="json_schema",
            extra_body=extra_body,
        )
        for model in judges
    ]

    out_rows = []
    tokens = 0
    for index, row in enumerate(rows, start=1):
        labels: list[int] = []
        reasons: list[str] = []
        for client in clients:
            judgment = judge_row(client, row, extra_body)
            if client.last_usage is not None:
                tokens += client.last_usage.total_tokens
            labels.append(1 if judgment.relevant else 0)
            reasons.append(judgment.reason)
        agree = labels[0] == labels[1]
        out_rows.append(
            {
                "topic_id": row["topic_id"],
                "ref_id": row["ref_id"],
                "title": row["title"],
                "judge_a": labels[0],
                "judge_b": labels[1],
                "agree": int(agree),
                "consensus": labels[0] if agree else "",
                "reason_a": reasons[0],
                "reason_b": reasons[1],
            }
        )
        if index % 30 == 0:
            print(f"{index}/{len(rows)} 行，累计约 {tokens} token", flush=True)

    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    agree_n = sum(int(r["agree"]) for r in out_rows)
    print(f"完成：{len(out_rows)} 行，两个评委一致 {agree_n} 行（{agree_n / len(out_rows):.1%}）")
    print(f"本次新调用约 {tokens} token；输出 {args.out}")
    print(json.dumps({"judges": judges}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

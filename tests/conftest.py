"""pytest 公共 fixture。

Gate 1 的测试全部使用 Fake 适配器与确定性 fixture，不调用付费 API。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 测试与 .env 隔离（T-009）：必须在任何 researchflow 模块被导入之前执行。
# - RF_LOAD_DOTENV=0 让 load_settings() 跳过读取 .env；
# - 清掉 RF_LLM_* 让开发者 shell 里 export 的 Key 也不影响测试。
os.environ["RF_LOAD_DOTENV"] = "0"
for _key in [name for name in os.environ if name.startswith("RF_LLM_")]:
    os.environ.pop(_key, None)

# 这些 import 必须放在上面的 env 隔离代码之后（T-009），ruff 按需放行。
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from langgraph.types import Command  # noqa: E402

from researchflow.core.config import RunContext  # noqa: E402
from researchflow.core.schemas import LiteratureRef  # noqa: E402
from researchflow.graphs.orchestrator import build_graph  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
# 让 `from conftest import ...` 在 tests/unit 与 tests/e2e 下都能用：pytest 只会把
# 测试文件所在目录插进 sys.path，tests/ 本身不会。
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))


def _stub_refs() -> list[LiteratureRef]:
    """替身检索器的固定两条文献；`ref_id` 带 `stub:` 前缀，便于和本地实现区分。"""
    return [
        LiteratureRef(
            ref_id="stub:ref-1",
            title="替身文献一",
            source="替主语料",
            year=2020,
            summary="替身摘要一",
            score=2.0,
            retriever="stub",
        ),
        LiteratureRef(
            ref_id="stub:ref-2",
            title="替身文献二",
            source="替主语料",
            year=2021,
            summary="替身摘要二",
            score=1.0,
            retriever="stub",
        ),
    ]


class StubRetriever:
    """`Retriever` 的测试替身：忽略查询，原样返回预设文献（不传则两条）。

    `refs=[]` 就得到"完全没命中"的检索器，用来替代原来 monkeypatch
    `search_local` 返回 `[]` 的构造方式。
    """

    def __init__(self, refs: list[LiteratureRef] | None = None) -> None:
        self.refs = _stub_refs() if refs is None else list(refs)

    @property
    def name(self) -> str:
        return "stub"

    def search(self, query: str, *, k: int) -> list[LiteratureRef]:
        return self.refs[:k]


# --- P1-1 之后的统一调用约定 -------------------------------------------------
#
# 运行上下文不再塞进 `config["configurable"]["ctx"]`，而是走图的 `context_schema`
# 逐次调用传入（`graph.invoke(state, config, context=ctx)`），节点签名是
# `node(state, *, runtime)`。下面三个函数是全仓库测试唯一的构造入口。


def runtime_of(ctx: RunContext | None = None) -> Runtime:
    """把运行上下文包成 LangGraph 的 `Runtime`（节点第二个参数，仅限关键字）。"""
    return Runtime(context=ctx)


def thread_config(thread_id: str) -> dict:
    """图的 config：只剩 `thread_id`，上下文走 `context=` 参数。"""
    return {"configurable": {"thread_id": thread_id}}


def drive_chain(
    state: dict,
    ctx: RunContext,
    *,
    thread_id: str,
    decision: str = "approve",
    reason: str = "unit",
    checkpointer=None,
) -> dict:
    """跑完 ①→⑦：第一段停在审核处，再按 `decision` 恢复，返回最终 state。"""
    graph = build_graph(checkpointer)
    config = thread_config(thread_id)
    graph.invoke(state, config, context=ctx)
    return graph.invoke(Command(resume={"decision": decision, "reason": reason}), config)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def minimal_csv(fixture_dir: Path) -> Path:
    return fixture_dir / "minimal_dataset.csv"


@pytest.fixture
def signal_frame() -> pd.DataFrame:
    """带明确线性信号的确定性数据框，用于 DIRECT 与清洗测试。"""
    rng = np.random.default_rng(7)
    n = 80
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    x3 = rng.uniform(1, 3, n)
    y = 2.0 * x1 - 1.5 * x2 + 0.5 * x3 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


@pytest.fixture
def dirty_frame(signal_frame: pd.DataFrame) -> pd.DataFrame:
    """含缺失值与重复行的输入，用于清洗测试。"""
    frame = signal_frame.copy()
    frame.loc[0, "x2"] = np.nan
    frame.loc[1, "x3"] = np.nan
    return pd.concat([frame, frame.iloc[[2]]], ignore_index=True)


@pytest.fixture
def data_rules() -> dict:
    return {
        "required_columns": ["x1", "x2", "x3", "y"],
        "max_missing_rate": 0.1,
        "min_rows": 20,
        "ranges": {"x1": {"min": 0, "max": 10}},
    }


@pytest.fixture
def stub_retriever() -> StubRetriever:
    """返回固定两条文献的检索替身。"""
    return StubRetriever()


@pytest.fixture
def empty_retriever() -> StubRetriever:
    """什么都没命中的检索替身。"""
    return StubRetriever(refs=[])


@pytest.fixture
def submission_rules() -> dict:
    return {
        "metric_thresholds": {"r2": {"min": 0.8}},
        "required_sections": ["背景", "方法", "结果", "局限"],
        "min_draft_chars": 200,
        "required_artifacts": ["cleaned_data", "chart", "draft"],
        "forbidden_tokens": ["TODO", "待补充"],
        "owner_steps": {
            "metric": "compute",
            "section": "writing",
            "draft_min_chars": "writing",
            "forbidden_tokens": "writing",
            "artifact:cleaned_data": "data_review",
            "artifact:chart": "chart",
            "artifact:draft": "writing",
            "variables_in_dataset": "extraction",
            # T-029 引用硬规则：写草稿的 ⑥ 负责
            "citations_subset": "writing",
            "citation_has_claim": "writing",
        },
    }

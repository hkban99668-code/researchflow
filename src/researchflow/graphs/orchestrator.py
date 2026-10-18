"""Gate 1 链路编排：①→⑦ 与 CLI 人工审核断点。

图只描述结构，不持有状态：checkpointer 由 `make_checkpointer()` 按配置构造
（memory 或 sqlite），运行上下文走 `context_schema`（每次调用传入、不进 checkpoint），
因此同一份磁盘状态可以在另一个进程里继续跑完（P1-1）。
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from types import ModuleType

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ..core import schemas
from ..core.config import CHECKPOINT_BACKENDS, RunContext
from ..core.errors import ConfigError
from ..core.schemas import BaseModel
from ..core.state import GraphState
from ..nodes.attribution import attribute_node
from ..nodes.chart import chart_node
from ..nodes.compute import compute_node
from ..nodes.data_pipeline import data_prepare_node, data_review_node
from ..nodes.extraction import extraction_node
from ..nodes.retrieval import retrieval_node
from ..nodes.validation import validation_node
from ..nodes.writing import writing_node

# 每个 run 一个库文件，放在 run 目录下：拷走整个目录就能在别处恢复，
# 删掉目录就一起清掉。代价是不能跨 run 查询，Gate 1 用不到。
CHECKPOINT_FILENAME = "checkpoint.sqlite"


def collect_msgpack_modules(module: ModuleType | None = None) -> list[tuple[str, str]]:
    """收集进入 checkpoint 白名单的自定义类型：`(模块名, 类名)` 列表。

    遍历 `researchflow.core.schemas` 里**本模块定义**的 `BaseModel` 子类，不再手工
    维护列表：手工列表漏掉嵌套类型（`QualitySummary` / `CheckResult`）时，反序列化
    会退化成字典并丢字段。只收 `__module__` 等于本模块的，避免把 import 进来的
    pydantic 基类之类也收进来。
    """
    target = module if module is not None else schemas
    found = [
        (value.__module__, value.__name__)
        for _, value in inspect.getmembers(target, inspect.isclass)
        if issubclass(value, BaseModel) and value.__module__ == target.__name__
    ]
    return sorted(set(found))


# 显式登记进入 checkpoint 的自定义类型，避免反序列化告警与未来版本的阻断。
ALLOWED_MSGPACK_MODULES = collect_msgpack_modules()


def make_serializer() -> JsonPlusSerializer:
    """checkpoint 序列化器：白名单由 `collect_msgpack_modules()` 自动收集。"""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


def make_checkpointer(backend: str, run_dir: Path | None = None) -> BaseCheckpointSaver:
    """按后端构造 checkpointer。

    sqlite 后端需要 `run_dir`：文件写在 `run_dir/checkpoint.sqlite`。用毕要调
    `close_checkpointer()`——写入虽然每次都提交，但 WAL 模式下数据还留在
    `-wal` 里，不关连接主库文件就"看起来是空的"，拷走目录会丢状态。
    """
    if backend == "memory":
        return InMemorySaver(serde=make_serializer())
    if backend == "sqlite":
        if run_dir is None:
            raise ConfigError("sqlite checkpointer 需要 run_dir")
        path = Path(run_dir) / CHECKPOINT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：LangGraph 可能在别的线程里读写同一个 saver。
        return SqliteSaver(
            sqlite3.connect(str(path), check_same_thread=False), serde=make_serializer()
        )
    raise ConfigError(
        f"checkpoint.backend 取值非法: {backend!r}（只能是 {' / '.join(CHECKPOINT_BACKENDS)}）"
    )


def close_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """关闭 sqlite 连接（memory 后端是空操作）。

    连接关闭时 SQLite 会把 WAL 合并回主库文件并删掉 `-wal`，`checkpoint.sqlite`
    才是一个自包含的文件。
    """
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        conn.close()


def route_next(next_step: str):
    """节点之后的终止路由：error 进 attribute，其他终止进 END，否则走向下一个节点。"""

    def router(state: dict) -> str:
        termination = state.get("termination")
        if termination is None:
            return next_step
        return "attribute" if termination.kind == "error" else END

    return router


def route_after_validation(state: dict) -> str:
    """⑦ 之后：校验通过进 END，不通过进 attribute 归因回溯。"""
    validation = state.get("validation_result")
    if validation is not None and not validation.passed:
        return "attribute"
    return END


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """编译图；不传 checkpointer 时用进程内的 memory（测试与一次性运行）。"""
    builder = StateGraph(GraphState, context_schema=RunContext)

    builder.add_node("retrieval", retrieval_node)
    builder.add_node("extraction", extraction_node)
    builder.add_node("data_prepare", data_prepare_node)
    builder.add_node("data_review", data_review_node)
    builder.add_node("compute", compute_node)
    builder.add_node("chart", chart_node)
    builder.add_node("writing", writing_node)
    builder.add_node("validation", validation_node)
    builder.add_node("attribute", attribute_node)

    builder.add_edge(START, "retrieval")
    for source, target in (
        ("retrieval", "extraction"),
        ("extraction", "data_prepare"),
        ("data_prepare", "data_review"),
        ("data_review", "compute"),
        ("compute", "chart"),
        ("chart", "writing"),
        ("writing", "validation"),
    ):
        builder.add_conditional_edges(
            source, route_next(target), {target: target, END: END, "attribute": "attribute"}
        )
    builder.add_conditional_edges(
        "validation", route_after_validation, {"attribute": "attribute", END: END}
    )
    # attribute 没有出边：去向由节点返回的 Command(goto=...) 决定（回溯到
    # data_prepare/compute，或 goto=END 转人工 / 预算耗尽）。

    return builder.compile(checkpointer=checkpointer or make_checkpointer("memory"))

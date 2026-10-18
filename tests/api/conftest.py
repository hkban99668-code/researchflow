"""API 测试公共 fixture。

测试只用 fake 模式与仓库内 fixture 数据集，不发任何真实请求（任务卡"不做"）。
共享工具函数在 `helpers.py`（与上层 tests/conftest.py 避免 import 撞名）。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from researchflow.api import create_app
from researchflow.api import sse as sse_module
from researchflow.core.config import Settings, load_settings

# SSE 轮询缩到毫秒级（app 在请求时读模块常量）；keepalive 保持默认 15 秒，
# 专项测试直接给 iter_events 传小值，不真等。
sse_module.POLL_INTERVAL_SECONDS = 0.01


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def api_settings(tmp_path, project_root) -> Settings:
    """私有 settings：runs_dir 指向 tmp_path；.env 已被顶层 conftest 隔离。"""
    settings = load_settings(project_root=project_root)
    return replace(settings, runs_dir=tmp_path / "runs")


@pytest.fixture
def api_env(api_settings: Settings):
    """一个独立应用（独立线程池）。用完等线程收尾再关，不把 Future 泄漏进解释器退出。"""
    app = create_app(api_settings)
    yield app
    app.state.manager.shutdown()

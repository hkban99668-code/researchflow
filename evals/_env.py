"""eval 入口共用的 .env 隔离与真实层配置加载（T-031，issues-log F7/E4）。

背景：`evals/run.py` 与 `evals/citation/run.py` 曾各自复制"模块加载时保存并清掉
shell 里的 `RF_LLM_*`、真实层再写回"的代码，修复不同步导致命令行指定的服务商被
`.env` 悄悄换掉。本模块把这件事统一成一份实现，全部 eval 入口共用：

1. `isolate_dotenv()`：保存 shell 里显式给的 `RF_LLM_*` / `RF_EMBED_*`，设
   `RF_LOAD_DOTENV=0`（隔离 `.env` 对 Fake 层与测试的影响），再清掉这些变量。
   **每个 eval 入口必须在 `import researchflow` 之前调用它**（与 T-009 conftest
   同法；`load_dotenv` 发生在 `load_settings()` 里，晚于此调用即安全）。
2. `load_real_settings()`：真实层专用——写回 shell 覆盖值（dotenv 默认
   `override=False`，于是命令行值优先于 `.env`）→ 读 `.env` → 打印实际生效的
   模型与端点及其来源（命令行 / .env / 默认）→ 返回 `Settings`。

来源判定：`_SHELL_OVERRIDES` 里的键是"命令行"（shell/环境变量显式给的）；
`load_dotenv` 之后 `os.environ` 里新出现的键来自".env"；都没有是"默认"。
隔离时清掉了 shell 值、写回又先于 `load_dotenv`，三态互斥、判定可靠。
"""

from __future__ import annotations

import os
from pathlib import Path

# shell/命令行里显式给的覆盖值（isolate_dotenv 时保存，真实层写回）。
_SHELL_OVERRIDES: dict[str, str] = {}


def isolate_dotenv() -> None:
    """保存并清掉 shell 里的 `RF_LLM_*` / `RF_EMBED_*`，设 `RF_LOAD_DOTENV=0`。

    幂等：重复调用不会用已清空的环境覆盖首次保存的值。
    """
    if _SHELL_OVERRIDES:
        return
    _SHELL_OVERRIDES.update(
        {name: os.environ[name] for name in os.environ if name.startswith(("RF_LLM_", "RF_EMBED_"))}
    )
    os.environ["RF_LOAD_DOTENV"] = "0"
    for key in [name for name in os.environ if name.startswith(("RF_LLM_", "RF_EMBED_"))]:
        os.environ.pop(key, None)


# 导入即隔离：本模块被 import 时就完成保存与清理，入口只要保证"先 import 本模块
# 再 import researchflow"即可；入口里的显式 `isolate_dotenv()` 调用是幂等的、
# 只为可读性。
isolate_dotenv()

from researchflow.core.config import Settings, load_settings, sanitize_base_url  # noqa: E402


def _source_of(key: str) -> str:
    """某项配置的来源：命令行（shell 显式给的）/ .env（load_dotenv 写入）/ 默认。"""
    if key in _SHELL_OVERRIDES:
        return "命令行"
    if key in os.environ:
        return ".env"
    return "默认"


def print_effective_config(settings: Settings) -> None:
    """打印实际生效的模型与端点及其来源；不打印 Key 值（base_url 脱敏）。"""
    print(f"模型名: {settings.llm.model or '(未配置)'}（来源: {_source_of('RF_LLM_MODEL')}）")
    print(
        f"base_url: {sanitize_base_url(settings.llm.base_url)}"
        f"（来源: {_source_of('RF_LLM_BASE_URL')}）"
    )
    print(
        f"structured_output: {settings.llm.structured_output}"
        f"（来源: {_source_of('RF_LLM_STRUCTURED_OUTPUT')}）"
    )
    embed_base_url = sanitize_base_url(settings.embedding.base_url) or "(回落 llm)"
    print(
        f"嵌入模型: {settings.embedding.model or '(未配置)'}"
        f"（来源: {_source_of('RF_EMBED_MODEL')}）"
    )
    print(f"嵌入 base_url: {embed_base_url}（来源: {_source_of('RF_EMBED_BASE_URL')}）")


def load_real_settings(project_root: Path | None = None) -> Settings:
    """真实层读配置：写回 shell 覆盖值 → 读 `.env`（命令行优先）→ 打印生效配置。

    Fake 层不经过此函数、行为不变。
    """
    os.environ.update(_SHELL_OVERRIDES)
    settings = load_settings(project_root=project_root, load_env_file=True)
    print_effective_config(settings)
    return settings

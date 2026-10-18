"""T-015 A3：提示词全文不出现 cases.yaml 里的任何 fault 名称或 id。"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.run import load_cases  # noqa: E402


def test_prompt_contains_no_fault_names_or_ids(project_root):
    """A3：扫描提示词全文，任何 fault 名称或用例 id 都不该出现。"""
    prompt = (project_root / "configs" / "prompts" / "attribution_system.md").read_text(
        encoding="utf-8"
    )
    cases = load_cases()
    for case in cases:
        fault = case["fault"]
        case_id = case["id"]
        assert fault not in prompt, f"提示词出现故障名: {fault!r}"
        assert case_id not in prompt, f"提示词出现用例 id: {case_id!r}"

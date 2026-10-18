"""T-011 A2：同一种子生成的 CSV 逐字节相同。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# 让 `import evals` 可用：evals/ 在仓库根目录，不在 src/ 下，也不在 tests/ 的导入链里。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import faults  # noqa: E402


def _csv_sha256(frame, tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_same_csv_bytes(tmp_path):
    """A2：同一用例（同故障、同种子、同参数）生成两次，CSV 逐字节相同。"""
    frame = faults.load_sample()
    params = {"column": "x2", "rate": 0.3}
    a = faults.apply_fault("high_missing", frame, 101, params)
    b = faults.apply_fault("high_missing", frame, 101, params)
    assert _csv_sha256(a, tmp_path, "a") == _csv_sha256(b, tmp_path, "b")


def test_all_data_faults_deterministic():
    """每个数据故障在固定种子上都确定性（两次结果逐字段相等）。"""
    frame = faults.load_sample()
    for name in faults.DATA_FAULTS:
        a = faults.apply_fault(name, frame, 42, {})
        b = faults.apply_fault(name, frame, 42, {})
        assert a.equals(b), name


def test_different_seed_gives_different_data():
    """不同种子应产出不同数据（抽样/打乱类故障，避免种子失效却仍判"确定性"）。"""
    frame = faults.load_sample()
    a = faults.apply_fault("y_shuffled", frame, 301, {})
    b = faults.apply_fault("y_shuffled", frame, 302, {})
    assert not a["y"].equals(b["y"])

"""T-035：容器资产的结构性检查。

真正"构建 → 起服务 → 跑 fake"的验证由 `scripts/smoke_docker.sh`（CI 的 docker
job）完成；这里只锁住容易回退的约定：密钥/运行数据不进构建上下文、镜像非 root
且有健康检查、compose 只绑本机并用命名卷。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _compose(name: str = "docker-compose.yml") -> dict[str, Any]:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def _dockerignore() -> set[str]:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def test_dockerignore_excludes_secrets_and_runtime_data():
    """A1：.env*、runs/、.cache/、.venv/、.git/ 都不能进构建上下文。"""
    ignored = _dockerignore()
    for pattern in (".env", ".env*", "runs/", ".cache/", ".venv/", ".git/"):
        assert pattern in ignored


def test_dockerfile_uses_locked_uv_non_root_and_healthcheck():
    """A1/A2：按锁文件安装、非 root 运行、健康检查打 /healthz、带配置与示例。"""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uv sync --frozen --no-dev" in text
    assert "USER app" in text
    assert "HEALTHCHECK" in text and "/healthz" in text
    assert "MPLBACKEND=Agg" in text
    for asset in ("configs", "examples"):
        assert f"COPY --chown=app:app {asset}" in text


def test_compose_api_binds_localhost_and_persists_runs_volume():
    """A2/A4：只发布到 127.0.0.1，runs 用命名卷；默认 Fake 且不读 .env。"""
    compose = _compose()
    api = compose["services"]["api"]
    assert api["ports"] == ["127.0.0.1:${RF_HTTP_PORT:-8000}:8000"]
    assert "runs:/app/runs" in api["volumes"]
    assert "runs" in compose["volumes"]
    assert api["environment"]["RF_LLM_MODE"] == "fake"
    assert api["environment"]["RF_LOAD_DOTENV"] == "0"


def test_compose_demo_profile_runs_fake_with_auto_approve():
    """A5：demo 是一次性 profile，显式 fake + auto-approve，产出五类产物。"""
    demo = _compose()["services"]["demo"]
    assert demo["profiles"] == ["demo"]
    command = list(demo["command"])
    assert command[command.index("--mode") + 1] == "fake"
    assert "--auto-approve" in command
    assert "runs:/app/runs" in demo["volumes"]


def test_ci_runs_container_smoke_script():
    """A6：CI 有独立的 docker job，跑同一个冒烟脚本。"""
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    job = ci["jobs"]["docker"]
    runs = [step.get("run", "") for step in job["steps"]]
    assert any("scripts/smoke_docker.sh" in run for run in runs)


def test_readme_documents_container_workflow():
    """README 容器化一节给出起服务、demo、真实模式与持久化说明。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for token in (
        "### 3.8 容器化",
        "docker compose up -d",
        "--profile demo run --rm demo",
        "127.0.0.1",
        "researchflow_runs",
        "docker-compose.real.yml",
    ):
        assert token in readme

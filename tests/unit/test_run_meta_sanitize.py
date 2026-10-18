"""T-009 A8、A9：恢复时校验 model/structured_output，run_meta 的 base_url 脱敏。"""

from __future__ import annotations

import json

import pytest

from researchflow.core.config import (
    RunContext,
    load_settings,
    run_meta_payload,
    sanitize_base_url,
    write_run_meta,
)
from researchflow.core.errors import ConfigError

DEMO_TOPIC = "examples/demo_topic.yaml"


def _settings(tmp_path, project_root):
    settings = load_settings(project_root=project_root)
    settings.runs_dir = tmp_path / "runs"
    settings.llm.mode = "fake"
    return settings


def _write_meta(settings, run_id, project_root, *, llm_mode: str = "fake"):
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(
        run_dir,
        run_meta_payload(
            run_id=run_id,
            topic_path=project_root / DEMO_TOPIC,
            llm_mode=llm_mode,
            settings=settings,
            checkpoint_backend="sqlite",
        ),
    )
    return run_dir


def test_resume_rejects_model_mismatch_in_real_mode(tmp_path, project_root):
    """A8（T-009）：real 模式恢复时 llm.model 与首次不同，抛 ConfigError，消息含 model。"""
    settings = _settings(tmp_path, project_root)
    settings.llm.mode = "real"
    settings.llm.api_key = "sk-test"
    settings.llm.model = "model-a"
    _write_meta(settings, "m1", project_root, llm_mode="real")

    settings.llm.model = "model-b"
    with pytest.raises(ConfigError, match="model"):
        RunContext.from_run_id("m1", settings)


def test_resume_rejects_structured_output_mismatch_in_real_mode(tmp_path, project_root):
    """A8（T-009）：real 模式恢复时 structured_output 与首次不同，抛 ConfigError。"""
    settings = _settings(tmp_path, project_root)
    settings.llm.mode = "real"
    settings.llm.api_key = "sk-test"
    settings.llm.structured_output = "json_schema"
    _write_meta(settings, "m2", project_root, llm_mode="real")

    settings.llm.structured_output = "json_object"
    with pytest.raises(ConfigError, match="structured_output"):
        RunContext.from_run_id("m2", settings)


def test_resume_skips_model_and_structured_output_checks_in_fake_mode(tmp_path, project_root):
    """B17：Fake 模式恢复不校验 llm.model / structured_output。"""
    settings = _settings(tmp_path, project_root)
    settings.llm.model = "model-a"
    settings.llm.structured_output = "json_schema"
    _write_meta(settings, "m4", project_root, llm_mode="fake")

    settings.llm.model = "model-b"
    settings.llm.structured_output = "json_object"
    ctx = RunContext.from_run_id("m4", settings)  # 不抛错

    assert ctx.llm.mode == "fake"


def test_sanitize_base_url():
    """A9：userinfo 去掉、路径去掉、第一段子域名换成 *。"""
    url = "https://user:pass@llm-abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    assert sanitize_base_url(url) == "https://*.cn-beijing.maas.aliyuncs.com"


def test_run_meta_base_url_sanitized(tmp_path, project_root):
    """A9：run_meta.json 里的 base_url 脱敏，不含 abc123 / user / pass。"""
    settings = _settings(tmp_path, project_root)
    settings.llm.base_url = (
        "https://user:pass@llm-abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    run_dir = _write_meta(settings, "m3", project_root)

    raw = (run_dir / "run_meta.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["settings"]["llm"]["base_url"] == "https://*.cn-beijing.maas.aliyuncs.com"
    assert "abc123" not in raw
    assert "user:pass" not in raw


def test_run_meta_extra_body_sensitive_keys_masked(tmp_path, project_root):
    """T-015 裁定：extra_body 键名含 key/token/secret/authorization 的项替换成 ***。"""
    settings = _settings(tmp_path, project_root)
    settings.llm.extra_body = {
        "enable_thinking": False,
        "api_key": "leak-me",
        "x_api_token": "tok",
        "client_secret": "sec",
        "Authorization": "bearer-xyz",
    }
    run_dir = _write_meta(settings, "m5", project_root)

    raw = (run_dir / "run_meta.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    extra = payload["settings"]["llm"]["extra_body"]
    assert extra["enable_thinking"] is False
    assert extra["api_key"] == "***"
    assert extra["x_api_token"] == "***"
    assert extra["client_secret"] == "***"
    assert extra["Authorization"] == "***"
    assert "leak-me" not in raw
    assert "bearer-xyz" not in raw

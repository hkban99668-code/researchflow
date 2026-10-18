"""LLM 适配器：Fake（CI/e2e）与 OpenAI 兼容 HTTP（本地真实冒烟）共用同一契约。

- 两个实现都返回同一 Pydantic 模型，保证 e2e 与真实冒烟语义一致。
- Gate 1 只接一家 Provider，不做 Provider 抽象层。
- 未配置 API Key 时自动降级 Fake，保证无密钥环境也能跑通链路。

两条调用路径：

- `complete()` 要结构化输出：优先 `response_format={"type": "json_schema", "strict": true}`，
  Provider 不支持时降级 `json_object` + prompt 内 schema（配置项 `structured_output`）。
- `chat()` 要原生 tool calling：**只把模型的 `tool_calls` 交回来，不执行任何工具**。
  执行、校验、权限都在 `agent_loop.run_tool_loop` 里，权限控制因此只有唯一落点。
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings
from .errors import ConfigError, LLMError
from .schemas import (
    AssistantTurn,
    DraftContent,
    LLMCallRecord,
    LLMUsage,
    ResearchQuestion,
    ToolCall,
)
from .tools import strict_json_schema

T = TypeVar("T", bound=BaseModel)

_JSON_INSTRUCTION = (
    "\n\n只输出一个 JSON 对象，不要输出解释或代码块。JSON 必须满足以下 JSON Schema：\n"
)
_RETRY_INSTRUCTION = (
    "上次输出未通过结构化校验：{error}\n"
    "请根据以上错误修正，重新只输出一个合法的 JSON 对象，不要输出解释或代码块。"
)
_MAX_ERROR_DIGEST = 300

_TYPE_LABELS = {
    "string": "字符串",
    "number": "数值",
    "integer": "整数",
    "boolean": "布尔",
    "array": "数组",
    "object": "对象",
}


def _type_label(spec: dict[str, Any]) -> str:
    """把 JSON Schema 的类型值转成中文标签；nullable 的列表取非 null 那一支。"""
    raw = spec.get("type")
    if isinstance(raw, list):
        raw = next((item for item in raw if item != "null"), raw[0] if raw else "object")
    return _TYPE_LABELS.get(str(raw), str(raw))


def _field_notes(response_model: type[BaseModel]) -> str:
    """json_schema 模式下追加到 prompt 末尾的精简字段说明（不拼完整 JSON Schema）。

    每个字段一行，形如 `- 字段名（类型，必填/可选）：描述`，由 `response_model`
    的 schema 自动生成。C1 的根因是 json_schema 模式下把 schema 从 prompt 去掉了，
    遇到「接受 response_format 却不执行约束」的 Provider 首轮就不知道要输出哪些字段。
    """
    schema = response_model.model_json_schema()
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    lines = ["\n\n请输出一个 JSON 对象，字段如下："]
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        kind = "必填" if name in required else "可选"
        description = spec.get("description") or ""
        suffix = f"：{description}" if description else ""
        lines.append(f"- {name}（{_type_label(spec)}，{kind}）{suffix}")
    return "\n".join(lines)


class LLMClient(Protocol):
    """LLM 适配器契约：返回值不变，用量经 `last_usage` 旁路带出。"""

    @property
    def last_usage(self) -> LLMUsage | None:
        """上一次 complete() 的累计用量；调用前为 None。"""

    @property
    def structured_output(self) -> str | None:
        """结构化输出的取值方式：`json_schema` / `json_object`，不支持时为 None。

        进 `LLMCallRecord.structured_output`，供以后按模式统计"首轮合法率"。
        """

    @property
    def last_duration_ms(self) -> int | None:
        """上一次 `complete()` / `chat()` 的墙钟耗时（含结构化重试）；调用前为 None。

        Fake 适配器恒为 0，保证 trace 的确定性；真实适配器为实际耗时。
        """

    def complete(
        self,
        prompt: str,
        response_model: type[T],
        context: dict[str, Any] | None = None,
    ) -> T: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> AssistantTurn:
        """发一次带 tools 的请求，把模型的工具调用意图原样交回来（不执行）。"""

    ...


def _fake_research_question(context: dict[str, Any]) -> dict[str, Any]:
    topic = context.get("topic", "给定课题")
    target = context.get("target", "目标变量")
    features = context.get("features") or ["关键变量"]
    return {
        "question": f"在「{topic}」中，{'、'.join(features)} 如何影响 {target}？",
        "variables": [*features, target],
        "hypothesis": f"{'、'.join(features)} 与 {target} 之间存在可量化的线性关系。",
        "method": "对清洗后的样本使用单算法回归，并报告拟合指标。",
        "confidence": 0.6,
    }


def _fake_draft_content(context: dict[str, Any]) -> dict[str, Any]:
    topic = context.get("topic", "给定课题")
    algorithm = context.get("algorithm", "linear_regression")
    metrics = context.get("metrics") or {}
    metric_text = "、".join(f"{k}={v}" for k, v in metrics.items()) or "无指标"
    # 引用取自本次检索到的文献（至少 1 条，无文献则空），保证确定性与可复现测试：
    # 引用合法（ref_id 就在 literature_refs 里），⑦ 的 citations_subset 因此通过。
    literatures = context.get("literature_refs") or []
    citations = []
    if literatures:
        first = literatures[0]
        citations.append(
            {
                "ref_id": first.ref_id,
                "claim": f"{first.title}为「{topic}」的背景提供了参考。",
            }
        )
    return {
        "background": f"本初稿围绕「{topic}」展开，基于 Gate 1 的固定样本与固定参数完成一次可复现分析。",
        "findings": f"使用 DIRECT 算法 {algorithm} 得到 {metric_text}，结果由固定随机种子保证可复现。",
        "limitation": "样本量有限、仅使用单一算法与固定模板图表，结论需在 Gate 2 补充回溯与多算法对比后扩展。",
        "citations": citations,
    }


_FAKE_BUILDERS: dict[type[BaseModel], Any] = {
    ResearchQuestion: _fake_research_question,
    DraftContent: _fake_draft_content,
}


class FakeLLMClient:
    """确定性适配器：不联网、不计费，供 CI 与 e2e 使用。

    用量按字符数伪造而非除以 4，测试可以直接用 `len()` 精确断言。
    """

    def __init__(self, script: list[AssistantTurn] | None = None) -> None:
        """`script` 是 `chat()` 的回复脚本，按顺序逐条返回；不传则 `chat()` 直接报错。"""
        self.mode = "fake"
        # 结构化输出对 Fake 无意义：它直接构造 Pydantic 模型，不走 response_format。
        self.structured_output: str | None = None
        self.last_usage: LLMUsage | None = None
        # 耗时恒为 0：Fake 不花时间，trace 因此对同一份 state 确定性可复现。
        self.last_duration_ms: int | None = 0
        self.script: list[AssistantTurn] = list(script or [])
        self.chat_requests: list[list[dict[str, Any]]] = []
        self._script_pos = 0

    def complete(
        self,
        prompt: str,
        response_model: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        self.last_usage = None
        self.last_duration_ms = 0
        builder = _FAKE_BUILDERS.get(response_model)
        if builder is None:
            raise LLMError(f"Fake 适配器未登记 {response_model.__name__} 的构造方式")
        model = response_model.model_validate(builder(context or {}))
        self.last_usage = LLMUsage(
            prompt_tokens=len(prompt),
            completion_tokens=len(model.model_dump_json()),
            attempts=1,
        )
        return model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> AssistantTurn:
        # 深拷贝：调用方会原地追加 assistant/tool 消息，快照必须不受影响。
        self.chat_requests.append(deepcopy(messages))
        self.last_duration_ms = 0
        if self._script_pos >= len(self.script):
            raise LLMError("Fake 脚本已耗尽")
        turn = self.script[self._script_pos]
        self._script_pos += 1
        self.last_usage = LLMUsage(
            prompt_tokens=len(json.dumps(messages, ensure_ascii=False)),
            completion_tokens=len(turn.model_dump_json()),
            attempts=1,
        )
        return turn


class OpenAICompatClient:
    """OpenAI 兼容 HTTP 适配器，可用于 OpenAI / DeepSeek / Moonshot / 通义等。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        structured_output: str = "json_schema",
        supports_tools: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        # 结构化输出失败最多重试 max_retries 次（设计方案 788 行：最多重试 2 次）。
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.structured_output = structured_output
        self.supports_tools = supports_tools
        # Provider 专属参数（思考模式开关等），原样合并进每次请求 body。
        self.extra_body = dict(extra_body or {})
        self.mode = "real"
        self.last_usage: LLMUsage | None = None
        self.last_duration_ms: int | None = None

    def complete(
        self,
        prompt: str,
        response_model: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        # json_schema 模式在 response_format 里带 schema，但部分 Provider「接受却不执行」，
        # prompt 里仍保留一份精简字段说明（C1）：不拼完整 schema，只列字段名/类型/描述。
        if self.structured_output == "json_schema":
            schema_text = _field_notes(response_model)
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": strict_json_schema(response_model),
                    "strict": True,
                },
            }
        else:
            schema_text = _JSON_INSTRUCTION + json.dumps(
                response_model.model_json_schema(), ensure_ascii=False
            )
            response_format = {"type": "json_object"}

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "你是严谨的科研助手，只输出结构化 JSON。",
            },
            {"role": "user", "content": prompt + schema_text},
        ]
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": response_format,
            "messages": messages,
        }

        # 累计本次 complete() 全部请求的用量：失败的调用同样花钱，必须算进去。
        usage = LLMUsage(attempts=0)
        self.last_usage = None
        self.last_duration_ms = None
        last_error = ""
        # 计时含结构化重试的全部尝试，出参在 finally 里统一写回。
        started = time.perf_counter()
        try:
            for attempt in range(1, self.max_retries + 2):
                try:
                    payload = self._post(body)
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    # 请求已发出（可能已计费），计入次数；用量无从读取，只累加 attempts。
                    usage = add_usage(usage, LLMUsage(attempts=1))
                    self.last_usage = usage
                    last_error = f"真实 LLM 调用失败: {exc}"
                    if attempt <= self.max_retries and _is_retryable_http(exc):
                        time.sleep(self.retry_backoff * attempt)
                        continue
                    raise LLMError(
                        last_error, detail={"attempts": attempt, "error": str(exc)}
                    ) from exc

                usage = add_usage(usage, _usage_of(payload))
                self.last_usage = usage

                try:
                    return _parse_structured(payload, response_model)
                except (ValidationError, ValueError) as exc:
                    # 把校验错误摘要回传给模型，下一轮据此修正。
                    last_error = str(exc)[:_MAX_ERROR_DIGEST]
                    if attempt <= self.max_retries:
                        messages.append(
                            {
                                "role": "user",
                                "content": _RETRY_INSTRUCTION.format(error=last_error),
                            }
                        )
                        continue
                    raise LLMError(
                        "真实 LLM 输出未通过结构化校验",
                        detail={"attempts": attempt, "error": last_error},
                    ) from exc

            raise LLMError(
                "真实 LLM 输出未通过结构化校验",
                detail={"attempts": self.max_retries + 1, "error": last_error},
            )
        finally:
            self.last_duration_ms = int((time.perf_counter() - started) * 1000)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> AssistantTurn:
        """发一次请求，把正文与 `tool_calls` 原样交回；不做结构化重试，也不执行工具。"""
        if tools and not self.supports_tools:
            raise ConfigError("已配置 RF_LLM_SUPPORTS_TOOLS=0，本次请求却带了 tools")
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        usage = LLMUsage(attempts=0)
        self.last_usage = None
        self.last_duration_ms = None
        last_error = ""
        # chat() 不做结构化重试，但 HTTP 失败仍会退避重试，计时同样覆盖全部尝试。
        started = time.perf_counter()
        try:
            for attempt in range(1, self.max_retries + 2):
                try:
                    payload = self._post(body)
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    usage = add_usage(usage, LLMUsage(attempts=1))
                    self.last_usage = usage
                    last_error = f"真实 LLM 调用失败: {exc}"
                    if attempt <= self.max_retries and _is_retryable_http(exc):
                        time.sleep(self.retry_backoff * attempt)
                        continue
                    raise LLMError(
                        last_error, detail={"attempts": attempt, "error": str(exc)}
                    ) from exc

                usage = add_usage(usage, _usage_of(payload))
                self.last_usage = usage
                return AssistantTurn(
                    content=_content_of(payload),
                    tool_calls=_tool_calls_of(payload),
                    usage=usage,
                )

            raise LLMError(
                last_error, detail={"attempts": self.max_retries + 1, "error": last_error}
            )
        finally:
            self.last_duration_ms = int((time.perf_counter() - started) * 1000)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """发起一次请求并返回响应体（内部函数，形状不对外承诺）。

        `body["messages"]` 与调用方持有的是同一个列表，重试时追加的消息会体现在下一次请求里。
        `extra_body`（思考模式开关等 Provider 专属参数）先合入，标准字段（model/messages 等）
        后写、保持权威，避免用户误覆盖必要字段。
        """
        merged = {**self.extra_body, **body}
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=merged,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"响应体不是 JSON 对象: {type(payload).__name__}")
        return payload


def _content_of(payload: dict[str, Any]) -> str | None:
    """取 `choices[0].message.content`；缺失或为 null 时返回 None。

    不能写成 `str(...)`：OpenAI 在有 `tool_calls` 时 `content` 就是 `null`，
    `str(None)` 会变成字符串 "None" 混进对话里。
    """
    content = _assistant_message_of(payload).get("content")
    return content if isinstance(content, str) else None


def _tool_calls_of(payload: dict[str, Any]) -> list[ToolCall]:
    """解析 `choices[0].message.tool_calls`；没有该字段时为空列表。"""
    raw = _assistant_message_of(payload).get("tool_calls")
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_function = item.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        calls.append(
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=str(function.get("arguments") or ""),
            )
        )
    return calls


def _assistant_message_of(payload: dict[str, Any]) -> dict[str, Any]:
    """取 `choices[0].message`；结构缺失抛 KeyError，由调用方归为"调用失败"。"""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise KeyError("choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise KeyError("message")
    return message


def _parse_structured(payload: dict[str, Any], response_model: type[T]) -> T:
    """把模型正文解析成 `response_model`；strict 模式的可选字段可能是 null。

    strict 要求每个属性都出现，模型就把没话说的可选字段填成 `null`，
    直接 `model_validate` 会因为"null 不是 str"失败，所以先丢掉值为 None 的键，
    让它等价于"未提供"落回默认值（与 `ToolRegistry.invoke` 的做法一致）。
    """
    content = _content_of(payload)
    if content is None:
        raise ValueError("模型未返回正文（content 为 null）")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"结构化输出必须是 JSON 对象，收到 {type(data).__name__}")
    return response_model.model_validate(
        {key: value for key, value in data.items() if value is not None}
    )


def _usage_of(payload: dict[str, Any]) -> LLMUsage:
    """从响应体的 usage 字段读用量；缺失或不是对象时记 0，不报错。"""
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return LLMUsage(attempts=1)
    return LLMUsage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        attempts=1,
    )


def add_usage(left: LLMUsage, right: LLMUsage) -> LLMUsage:
    """两份用量相加；total_tokens 由模型校验器重算。"""
    return LLMUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        attempts=left.attempts + right.attempts,
    )


def summarize_llm_usage(calls: list[LLMCallRecord]) -> dict[str, Any]:
    """把逐条调用记录汇总成 manifest 需要的三层结构。

    返回 `{"calls": [...], "by_step": {步骤: 合计}, "total": 全局合计}`；
    空输入给出 `calls=[]`、`by_step={}` 与全 0 的 total。
    """
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "attempts")
    zero = dict.fromkeys(keys, 0)
    by_step: dict[str, dict[str, int]] = {}
    total = dict(zero)
    for call in calls:
        usage = call.usage.model_dump()
        bucket = by_step.setdefault(call.step, dict(zero))
        for key in keys:
            bucket[key] += usage[key]
            total[key] += usage[key]
    return {
        "calls": [call.model_dump() for call in calls],
        "by_step": by_step,
        "total": total,
    }


def _is_retryable_http(exc: BaseException) -> bool:
    """HTTP 5xx 与超时退避重试；4xx 属于配置问题，不重试。"""
    if isinstance(exc, httpx.TimeoutException):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


def build_llm_client(settings: Settings) -> LLMClient:
    """按配置解析模式并构造适配器。"""
    mode = settings.resolve_llm_mode()
    if mode == "fake":
        return FakeLLMClient()
    if not settings.llm.model:
        raise LLMError("使用真实 LLM 时必须配置 RF_LLM_MODEL")
    return OpenAICompatClient(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key or "",
        model=settings.llm.model,
        timeout_seconds=settings.llm.timeout_seconds,
        temperature=settings.llm.temperature,
        structured_output=settings.llm.structured_output,
        supports_tools=settings.llm.supports_tools,
        extra_body=settings.llm.extra_body,
    )


def probe_structured_output(
    client: OpenAICompatClient,
    response_model: type[T],
) -> dict[str, Any]:
    """探测 Provider 是否真的执行 json_schema 约束（只给冒烟脚本用，不进主链路）。

    用一个故意不提示任何字段的 prompt 直接发一次 json_schema 请求：真正执行约束
    的 Provider 会按 strict schema 补全全部字段；忽略约束的 Provider 会返回缺字段
    的 JSON。这里不走 `complete()`，避免它追加的字段说明把字段泄漏进 prompt。
    """
    schema = response_model.model_json_schema()
    fields = list(schema.get("properties") or {})
    body: dict[str, Any] = {
        "model": client.model,
        "temperature": client.temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": strict_json_schema(response_model),
                "strict": True,
            },
        },
        "messages": [
            {"role": "system", "content": "你是严谨的科研助手，只输出结构化 JSON。"},
            {"role": "user", "content": "请直接输出结果。"},
        ],
    }
    try:
        payload = client._post(body)
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        return {"enforced": False, "detail": f"请求失败: {exc}"}

    content = _content_of(payload)
    if content is None:
        return {"enforced": False, "detail": "模型未返回正文（content 为 null）"}
    try:
        data = json.loads(content)
    except ValueError as exc:
        return {"enforced": False, "detail": f"正文不是 JSON: {exc}"}
    if not isinstance(data, dict):
        return {"enforced": False, "detail": f"正文不是 JSON 对象: {type(data).__name__}"}

    missing = [name for name in fields if name not in data]
    if missing:
        return {"enforced": False, "detail": "缺字段：" + "、".join(missing)}
    return {"enforced": True, "detail": "全部字段都在"}

"""统一异常与结构化错误码。

Gate 1 中所有可预期失败都通过 ResearchFlowError 子类表达，
CLI 只负责把 code 与 message 输出到标准错误流。
"""

from __future__ import annotations


class ResearchFlowError(Exception):
    """所有业务异常的基类。

    除了错误码，还声明两个回溯相关属性（P1-8）：
    - `retryable`：原地再跑一次是否可能成功（如 LLM 输出偶发不合格）。
    - `default_backtrack_to`：要换个起点重跑时，默认回到哪个节点。
    子类必须显式声明，`@step` 装饰器据此填进 `ErrorRecord`，Gate 2 的规则基线
    才会读到可路由的错误。
    """

    code = "RF-000"
    retryable: bool = False
    default_backtrack_to: str | None = None

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigError(ResearchFlowError):
    code = "RF-001"
    retryable = False
    default_backtrack_to = None


class LLMError(ResearchFlowError):
    code = "RF-002"
    retryable = True
    default_backtrack_to = None


class DataValidationError(ResearchFlowError):
    code = "RF-003"
    retryable = False
    default_backtrack_to = "data_prepare"


class ComputeError(ResearchFlowError):
    code = "RF-004"
    retryable = False
    default_backtrack_to = "compute"


class ArtifactError(ResearchFlowError):
    code = "RF-005"
    retryable = False
    default_backtrack_to = None


class ReviewRejectedError(ResearchFlowError):
    code = "RF-006"
    retryable = False
    default_backtrack_to = None


class RetrievalError(ResearchFlowError):
    """外部检索源失败（超时、HTTP 错误、响应结构异常）。

    与"检索为空"严格区分：空列表表示真的没命中，网络故障必须报错，否则会被当成
    "没找到文献"静默终止。超时与 5xx 原地重试可能成功，所以 retryable=True。
    """

    code = "RF-007"
    retryable = True
    default_backtrack_to = "retrieval"

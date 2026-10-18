"""API 层的可预期错误：统一错误体 `{"error": {"code", "message"}}`。

沿用 T-013 的风格：message 要说清"怎么改"，让调用方知道下一步做什么，而不是只说
"409 Conflict"。
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

# 状态码 → 稳定错误码：客户端可以按 code 分支，不用去解析 message。
NOT_FOUND = "not_found"
CONFLICT = "conflict"
REAL_NOT_ALLOWED = "real_not_allowed"


class ApiError(Exception):
    """带 HTTP 状态码与错误码的业务异常，由 `api_error_handler` 转成统一错误体。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    """统一错误体。"""
    return {"error": {"code": code, "message": message}}


async def api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """把 `ApiError` 渲染成 JSON 错误体（注册到 app.add_exception_handler）。"""
    if not isinstance(exc, ApiError):  # pragma: no cover - 注册时已限定类型
        raise exc
    return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message))

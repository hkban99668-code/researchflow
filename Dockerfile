# 一个镜像、两种用法（T-035）：
#   researchflow serve --host 0.0.0.0 --port 8000   # 默认（API）
#   researchflow run --topic examples/demo_topic.yaml --mode fake --auto-approve
#
# 构建分两段：builder 用 uv 按 uv.lock 装依赖与项目本身，runtime 只拷 .venv 与
# 运行所需的代码/配置/示例，不含 uv、编译器与 dev 依赖。默认 Fake 模式，
# 不带任何 Key 也能起服务；.env 不进镜像（见 .dockerignore）。

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 先装依赖再装项目：依赖层只随 pyproject/uv.lock 变化。
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# --frozen：严格按 uv.lock 解析，绝不在构建时改锁文件。
RUN uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 图表固定 Agg 后端（⑤）；字体缓存写到可写目录，非 root 也能出图。
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# 非 root 运行（A1）；固定 uid/gid，挂载命名卷时权限可预期。
RUN groupadd --system --gid "${APP_GID}" app \
    && useradd --system --uid "${APP_UID}" --gid app --create-home --home-dir /home/app app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app examples ./examples

# runs/ 是命名卷的挂载点：镜像内先建好并归 app，卷首次创建时继承该权限。
RUN mkdir -p /app/runs && chown app:app /app/runs

USER app

EXPOSE 8000

# 不依赖 curl（slim 镜像里没有）：直接用标准库打 /healthz。
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["researchflow", "serve", "--host", "0.0.0.0", "--port", "8000"]

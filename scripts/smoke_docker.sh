#!/usr/bin/env bash
# 容器冒烟（T-035 A3）：构建 → 起 api → 等健康 → POST /runs(fake) →
# 等 awaiting_review → POST resume approve → 等到 ok → GET trace → 停服务。
#
# 只用 Fake 模式：不读 .env、不传 --confirm-real-calls、不产生真实费用。
# 任何一步失败都以非零退出。CI 的 docker job 直接跑本脚本。
#
# 可用环境变量：
#   SMOKE_BASE_URL        默认 http://127.0.0.1:8000
#   SMOKE_TIMEOUT_SECONDS 单步轮询超时，默认 120
#   SMOKE_KEEP_RUNNING=1  失败/结束后保留容器，便于排查（默认清理）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_URL="${SMOKE_BASE_URL:-http://127.0.0.1:8000}"
TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-120}"
# --env-file /dev/null：连 Compose 用于插值的 .env 都不读。
COMPOSE=(docker compose --env-file /dev/null -f docker-compose.yml)

log() { printf '[smoke_docker] %s\n' "$*"; }
fail() { printf '[smoke_docker] FAIL: %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || fail "找不到 docker"
command -v curl >/dev/null 2>&1 || fail "找不到 curl"

cleanup() {
    if [ "${SMOKE_KEEP_RUNNING:-0}" != "1" ]; then
        "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

log "构建镜像（uv sync --frozen --no-dev，Fake、零密钥）"
"${COMPOSE[@]}" build api || fail "docker compose build api 失败"

log "启动 api 容器"
"${COMPOSE[@]}" up -d api || fail "docker compose up -d api 失败"

log "等待 /healthz 就绪"
deadline=$((SECONDS + TIMEOUT_SECONDS))
until curl -fsS "${BASE_URL}/healthz" 2>/dev/null | grep -q '"status":"ok"'; do
    [ "$SECONDS" -lt "$deadline" ] || {
        "${COMPOSE[@]}" logs --tail=50 api >&2 || true
        fail "健康检查超时（${TIMEOUT_SECONDS}s）"
    }
    sleep 1
done
log "健康检查通过：$(curl -fsS "${BASE_URL}/healthz")"

payload='{"mode":"fake","topic":{"topic":"容器冒烟：科研数据质量与线性回归可复现性评估","dataset":{"path":"examples/sample_dataset.csv","target":"y","features":["x1","x2","x3"]},"param_set":{"algorithm":"linear_regression","seed":42,"params":{"test_size":0.25}}}}'

created="$(curl -fsS -X POST "${BASE_URL}/runs" \
    -H 'Content-Type: application/json' --data "${payload}")" || fail "POST /runs 失败"
run_id="$(printf '%s' "$created" | sed -n 's/^{"run_id":"\([0-9a-f]\{12\}\)".*/\1/p')"
[ -n "$run_id" ] || fail "POST /runs 响应里没有 run_id：${created}"
log "run_id=${run_id}"

status_of() {
    # 只看顶层的 run_id/status（响应按插入顺序序列化，status 紧跟 run_id）。
    curl -fsS "${BASE_URL}/runs/${run_id}" \
        | sed -n 's/^{"run_id":"[0-9a-f]\{12\}","status":"\([a-z_]*\)".*/\1/p'
}

wait_status() {
    local want="$1" last="" step_deadline=$((SECONDS + TIMEOUT_SECONDS))
    while :; do
        last="$(status_of || true)"
        if [ "$last" = "$want" ]; then
            return 0
        fi
        case "$last" in
            ok | terminated | error | interrupted)
                fail "run ${run_id} 停在 ${last}，等不到 ${want}"
                ;;
        esac
        [ "$SECONDS" -lt "$step_deadline" ] || {
            "${COMPOSE[@]}" logs --tail=50 api >&2 || true
            fail "等 ${want} 超时（最后状态 ${last:-unknown}）"
        }
        sleep 1
    done
}

log "等待停在审核处（awaiting_review）"
wait_status awaiting_review
review="$(curl -fsS "${BASE_URL}/runs/${run_id}")"
printf '%s' "$review" | grep -q '"stage":"data_review"' || fail "审核载荷缺 stage=data_review"

log "POST resume approve"
curl -fsS -X POST "${BASE_URL}/runs/${run_id}/resume" \
    -H 'Content-Type: application/json' \
    --data '{"decision":"approve","reason":"smoke_docker"}' >/dev/null \
    || fail "POST /runs/${run_id}/resume 失败"

log "等待运行结束（ok）"
wait_status ok

log "取 /runs/${run_id}/trace"
trace="$(curl -fsS "${BASE_URL}/runs/${run_id}/trace")" || fail "GET /runs/${run_id}/trace 失败"
printf '%s' "$trace" | grep -q '"trace_id"' || fail "trace.json 里没有 trace_id"

log "run 目录内容："
"${COMPOSE[@]}" exec -T api ls "runs/${run_id}" || fail "列出 run 目录失败"

log "PASS：fake 全流程（run_id=${run_id}）"

#!/usr/bin/env bash
# 모델 서버 일괄 관리 (RTX 4090 24GB 1장 기준)
#
#   bash scripts/serve.sh start [27b|9b]   # 기본 27b
#   bash scripts/serve.sh stop
#   bash scripts/serve.sh status
#   bash scripts/serve.sh restart-tools    # 도메인 API·도구 서버만 다시 (카탈로그·도메인 코드를 바꿨을 때)
#
# 구성 (2026-10-12 실측으로 정한 배치)
#   :8011  임베딩 Qwen3-Embedding-0.6B — CPU (쿼리 1건씩이라 충분, GPU를 LLM에 양보)
#   :8012  리랭커 Qwen3-Reranker-0.6B — GPU 약 2.0GB (KV 256MB 고정 — max-model-len 2048의 최소선)
#          (2026-10-14 --enforce-eager: CUDA graph 메모리 제거 2.3→2.0GB, 지연 144→165ms, 순위 품질 동일)
#   :8002  LLM Qwen3.8-27B-INT4 — 가중치 16.8GB + KV 0.5GB(약 9.1k 토큰 — max-model-len 8192의 최소선,
#          평가 로그 최대 입력 3.7k), 이미지 입력 끔
#          (2026-10-14 KV 1.5→0.75→0.5GB: STT·TTS 올릴 자리 확보, 27B 점유 18.9→18.5GB)
#   :8000  LLM Qwen3.5-9B-FP8 (대안)
#   :8013  음성 scripts/voice_server.py (env voice-eval) — STT Whisper turbo int8 GPU 약 1.55GB, TTS Supertonic 3 CPU
#          (2026-10-14 실측: STT·TTS를 둘 다 GPU에 두면 넘침 → TTS는 CPU). LLM 다음에 띄운다
#   :8020  도메인 API · :8021 도구 서버(MCP) — scripts/run_tools.py (env customer-service-agent, GPU 안 씀, 127.0.0.1)
#          평가·웹 서버보다 먼저 떠 있어야 한다 (docs/08 3절)
# GPU를 꽉 채우면 Windows가 GPU 메모리를 시스템 메모리로 넘겨 생성이 초당 3~4토큰까지 느려진다 → 여유를 남긴다.
# start는 .env의 LLM_URL·LLM_MODEL·EMBEDDER_URL·RERANKER_URL을 현재 WSL IP로 갱신한다.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 기계마다 다른 경로는 .serve.env(커밋 안 함)에서 덮어쓴다: VENV · APP_PY · MODEL_27B · MODEL_9B · CPU_EMBED_SERVER
[ -f "$ROOT/.serve.env" ] && . "$ROOT/.serve.env"
VENV=${VENV:-$HOME/miniconda3/envs/vllm}
APP_PY=${APP_PY:-$HOME/miniconda3/envs/customer-service-agent/bin/python}
LOGDIR=${LOGDIR:-$HOME/csagent_logs}; mkdir -p "$LOGDIR"
IP=$(hostname -I | awk '{print $1}')
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1

MODEL_27B=${MODEL_27B:-$HOME/models/Qwen3.8-27B-INT4}
MODEL_9B=${MODEL_9B:-$HOME/models/Qwen3.5-9B-FP8-dynamic}
CPU_EMBED_SERVER=${CPU_EMBED_SERVER:-}

up()   { curl -s --noproxy '*' --max-time 3 "$1" 2>/dev/null | grep -q "$2"; }
wait_up() { # url pattern log name
  local n=0
  until up "$1" "$2"; do
    if grep -qE "initialization failed|CUDA out of memory|Address already in use|^RuntimeError|^ImportError|^ModuleNotFoundError" "$3" 2>/dev/null; then
      echo "!! $4 실패 — $3 확인"; return 1; fi
    sleep 5; n=$((n+5)); [ $n -ge 600 ] && { echo "!! $4 타임아웃 — $3 확인"; return 1; }
  done
  echo "ok: $4"
}
set_env() { # key value
  touch "$ROOT/.env"
  if grep -q "^$1=" "$ROOT/.env"; then sed -i "s#^$1=.*#$1=$2#" "$ROOT/.env"; else echo "$1=$2" >> "$ROOT/.env"; fi
}

start_tools() {
  if up "http://127.0.0.1:8020/health" '"domain"' && up "http://127.0.0.1:8021/health" '"catalog"'; then
    echo "ok: 도메인 API·도구 서버 이미 실행 중"; return 0; fi
  (cd "$ROOT" && setsid nohup "$APP_PY" scripts/run_tools.py > "$LOGDIR/tools.log" 2>&1 &)
  wait_up "http://127.0.0.1:8020/health" '"domain"' "$LOGDIR/tools.log" "도메인 API" || return 1
  wait_up "http://127.0.0.1:8021/health" '"catalog"' "$LOGDIR/tools.log" "도구 서버(MCP)"
}

stop_tools() {
  for pat in "[s]cripts/run_tools.py" "[c]sagent.domain.app:create_dev_app" "[c]sagent.toolserver.server:create_default_app"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  local n=0
  while { up "http://127.0.0.1:8020/health" '"domain"' || up "http://127.0.0.1:8021/health" '"catalog"'; } && [ $n -lt 10 ]; do
    sleep 1; n=$((n+1)); done
}

start() {
  local which=${1:-27b}
  echo "== GPU 사용 중: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

  if ! up "http://$IP:8011/" qwen3-emb-cpu; then
    [ -n "$CPU_EMBED_SERVER" ] || { echo "CPU 임베딩 서버 스크립트 경로를 .serve.env의 CPU_EMBED_SERVER에 지정하세요"; return 1; }
    CUDA_VISIBLE_DEVICES="" setsid nohup $VENV/bin/python "$CPU_EMBED_SERVER" \
      --port 8011 > "$LOGDIR/cpu_embed.log" 2>&1 &
    wait_up "http://$IP:8011/" qwen3-emb-cpu "$LOGDIR/cpu_embed.log" "임베딩(CPU)" || return 1
  else echo "ok: 임베딩(CPU) 이미 실행 중"; fi

  if ! up "http://$IP:8012/v1/models" qwen3-reranker; then
    setsid nohup $VENV/bin/vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling \
      --served-model-name qwen3-reranker --max-model-len 2048 \
      --gpu-memory-utilization 0.08 --kv-cache-memory-bytes 268435456 --enforce-eager --host 0.0.0.0 --port 8012 \
      --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
      > "$LOGDIR/reranker.log" 2>&1 &
    wait_up "http://$IP:8012/v1/models" qwen3-reranker "$LOGDIR/reranker.log" "리랭커(GPU)" || return 1
  else echo "ok: 리랭커 이미 실행 중"; fi

  if [ "$which" = "9b" ]; then
    local name=qwen35-9b port=8000
    if ! up "http://$IP:$port/v1/models" $name; then
      setsid nohup $VENV/bin/vllm serve "$MODEL_9B" --served-model-name $name --max-model-len 8192 \
        --enforce-eager --gpu-memory-utilization 0.60 --kv-cache-memory-bytes 1610612736 \
        --host 0.0.0.0 --port $port > "$LOGDIR/llm9b.log" 2>&1 &
      wait_up "http://$IP:$port/v1/models" $name "$LOGDIR/llm9b.log" "LLM 9B" || return 1
    else echo "ok: LLM 9B 이미 실행 중"; fi
  else
    local name=qwen3.8-27b-redhat-int4 port=8002
    if ! up "http://$IP:$port/v1/models" $name; then
      setsid nohup $VENV/bin/vllm serve "$MODEL_27B" --served-model-name $name \
        --quantization compressed-tensors --attention-backend TRITON_ATTN \
        --gpu-memory-utilization 0.84 --kv-cache-memory-bytes 536870912 \
        --max-model-len 8192 --max-num-seqs 2 --enforce-eager --trust-remote-code \
        --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
        --host 0.0.0.0 --port $port > "$LOGDIR/llm27b.log" 2>&1 &
      wait_up "http://$IP:$port/v1/models" $name "$LOGDIR/llm27b.log" "LLM 27B" || return 1
    else echo "ok: LLM 27B 이미 실행 중"; fi
  fi

  if ! up "http://$IP:8013/health" supertonic; then
    setsid nohup $HOME/miniconda3/envs/voice-eval/bin/python "$ROOT/scripts/voice_server.py" --port 8013 \
      > "$LOGDIR/voice.log" 2>&1 &
    wait_up "http://$IP:8013/health" supertonic "$LOGDIR/voice.log" "음성(STT·TTS)" || return 1
  else echo "ok: 음성 이미 실행 중"; fi

  start_tools || return 1

  set_env LLM_URL "http://$IP:$port/v1"
  set_env LLM_MODEL "$name"
  set_env EMBEDDER_URL "http://$IP:8011/v1"
  set_env RERANKER_URL "http://$IP:8012"
  set_env VOICE_URL "http://$IP:8013"
  echo "== .env 갱신 (WSL IP $IP, LLM $name) / GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

stop() {
  stop_tools
  # 패턴에 대괄호를 넣어 이 스크립트 자신의 명령줄이 걸리지 않게 한다
  for pat in "[Q]wen3.8-27B-INT4" "[Q]wen3.5-9B-FP8" "[Q]wen3-Reranker-0.6B" "[c]pu_embed_server.py" "[v]oice_server.py"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  sleep 10
  for p in $(pgrep -f "[V]LLM::EngineCore") $(pgrep -f "[c]pu_embed_server.py"); do kill -9 "$p" 2>/dev/null; done
  echo "== 정지. GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

status() {
  for s in "8011/|qwen3-emb-cpu|임베딩(CPU)" "8012/v1/models|qwen3-reranker|리랭커" \
           "8002/v1/models|qwen3.8-27b|LLM 27B" "8000/v1/models|qwen35-9b|LLM 9B" \
           "8013/health|supertonic|음성(STT·TTS)"; do
    IFS='|' read -r path pat name <<< "$s"
    if up "http://$IP:$path" "$pat"; then echo "● $name"; else echo "○ $name"; fi
  done
  for s in '8020/health|"domain"|도메인 API' '8021/health|"catalog"|도구 서버(MCP)'; do
    IFS='|' read -r path pat name <<< "$s"
    if up "http://127.0.0.1:$path" "$pat"; then echo "● $name"; else echo "○ $name"; fi
  done
  echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
}

case "${1:-status}" in
  start) start "${2:-27b}" ;;
  stop) stop ;;
  status) status ;;
  restart-tools) stop_tools; start_tools ;;
  *) echo "사용법: $0 start [27b|9b] | stop | status | restart-tools"; exit 1 ;;
esac

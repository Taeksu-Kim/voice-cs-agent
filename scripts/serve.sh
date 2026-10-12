#!/usr/bin/env bash
# 모델 서버 일괄 관리 (RTX 4090 24GB 1장 기준)
#
#   bash scripts/serve.sh start [27b|9b]   # 기본 27b
#   bash scripts/serve.sh stop
#   bash scripts/serve.sh status
#
# 구성 (2026-10-12 실측으로 정한 배치)
#   :8011  임베딩 Qwen3-Embedding-0.6B — CPU (쿼리 1건씩이라 충분, GPU를 LLM에 양보)
#   :8012  리랭커 Qwen3-Reranker-0.6B — GPU 약 2GB (KV 256MB 고정)
#   :8002  LLM Qwen3.8-27B-INT4 — GPU 약 18GB, 이미지 입력 끔 (켜면 메모리 프로파일링에서 멈춤)
#   :8000  LLM Qwen3.5-9B-FP8 (대안)
# GPU를 꽉 채우면 Windows가 GPU 메모리를 시스템 메모리로 넘겨 생성이 초당 3~4토큰까지 느려진다 → 여유를 남긴다.
# start는 .env의 LLM_URL·LLM_MODEL·EMBEDDER_URL·RERANKER_URL을 현재 WSL IP로 갱신한다.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=$HOME/miniconda3/envs/vllm
LOGDIR=${LOGDIR:-$HOME/csagent_logs}; mkdir -p "$LOGDIR"
IP=$(hostname -I | awk '{print $1}')
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1

MODEL_27B=/home/ubuntu/models/Qwen3.8-27B-INT4
MODEL_9B="/mnt/d/workspace/stock dataset/models/Qwen3.5-9B-FP8-dynamic"

up()   { curl -s --noproxy '*' --max-time 3 "$1" 2>/dev/null | grep -q "$2"; }
wait_up() { # url pattern log name
  local n=0
  until up "$1" "$2"; do
    if grep -qE "initialization failed|CUDA out of memory|Address already in use" "$3" 2>/dev/null; then
      echo "!! $4 실패 — $3 확인"; return 1; fi
    sleep 5; n=$((n+5)); [ $n -ge 600 ] && { echo "!! $4 타임아웃 — $3 확인"; return 1; }
  done
  echo "ok: $4"
}
set_env() { # key value
  touch "$ROOT/.env"
  if grep -q "^$1=" "$ROOT/.env"; then sed -i "s#^$1=.*#$1=$2#" "$ROOT/.env"; else echo "$1=$2" >> "$ROOT/.env"; fi
}

start() {
  local which=${1:-27b}
  echo "== GPU 사용 중: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

  if ! up "http://$IP:8011/" qwen3-emb-cpu; then
    CUDA_VISIBLE_DEVICES="" setsid nohup $VENV/bin/python /mnt/d/workspace/agentic_rag/scripts/cpu_embed_server.py \
      --port 8011 > "$LOGDIR/cpu_embed.log" 2>&1 &
    wait_up "http://$IP:8011/" qwen3-emb-cpu "$LOGDIR/cpu_embed.log" "임베딩(CPU)" || return 1
  else echo "ok: 임베딩(CPU) 이미 실행 중"; fi

  if ! up "http://$IP:8012/v1/models" qwen3-reranker; then
    setsid nohup $VENV/bin/vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling \
      --served-model-name qwen3-reranker --max-model-len 2048 \
      --gpu-memory-utilization 0.08 --kv-cache-memory-bytes 268435456 --host 0.0.0.0 --port 8012 \
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
        --gpu-memory-utilization 0.84 --kv-cache-memory-bytes 1610612736 \
        --max-model-len 8192 --max-num-seqs 2 --enforce-eager --trust-remote-code \
        --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
        --host 0.0.0.0 --port $port > "$LOGDIR/llm27b.log" 2>&1 &
      wait_up "http://$IP:$port/v1/models" $name "$LOGDIR/llm27b.log" "LLM 27B" || return 1
    else echo "ok: LLM 27B 이미 실행 중"; fi
  fi

  set_env LLM_URL "http://$IP:$port/v1"
  set_env LLM_MODEL "$name"
  set_env EMBEDDER_URL "http://$IP:8011/v1"
  set_env RERANKER_URL "http://$IP:8012"
  echo "== .env 갱신 (WSL IP $IP, LLM $name) / GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

stop() {
  # 패턴에 대괄호를 넣어 이 스크립트 자신의 명령줄이 걸리지 않게 한다
  for pat in "[Q]wen3.8-27B-INT4" "[Q]wen3.5-9B-FP8" "[Q]wen3-Reranker-0.6B" "[c]pu_embed_server.py"; do
    for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null; done
  done
  sleep 10
  for p in $(pgrep -f "[V]LLM::EngineCore") $(pgrep -f "[c]pu_embed_server.py"); do kill -9 "$p" 2>/dev/null; done
  echo "== 정지. GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

status() {
  for s in "8011/|qwen3-emb-cpu|임베딩(CPU)" "8012/v1/models|qwen3-reranker|리랭커" \
           "8002/v1/models|qwen3.8-27b|LLM 27B" "8000/v1/models|qwen35-9b|LLM 9B"; do
    IFS='|' read -r path pat name <<< "$s"
    if up "http://$IP:$path" "$pat"; then echo "● $name"; else echo "○ $name"; fi
  done
  echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
}

case "${1:-status}" in
  start) start "${2:-27b}" ;;
  stop) stop ;;
  status) status ;;
  *) echo "사용법: $0 start [27b|9b] | stop | status"; exit 1 ;;
esac

#!/bin/bash
set -e

# 가상환경 활성화
source ~/.venv_n/bin/activate

# 로그 디렉토리
mkdir -p logs

echo "vLLM 서버 시작"
echo "$(date '+%Y-%m-%d %H:%M:%S') - vLLM 서버 시작" >> logs/vllm_server.log
nohup python -m vllm.entrypoints.openai.api_server \
  --model TommyKong/gemma-3-finetune-4bit \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --download-dir ./model_cache \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.5 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --quantization bitsandbytes \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  >> logs/vllm_server.log 2>&1 &


sleep 30

echo "FastAPI 서버 시작"
echo "$(date '+%Y-%m-%d %H:%M:%S') - FastAPI 시작" >> logs/fastapi_server.log
nohup python -m uvicorn app.main:app --host=0.0.0.0 --port=8000 \
  >> logs/fastapi_server.log 2>&1 &

sleep 3m

echo "퀴즈 생성 워커 시작"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Quiz Worker 시작" >> logs/quiz_worker.log
nohup python -m app.workers.quiz_worker \
  >> logs/quiz_worker.log 2>&1 &

echo "모든 서비스가 백그라운드에서 실행되었습니다."

# 실행된 프로세스 확인 (선택)
ps aux | grep -E 'vllm|uvicorn|quiz_worker' | grep -v grep
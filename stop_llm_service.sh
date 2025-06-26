#!/bin/bash
set -e

echo "서비스 종료 중..."

# vLLM 종료
echo "→ vLLM 서버 종료"
pkill -f "vllm.entrypoints.openai.api_server" || echo "vLLM 서버가 실행 중이지 않음"

# FastAPI 종료
echo "→ FastAPI 서버 종료"
pkill -f "uvicorn app.main:app" || echo "FastAPI 서버가 실행 중이지 않음"

# Quiz Worker 종료
echo "→ 퀴즈 생성 워커 종료"
pkill -f "app.workers.quiz_worker" || echo "Quiz 워커가 실행 중이지 않음"

echo "모든 서비스 종료 완료"
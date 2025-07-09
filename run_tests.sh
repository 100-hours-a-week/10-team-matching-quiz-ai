#!/bin/bash

# 꼬리질문 생성 기능 테스트 및 커버리지 측정 스크립트

echo "=== 꼬리질문 생성 기능 테스트 시작 ==="

# 테스트 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export ENVIRONMENT="test"
export ENABLED_MODELS="question_generator"

# 테스트 의존성 설치 확인
echo "테스트 의존성 확인 중..."
pip install -r requirements-test.txt

# 테스트 실행 및 커버리지 측정
echo "테스트 실행 중..."
pytest tests/ \
    --cov=app \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --cov-config=.coveragerc \
    --verbose \
    -x \
    --tb=short

# 커버리지 결과 요약
echo ""
echo "=== 커버리지 측정 완료 ==="
echo "상세 HTML 리포트: htmlcov/index.html"
echo "XML 리포트: coverage.xml"

# 커버리지 HTML 리포트 열기 (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "HTML 커버리지 리포트를 열고 있습니다..."
    open htmlcov/index.html
fi

echo ""
echo "=== 주요 테스트 영역 ==="
echo "✅ 꼬리질문 생성 API 엔드포인트"
echo "✅ RAG 검색 기능"
echo "✅ vLLM/OpenAI 모델 통합"
echo "✅ 컨텍스트 준비 및 프롬프트 처리"
echo "✅ 질문 파싱 및 검증"
echo "✅ 에러 처리 및 예외 상황"
echo "✅ Langfuse 추적 및 모니터링"

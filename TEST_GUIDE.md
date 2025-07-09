# 꼬리질문 생성 기능 테스트 가이드

## 📋 현재 상황

✅ **테스트 환경 구축 완료**
- 기본 테스트 프레임워크 설정 (pytest, coverage)
- Mock 클라이언트 구현 (vLLM, OpenAI, RAG, Langfuse)
- 커버리지 측정 및 분석 도구

✅ **현재 커버리지: 36.7%** ⬆️ (17.2% → 36.7% 향상!)
- question_generator_schema.py: 100% ✅ (완료)
- question_generator_config.py: 93.75% ✅
- question_generator_parser.py: 60.98% 🟡
- question_generator_model.py: 63.04% 🟡 (Mock 테스트 완료)
- question_generator_api.py: 0% ❌ (미구현)

## 🚀 테스트 실행 방법

### 1. 기본 테스트 실행
```bash
# 전체 테스트 실행
pytest tests/ -v

# 꼬리질문 생성 모듈만 테스트
pytest tests/test_basic_functionality.py -v

# 커버리지와 함께 실행
pytest tests/ --cov=app/api/question_generator --cov-report=html --cov-report=term-missing
```

### 2. 테스트 스크립트 사용
```bash
# 자동화된 테스트 실행
./run_tests.sh

# 커버리지 분석
python analyze_coverage.py
```

### 3. HTML 리포트 확인
```bash
# 브라우저에서 상세 커버리지 리포트 열기
open htmlcov/index.html
```

## 📊 모듈별 커버리지 현황

### 🟢 높은 커버리지 (80%+)
- `question_generator_schema.py` (100%) - 데이터 스키마 완료 ✅
- `question_generator_config.py` (93.75%) - 설정 관리 거의 완료 ✅

### 🟡 중간 커버리지 (40-80%)
- `question_generator_model.py` (63.04%) - **vLLM Mock 테스트 완료** 🚀
- `question_generator_parser.py` (60.98%) - 질문 파싱 로직 부분 완료

### 🔴 낮은 커버리지 (0-40%)
- `question_generator_api.py` (0%) - **핵심 API 로직 미테스트** (다음 우선순위)

## 🎯 커버리지 개선 우선순위

### 1. 최우선 (API 핵심 로직) - **다음 단계**
```bash
# question_generator_api.py 테스트 추가 필요 (현재 0%)
- perform_rag_search() 함수
- generate_questions_with_fallback() 함수  
- prepare_context() 함수
- /followup-questions 엔드포인트
```

### 2. ✅ 완료 (모델 통합) - **Mock 테스트 완료 (63.04%)**
```bash
# question_generator_model.py 테스트 완료 ✅
✅ call_llm() 함수 Mock 테스트
✅ call_openai_api() 폴백 테스트
✅ initialize_llm() 초기화 테스트
✅ check_vllm_health() 헬스체크
✅ API 파라미터 검증
✅ 에러 처리 시나리오
```

### 3. 완료 필요 (파서 로직)
```bash
# question_generator_parser.py 커버리지 개선
- parse_questions() 함수의 모든 케이스
- 다양한 입력 형식 처리
- 에러 처리 로직
```

## 🛠 GPU/GCP 환경 요구사항

### Mac 환경의 한계
❌ **Mac에서는 완전한 테스트 불가능**
- CUDA GPU 미지원 (코드에서 `device = "cuda"` 사용)
- vLLM 서버 실행 불가
- 대용량 LLM 모델 로딩 불가

### GCP 환경에서 실행 필요
✅ **GCP GPU 인스턴스에서 실행해야 하는 이유:**
1. **CUDA GPU 필요**: `torch.cuda.is_available()` 체크
2. **vLLM API 서버**: `http://127.0.0.1:8080/v1` 엔드포인트 필요
3. **대용량 모델**: `TommyKong/gemma-3-finetune-4bit` 모델 로딩
4. **GPU 메모리**: 4-bit quantized 모델도 상당한 GPU 메모리 필요

### GCP 배포 명령어
```bash
# GCP 인스턴스에서
export ENVIRONMENT=gcp-gke
export ENABLED_MODELS=question_generator
export VLLM_GPU_MEMORY_UTILIZATION=0.8

# GPU 확인
nvidia-smi

# vLLM 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model TommyKong/gemma-3-finetune-4bit \
    --port 8080 \
    --gpu-memory-utilization 0.8

# 별도 터미널에서 테스트 실행
pytest tests/ --cov=app --cov-report=html
```

## 🔧 추가 테스트 구현 필요

### ✅ Mock 기반 vLLM API 테스트 (완료!)
```python
# tests/test_vllm_api.py ✅ 구현 완료
✅ vLLM Mock API 호출 테스트
✅ OpenAI 폴백 테스트  
✅ 헬스체크 테스트
✅ 에러 처리 테스트
✅ API 파라미터 검증
커버리지: question_generator_model.py 63.04%
```

### 🔄 다음 우선순위: API 엔드포인트 테스트
```python
# tests/test_question_generator_api_mock.py 생성 필요
- FastAPI TestClient 사용
- perform_rag_search() Mock 테스트
- generate_questions_with_fallback() 테스트
- /followup-questions 엔드포인트 플로우 테스트
목표: question_generator_api.py 50-70% 커버리지
```

### 통합 테스트
```python
# tests/test_question_generator_integration.py 수정 필요
- 실제 모델 호출 시뮬레이션
- 전체 파이프라인 테스트
- 에러 시나리오 테스트
```

## 📈 목표 커버리지

### 단계별 목표
1. **✅ 1차 목표 달성**: 36.7% (vLLM Mock 테스트 완료) 
2. **🎯 2차 목표**: 60% (API 엔드포인트 테스트 추가)
3. **🎯 최종 목표**: 85% (GCP 실제 환경 테스트)

### 현재 달성 커버리지 (36.7%)
- `question_generator_schema.py`: 100% ✅
- `question_generator_config.py`: 93.75% ✅
- `question_generator_model.py`: 63.04% ✅ (Mock 완료)
- `question_generator_parser.py`: 60.98% 🟡
- `question_generator_api.py`: 0% ❌ (다음 목표)

## 💡 다음 단계

1. **✅ vLLM Mock 테스트 완성** (완료 - 63.04% 커버리지)
2. **🎯 API 엔드포인트 테스트 추가** (다음 우선순위 - 목표 60% 전체 커버리지)
3. **🌩 GCP GPU 인스턴스 설정** (실제 모델 테스트 - 목표 85%)
4. **🔄 CI/CD 파이프라인 구축** (자동화된 테스트)

---

## 🎉 주요 성과

### ✅ vLLM API 테스트 평가 완료!

**Mac 환경에서 성공적으로 달성:**
- **Mock 기반 vLLM API 테스트**: 5개 테스트 모두 통과 ✅
- **커버리지 2.1배 향상**: 17.2% → 36.7% (19.5%p 증가)
- **question_generator_model.py**: 0% → 63.04% (완전한 Mock 테스트)

### 🎯 vLLM API 평가 방법론 확립

1. **Mac 환경 (현재)**: Mock 기반으로 63% 커버리지 달성
   - ✅ API 호출 로직 검증
   - ✅ 에러 처리 및 폴백 테스트
   - ✅ 파라미터 검증 및 설정 테스트

2. **GCP GPU 환경**: 실제 vLLM 서버로 완전한 테스트
   - 🌩 실제 모델 추론 테스트
   - 🌩 성능 벤치마크 
   - 🌩 GPU 리소스 관리 검증

**결론**: 현재 36.7% 커버리지에서 시작하여, API 엔드포인트 테스트 추가로 60%까지, GCP GPU 환경에서 실제 모델 테스트를 통해 85% 목표 달성이 가능합니다.

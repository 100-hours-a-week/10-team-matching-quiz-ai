# vLLM API 테스트 전략 가이드

## 🎯 vLLM API 테스트 접근 방법

vLLM API는 환경에 따라 다르게 테스트해야 합니다:

### 1️⃣ **Mac 환경 (현재)**: Mock 기반 테스트
### 2️⃣ **GCP GPU 환경**: 실제 API 테스트

---

## 🖥 Mac 환경에서의 테스트 (Mock 기반)

### ✅ 가능한 테스트
```python
# 1. API 호출 로직 검증
- call_llm() 함수의 파라미터 전달
- 응답 파싱 및 처리 로직
- 에러 처리 및 폴백 메커니즘
- 설정값 검증

# 2. Mock을 통한 시나리오 테스트
- 정상 응답 시나리오
- 타임아웃 시나리오
- 연결 실패 시나리오
- OpenAI 폴백 시나리오
```

### 🔧 Mock 테스트 실행
```bash
# vLLM Mock 테스트 실행
pytest tests/test_vllm_api.py::TestVLLMAPIIntegration -v

# 에러 처리 테스트
pytest tests/test_vllm_api.py::TestVLLMAPIErrorHandling -v

# 설정 검증 테스트
pytest tests/test_vllm_api.py::TestVLLMAPIConfiguration -v
```

### 📊 Mock 테스트로 달성 가능한 커버리지
- **question_generator_model.py**: 60-70%
- **question_generator_api.py**: 50-60%
- **전체 모듈**: 40-50%

---

## 🌩 GCP GPU 환경에서의 테스트 (실제 API)

### 🏗 GCP 환경 설정

#### 1. GPU 인스턴스 생성
```bash
# GCP에서 NVIDIA GPU 인스턴스 생성
gcloud compute instances create vllm-test-instance \
    --machine-type=n1-highmem-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --zone=us-central1-a
```

#### 2. vLLM 서버 시작
```bash
# SSH로 인스턴스 접속 후
pip install vllm

# vLLM 서버 시작 (백그라운드)
python -m vllm.entrypoints.openai.api_server \
    --model TommyKong/gemma-3-finetune-4bit \
    --port 8080 \
    --gpu-memory-utilization 0.8 \
    --quantization awq &

# 서버 상태 확인
curl http://localhost:8080/v1/models
```

#### 3. 환경변수 설정
```bash
export VLLM_API_BASE_URL="http://127.0.0.1:8080/v1"
export VLLM_MODEL_NAME="TommyKong/gemma-3-finetune-4bit"
export VLLM_API_KEY="EMPTY"
export ENVIRONMENT="gcp-gpu"
export ENABLED_MODELS="question_generator"
```

### ✅ 실제 환경에서 가능한 테스트
```python
# 1. 실제 모델 추론 테스트
- 실제 프롬프트로 질문 생성
- 응답 품질 검증
- 성능 벤치마크

# 2. 시스템 통합 테스트
- 전체 API 파이프라인
- RAG + vLLM 통합
- Langfuse 추적

# 3. 성능 및 부하 테스트
- 동시 요청 처리
- 메모리 사용량
- GPU 활용률
```

### 🧪 실제 환경 테스트 실행
```bash
# GPU 환경 테스트 (GPU 마크가 있는 테스트만)
pytest tests/test_vllm_api.py -m gpu -v

# 성능 벤치마크
pytest tests/test_vllm_api.py::TestVLLMAPIRealEnvironment::test_vllm_performance_benchmark -v

# 전체 통합 테스트
pytest tests/test_vllm_api.py::TestVLLMAPIRealEnvironment -v
```

---

## 📋 테스트 시나리오별 분류

### 🟢 Level 1: 기본 기능 (Mac에서 가능)
```python
✅ API 호출 파라미터 검증
✅ 응답 파싱 로직
✅ 에러 처리 및 폴백
✅ 설정값 검증
✅ Mock 시나리오 테스트
```

### 🟡 Level 2: 통합 기능 (GCP 필요)
```python
🌩 실제 모델 추론
🌩 vLLM 서버 연동
🌩 GPU 메모리 관리
🌩 실제 응답 품질
```

### 🔴 Level 3: 고급 기능 (GCP + 최적화)
```python
🌩 성능 벤치마크
🌩 동시 요청 처리
🌩 메모리 최적화
🌩 부하 테스트
```

---

## 🔍 각 환경별 커버리지 전략

### Mac 환경 목표: **50-60% 커버리지**
```python
# 주요 테스트 영역
✅ call_llm() 함수 로직 (Mock)
✅ call_openai_api() 폴백 로직  
✅ initialize_llm() 초기화
✅ check_vllm_health() 헬스체크
✅ API 파라미터 검증
✅ 에러 처리 시나리오
```

### GCP 환경 목표: **80-85% 커버리지**
```python
# 추가 테스트 영역 (Mac + GCP)
✅ 위의 모든 Mock 테스트
🌩 실제 vLLM 서버 통신
🌩 실제 모델 추론 결과
🌩 GPU 리소스 관리
🌩 성능 최적화 검증
🌩 메모리 사용량 모니터링
```

---

## 💻 현재 Mac에서 실행 가능한 테스트

### 즉시 실행 가능
```bash
# 기본 vLLM Mock 테스트
pytest tests/test_vllm_api.py::TestVLLMAPIIntegration -v

# 설정 검증 테스트
pytest tests/test_vllm_api.py::TestVLLMAPIConfiguration -v

# 에러 처리 테스트
pytest tests/test_vllm_api.py::TestVLLMAPIErrorHandling -v
```

### 예상 결과
```
✅ TestVLLMAPIIntegration: 6개 테스트 통과
✅ TestVLLMAPIConfiguration: 3개 테스트 통과  
✅ TestVLLMAPIErrorHandling: 3개 테스트 통과
📊 커버리지: question_generator_model.py 60-70%
```

---

## 🚀 GCP 환경으로 확장 시

### 사전 준비사항
1. **GPU 인스턴스**: NVIDIA T4 이상
2. **메모리**: 최소 16GB RAM
3. **스토리지**: 100GB (모델 캐시용)
4. **네트워크**: vLLM 서버 포트 8080 오픈

### 단계별 실행
```bash
# 1. vLLM 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model TommyKong/gemma-3-finetune-4bit \
    --port 8080 \
    --gpu-memory-utilization 0.8

# 2. 서버 상태 확인
curl http://localhost:8080/v1/models

# 3. 실제 환경 테스트 실행
pytest tests/test_vllm_api.py -m gpu -v

# 4. 성능 벤치마크
pytest tests/test_vllm_api.py::TestVLLMAPIRealEnvironment::test_vllm_performance_benchmark -v
```

---

## 📈 커버리지 개선 로드맵

### 현재 (17.2%) → Mock 테스트 (50-60%)
```python
✅ vLLM API Mock 테스트 추가
✅ 에러 처리 시나리오 확장
✅ 설정 검증 강화
✅ 폴백 로직 테스트
```

### Mock 테스트 (50-60%) → 실제 환경 (80-85%)
```python
🌩 GCP GPU 인스턴스 설정
🌩 vLLM 서버 배포
🌩 실제 모델 추론 테스트
🌩 성능 벤치마크 추가
```

---

## 💡 결론 및 권장사항

### 단계적 접근 전략
1. **1단계 (현재 Mac)**: Mock 기반 테스트로 50-60% 커버리지 달성
2. **2단계 (GCP 확장)**: 실제 환경 테스트로 80-85% 커버리지 달성
3. **3단계 (최적화)**: 성능 및 부하 테스트로 완전한 검증

### 즉시 실행 가능한 다음 단계
```bash
# Mac에서 지금 실행 가능
pytest tests/test_vllm_api.py -v --cov=app/api/question_generator --cov-report=html
```

이렇게 하면 vLLM API를 환경에 맞게 효과적으로 테스트할 수 있습니다!

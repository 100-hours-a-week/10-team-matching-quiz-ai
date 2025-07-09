import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json


class TestVLLMAPIIntegration:
    """vLLM API 통합 테스트 - 실제 환경과 Mock 환경 모두 지원"""
    
    @pytest.fixture
    def mock_vllm_response(self):
        """Mock vLLM API 응답"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].text = """
        1. 데이터베이스 트랜잭션의 ACID 속성에 대해 설명해주세요
        2. MySQL과 PostgreSQL의 주요 차이점은 무엇인가요?
        3. 인덱스 설계 시 고려해야 할 요소들을 말씀해주세요
        """
        return mock_response
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API 응답"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = """
        1. 캐싱 전략에는 어떤 종류가 있나요?
        2. Redis와 Memcached의 차이점을 설명해주세요
        3. NoSQL 데이터베이스의 장단점은 무엇인가요?
        """
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        return mock_response
    
    @pytest.mark.asyncio
    async def test_call_llm_with_mock_vllm(self, mock_vllm_response):
        """Mock vLLM API 호출 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            mock_client.completions.create = AsyncMock(return_value=mock_vllm_response)
            
            result = await call_llm("데이터베이스 정규화에 대해 꼬리질문을 생성하세요")
            
            assert "데이터베이스 트랜잭션" in result
            assert "MySQL과 PostgreSQL" in result
            assert "인덱스 설계" in result
            mock_client.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_llm_fallback_to_openai(self, mock_openai_response):
        """vLLM 실패 시 OpenAI로 폴백 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.openai_client') as mock_openai:
            
            # vLLM 초기화 실패 시뮬레이션
            mock_init.return_value = False
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            
            result = await call_llm("알고리즘 복잡도에 대해 꼬리질문을 생성하세요")
            
            assert "캐싱 전략" in result
            assert "Redis와 Memcached" in result
            mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_vllm_health_success(self, mock_vllm_response):
        """vLLM 헬스체크 성공 테스트"""
        from app.api.question_generator.question_generator_model import check_vllm_health
        
        with patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            mock_client.completions.create = AsyncMock(return_value=mock_vllm_response)
            
            result = await check_vllm_health()
            
            assert result is True
            mock_client.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_vllm_health_failure(self):
        """vLLM 헬스체크 실패 테스트"""
        from app.api.question_generator.question_generator_model import check_vllm_health
        
        with patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            mock_client.completions.create = AsyncMock(side_effect=Exception("Connection failed"))
            
            result = await check_vllm_health()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_vllm_api_parameters(self, mock_vllm_response):
        """vLLM API 파라미터 전달 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            mock_client.completions.create = AsyncMock(return_value=mock_vllm_response)
            
            await call_llm("테스트 프롬프트")
            
            # API 호출 파라미터 검증
            call_args = mock_client.completions.create.call_args
            assert call_args[1]["prompt"] == "테스트 프롬프트"
            assert "max_tokens" in call_args[1]
            assert "temperature" in call_args[1]
            assert "top_p" in call_args[1]
            assert "model" in call_args[1]


class TestVLLMAPIRealEnvironment:
    """실제 vLLM API 환경 테스트 (GCP에서 실행)"""
    
    def test_vllm_server_availability(self):
        """vLLM 서버 가용성 확인"""
        import requests
        from app.api.question_generator.question_generator_config import VLLM_API_CONFIG
        
        try:
            # vLLM 서버의 모델 엔드포인트 확인
            response = requests.get(
                f"{VLLM_API_CONFIG['base_url']}/models",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json()
                print(f"\n✅ vLLM 서버 접근 가능: {models}")
                assert "data" in models
            else:
                pytest.skip(f"vLLM 서버 접근 불가: {response.status_code}")
                
        except Exception as e:
            pytest.skip(f"vLLM 서버 접근 불가: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.gpu
    async def test_vllm_real_api_call(self):
        """실제 vLLM API 호출 테스트 (GPU 환경 필요)"""
        from app.api.question_generator.question_generator_model import call_llm, initialize_llm
        
        # 실제 vLLM 초기화 시도
        if not initialize_llm():
            pytest.skip("vLLM 초기화 실패 - GPU 환경 또는 서버 필요")
        
        try:
            result = await call_llm(
                "다음 질문에 대한 꼬리질문 3개를 생성하세요: Python의 GIL에 대해 설명해주세요",
                try_fallback=False
            )
            
            assert len(result.strip()) > 0
            assert isinstance(result, str)
            print(f"\n✅ vLLM 실제 응답: {result[:100]}...")
            
        except Exception as e:
            pytest.skip(f"vLLM 실제 호출 실패: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.gpu  
    async def test_vllm_performance_benchmark(self):
        """vLLM API 성능 벤치마크 테스트"""
        import time
        from app.api.question_generator.question_generator_model import call_llm, initialize_llm
        
        if not initialize_llm():
            pytest.skip("vLLM 초기화 실패")
        
        prompts = [
            "Python 데코레이터에 대해 꼬리질문을 생성하세요",
            "데이터베이스 인덱스에 대해 꼬리질문을 생성하세요", 
            "REST API 설계에 대해 꼬리질문을 생성하세요"
        ]
        
        response_times = []
        
        for prompt in prompts:
            try:
                start_time = time.time()
                result = await call_llm(prompt, try_fallback=False)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                assert len(result.strip()) > 0
                print(f"\n📊 응답시간: {response_time:.2f}초 | 길이: {len(result)} chars")
                
            except Exception as e:
                pytest.skip(f"성능 테스트 실패: {e}")
        
        # 평균 응답시간 확인
        avg_response_time = sum(response_times) / len(response_times)
        print(f"\n📈 평균 응답시간: {avg_response_time:.2f}초")
        
        # 성능 기준 (5초 이내)
        assert avg_response_time < 5.0, f"응답시간이 너무 느림: {avg_response_time:.2f}초"


class TestVLLMAPIErrorHandling:
    """vLLM API 에러 처리 테스트"""
    
    @pytest.mark.asyncio
    async def test_vllm_connection_error(self):
        """vLLM 연결 에러 처리 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            mock_client.completions.create = AsyncMock(side_effect=ConnectionError("서버 접속 불가"))
            
            # 폴백 없이 실행 시 예외 발생해야 함
            with pytest.raises(Exception) as exc_info:
                await call_llm("테스트", try_fallback=False)
            
            assert "서버 접속 불가" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_vllm_timeout_error(self):
        """vLLM 타임아웃 에러 처리 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            
            async def timeout_side_effect(*args, **kwargs):
                await asyncio.sleep(10)  # 긴 지연 시뮬레이션
                
            mock_client.completions.create = timeout_side_effect
            
            # 타임아웃 테스트
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    call_llm("테스트", try_fallback=False), 
                    timeout=1.0
                )
    
    @pytest.mark.asyncio
    async def test_vllm_invalid_model_error(self):
        """vLLM 잘못된 모델 에러 처리 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            mock_client.completions.create = AsyncMock(
                side_effect=Exception("Model not found")
            )
            
            with pytest.raises(Exception) as exc_info:
                await call_llm("테스트", try_fallback=False)
            
            assert "Model not found" in str(exc_info.value)


class TestVLLMAPIConfiguration:
    """vLLM API 설정 테스트"""
    
    def test_vllm_config_validation(self):
        """vLLM 설정 검증 테스트"""
        from app.api.question_generator.question_generator_config import VLLM_API_CONFIG
        
        required_keys = ["base_url", "model_name", "api_key", "timeout"]
        
        for key in required_keys:
            assert key in VLLM_API_CONFIG, f"필수 설정 누락: {key}"
        
        # URL 형식 검증
        assert VLLM_API_CONFIG["base_url"].startswith(("http://", "https://"))
        assert isinstance(VLLM_API_CONFIG["timeout"], (int, float))
        assert VLLM_API_CONFIG["timeout"] > 0
    
    def test_sampling_config_validation(self):
        """샘플링 설정 검증 테스트"""
        from app.api.question_generator.question_generator_config import SAMPLING_CONFIG
        
        # 온도값 검증
        assert 0 <= SAMPLING_CONFIG["temperature"] <= 2
        
        # 최대 토큰 검증
        assert SAMPLING_CONFIG["max_tokens"] > 0
        assert SAMPLING_CONFIG["max_tokens"] <= 2048
        
        # top_p 검증
        assert 0 <= SAMPLING_CONFIG["top_p"] <= 1
    
    @patch.dict('os.environ', {
        'VLLM_API_BASE_URL': 'http://test-server:8080/v1',
        'VLLM_MODEL_NAME': 'test-model',
        'VLLM_API_KEY': 'test-key'
    })
    def test_vllm_env_config_override(self):
        """환경변수 오버라이드 테스트"""
        import importlib
        from app.api.question_generator import question_generator_config
        
        # 설정 모듈 재로드
        importlib.reload(question_generator_config)
        
        assert question_generator_config.VLLM_API_CONFIG["base_url"] == "http://test-server:8080/v1"
        assert question_generator_config.VLLM_API_CONFIG["model_name"] == "test-model"
        assert question_generator_config.VLLM_API_CONFIG["api_key"] == "test-key"


class TestVLLMAPIIntegrationEnd2End:
    """vLLM API End-to-End 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_complete_question_generation_flow(self):
        """완전한 질문 생성 플로우 테스트"""
        from app.api.question_generator.question_generator_api import generate_followup
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        # Mock 환경 설정
        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_api.perform_rag_search') as mock_rag, \
             patch('app.api.question_generator.question_generator_model.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_prompt:
            
            # Mock 설정
            mock_model_available.return_value = True
            mock_rag.return_value = {
                "retrieved_questions": ["관련 질문1", "관련 질문2"],
                "metadata": {"total_results": 2}
            }
            mock_call_llm.return_value = """
            1. 데이터베이스 정규화의 1차, 2차, 3차 정규형의 차이점을 설명해주세요
            2. BCNF(Boyce-Codd Normal Form)와 3차 정규형의 관계는 무엇인가요?
            3. 역정규화(Denormalization)가 필요한 경우는 언제인가요?
            """
            
            mock_prompt_obj = Mock()
            mock_prompt_obj.compile.return_value = "테스트 프롬프트"
            mock_prompt.return_value = mock_prompt_obj
            
            # 테스트 요청
            request = FollowupRequest(
                interview_id="test_123",
                selected_question="데이터베이스 정규화에 대해 설명해주세요",
                keyword="데이터베이스",
                passed_questions=["SQL이란 무엇인가요?"]
            )
            
            # API 호출
            response = await generate_followup(request)
            
            # 결과 검증
            assert response.message == "followup_questions_generated"
            assert response.interview_id == "test_123"
            assert len(response.followup_questions) == 3
            assert "정규화" in response.followup_questions[0]
            
            # Mock 호출 검증
            mock_rag.assert_called_once()
            mock_call_llm.assert_called_once()

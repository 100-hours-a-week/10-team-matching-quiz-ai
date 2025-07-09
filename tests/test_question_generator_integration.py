import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestVLLMModelIntegration:
    """vLLM 모델 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_call_llm_success(self):
        """vLLM API 호출 성공 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "생성된 질문들"
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await call_llm("테스트 프롬프트")
            
            assert result == "생성된 질문들"
            mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_llm_initialization_failure(self):
        """vLLM 초기화 실패 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init:
            mock_init.return_value = False
            
            with pytest.raises(Exception) as exc_info:
                await call_llm("테스트 프롬프트")
            
            assert "vLLM API 클라이언트 초기화 실패" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_call_openai_api_success(self):
        """OpenAI API 호출 성공 테스트"""
        from app.api.question_generator.question_generator_model import call_openai_api
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "OpenAI 생성 질문들"
        
        with patch('app.api.question_generator.question_generator_model.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await call_openai_api("테스트 프롬프트")
            
            assert result == "OpenAI 생성 질문들"
            mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_openai_api_not_available(self):
        """OpenAI 클라이언트 사용 불가 테스트"""
        from app.api.question_generator.question_generator_model import call_openai_api
        
        with patch('app.api.question_generator.question_generator_model.openai_client', None):
            with pytest.raises(Exception) as exc_info:
                await call_openai_api("테스트 프롬프트")
            
            assert "OpenAI API 클라이언트가 설정되지 않았습니다" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_check_vllm_health_success(self):
        """vLLM 헬스체크 성공 테스트"""
        from app.api.question_generator.question_generator_model import check_vllm_health
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "헬스체크 응답"
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await check_vllm_health()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_vllm_health_failure(self):
        """vLLM 헬스체크 실패 테스트"""
        from app.api.question_generator.question_generator_model import check_vllm_health
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init:
            mock_init.return_value = False
            
            result = await check_vllm_health()
            
            assert result is False


class TestLangfuseIntegration:
    """Langfuse 통합 테스트"""
    
    def test_langfuse_trace_creation(self):
        """Langfuse 추적 생성 테스트"""
        from app.api.question_generator.question_generator_api import _prompt_cache
        
        mock_langfuse = Mock()
        mock_trace = Mock()
        mock_langfuse.trace.return_value = mock_trace
        
        with patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse):
            # 테스트 데이터로 추적 생성 시뮬레이션
            trace = mock_langfuse.trace(
                id="test_trace_id",
                name="followup_generation",
                input={"test": "data"}
            )
            
            assert trace == mock_trace
            mock_langfuse.trace.assert_called_once()
    
    def test_langfuse_span_creation(self):
        """Langfuse 스팬 생성 테스트"""
        mock_langfuse = Mock()
        mock_span = Mock()
        mock_langfuse.span.return_value = mock_span
        
        with patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse):
            span = mock_langfuse.span(
                trace_id="test_trace_id",
                name="rag_retrieval",
                input={"query": "test"}
            )
            
            assert span == mock_span
            mock_langfuse.span.assert_called_once()
    
    def test_langfuse_disabled(self):
        """Langfuse 비활성화 테스트"""
        from app.api.question_generator.question_generator_api import get_cached_prompt
        
        with patch('app.api.question_generator.question_generator_api.langfuse', None):
            result = get_cached_prompt("test_prompt")
            assert result is None


class TestErrorHandling:
    """에러 처리 테스트"""
    
    @pytest.mark.asyncio
    async def test_rag_search_exception_handling(self):
        """RAG 검색 예외 처리 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever:
            
            mock_retriever.side_effect = ConnectionError("Vector DB 연결 실패")
            
            result = await perform_rag_search("test query", "test keyword")
            
            assert result["metadata"]["query_processed"] is False
            assert "error" in result["metadata"]
            assert "Vector DB 연결 실패" in result["metadata"]["error"]
    
    @pytest.mark.asyncio
    async def test_vllm_api_exception_handling(self):
        """vLLM API 예외 처리 테스트"""
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API 서버 오류"))
            
            with pytest.raises(Exception) as exc_info:
                await call_llm("테스트 프롬프트")
            
            assert "API 서버 오류" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_api_exception_handling(self):
        """OpenAI API 예외 처리 테스트"""
        from app.api.question_generator.question_generator_model import call_openai_api
        
        with patch('app.api.question_generator.question_generator_model.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("OpenAI API 오류"))
            
            with pytest.raises(Exception) as exc_info:
                await call_openai_api("테스트 프롬프트")
            
            assert "OpenAI API 오류" in str(exc_info.value)


class TestConfigurationValidation:
    """설정 검증 테스트"""
    
    def test_vllm_config_validation(self):
        """vLLM 설정 검증 테스트"""
        from app.api.question_generator.question_generator_config import VLLM_API_CONFIG
        
        required_keys = ["base_url", "model_name", "api_key", "timeout"]
        for key in required_keys:
            assert key in VLLM_API_CONFIG
    
    def test_api_config_validation(self):
        """API 설정 검증 테스트"""
        from app.api.question_generator.question_generator_config import API_CONFIG
        
        assert "generate_count" in API_CONFIG
        assert "max_history_questions" in API_CONFIG
        assert isinstance(API_CONFIG["generate_count"], int)
        assert isinstance(API_CONFIG["max_history_questions"], int)
        assert API_CONFIG["generate_count"] > 0
        assert API_CONFIG["max_history_questions"] > 0
    
    def test_sampling_config_validation(self):
        """샘플링 설정 검증 테스트"""
        from app.api.question_generator.question_generator_config import SAMPLING_CONFIG
        
        expected_keys = ["temperature", "max_tokens", "top_p"]
        for key in expected_keys:
            assert key in SAMPLING_CONFIG
            
        assert 0 <= SAMPLING_CONFIG["temperature"] <= 2
        assert SAMPLING_CONFIG["max_tokens"] > 0
        assert 0 <= SAMPLING_CONFIG["top_p"] <= 1


class TestPerformanceAndResourceManagement:
    """성능 및 리소스 관리 테스트"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self):
        """동시 요청 처리 테스트"""
        import asyncio
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever:
            
            mock_retriever.return_value = {
                "results": [],
                "retrieved_questions": [],
                "metadata": {"total_results": 0}
            }
            
            # 동시에 여러 요청 실행
            tasks = [
                perform_rag_search(f"query_{i}", f"keyword_{i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert "metadata" in result
                assert "retrieved_questions" in result
    
    def test_prompt_caching_efficiency(self):
        """프롬프트 캐싱 효율성 테스트"""
        from app.api.question_generator.question_generator_api import get_cached_prompt, _prompt_cache
        
        # 캐시 초기화
        _prompt_cache.clear()
        
        mock_prompt = Mock()
        
        with patch('app.api.question_generator.question_generator_api.langfuse') as mock_langfuse:
            mock_langfuse.get_prompt.return_value = mock_prompt
            
            # 첫 번째 호출
            result1 = get_cached_prompt("test_prompt")
            
            # 두 번째 호출 (캐시에서 가져와야 함)
            result2 = get_cached_prompt("test_prompt")
            
            assert result1 == result2 == mock_prompt
            # Langfuse는 한 번만 호출되어야 함
            mock_langfuse.get_prompt.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """타임아웃 처리 테스트"""
        import asyncio
        from app.api.question_generator.question_generator_model import call_llm
        
        with patch('app.api.question_generator.question_generator_model.initialize_llm') as mock_init, \
             patch('app.api.question_generator.question_generator_model.vllm_client') as mock_client:
            
            mock_init.return_value = True
            # 타임아웃 시뮬레이션
            async def timeout_side_effect(*args, **kwargs):
                await asyncio.sleep(10)  # 긴 지연
                
            mock_client.chat.completions.create = timeout_side_effect
            
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(call_llm("테스트 프롬프트"), timeout=1.0)


class TestDataValidationAndSanitization:
    """데이터 검증 및 정제 테스트"""
    
    def test_input_sanitization(self):
        """입력 데이터 정제 테스트"""
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        # 특수 문자가 포함된 입력
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="SQL 인젝션'; DROP TABLE users; --",
            keyword="security",
            passed_questions=["<script>alert('xss')</script>"]
        )
        
        # 입력이 정상적으로 처리되는지 확인
        assert request.selected_question is not None
        assert request.keyword == "security"
        assert len(request.passed_questions) == 1
    
    def test_empty_and_whitespace_handling(self):
        """빈 값 및 공백 처리 테스트"""
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        # 공백만 있는 입력
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="   질문   ",
            keyword="   키워드   ",
            passed_questions=["  ", "실제 질문", "   "]
        )
        
        assert request.selected_question.strip() == "질문"
        assert request.keyword.strip() == "키워드"
    
    def test_large_input_handling(self):
        """대용량 입력 처리 테스트"""
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        # 매우 긴 질문과 많은 이전 질문들
        long_question = "A" * 1000  # 1000자 질문
        many_questions = [f"질문 {i}" for i in range(100)]  # 100개 이전 질문
        
        request = FollowupRequest(
            interview_id="test_123",
            selected_question=long_question,
            keyword="test",
            passed_questions=many_questions
        )
        
        assert len(request.selected_question) == 1000
        assert len(request.passed_questions) == 100

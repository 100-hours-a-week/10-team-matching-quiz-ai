import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestQuestionGeneratorSchema:
    """질문 생성기 스키마 테스트"""
    
    def test_followup_request_valid(self):
        """유효한 FollowupRequest 테스트"""
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화에 대해 설명해주세요",
            keyword="데이터베이스",
            passed_questions=["SQL이란 무엇인가요?"]
        )
        
        assert request.interview_id == "test_123"
        assert request.selected_question == "데이터베이스 정규화에 대해 설명해주세요"
        assert request.keyword == "데이터베이스"
        assert len(request.passed_questions) == 1
    
    def test_followup_request_optional_fields(self):
        """선택적 필드가 있는 FollowupRequest 테스트"""
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문"
        )
        
        assert request.keyword is None
        assert request.passed_questions is None
    
    def test_followup_response_valid(self):
        """유효한 FollowupResponse 테스트"""
        from app.api.question_generator.question_generator_schema import FollowupResponse
        
        response = FollowupResponse(
            message="followup_questions_generated",
            interview_id="test_123",
            followup_questions=["질문1", "질문2", "질문3"]
        )
        
        assert response.message == "followup_questions_generated"
        assert response.interview_id == "test_123"
        assert len(response.followup_questions) == 3


class TestQuestionGeneratorConfig:
    """질문 생성기 설정 테스트"""
    
    def test_config_loading(self):
        """설정 로딩 테스트"""
        from app.api.question_generator.question_generator_config import API_CONFIG, VLLM_API_CONFIG
        
        assert "generate_count" in API_CONFIG
        assert "max_history_questions" in API_CONFIG
        assert "base_url" in VLLM_API_CONFIG
        assert "model_name" in VLLM_API_CONFIG
    
    @patch.dict('os.environ', {'VLLM_API_BASE_URL': 'http://test:8000/v1'})
    def test_config_with_env_override(self):
        """환경변수 오버라이드 테스트"""
        # 설정 모듈을 다시 임포트하여 환경변수 반영
        import importlib
        from app.api.question_generator import question_generator_config
        importlib.reload(question_generator_config)
        
        assert question_generator_config.VLLM_API_CONFIG["base_url"] == "http://test:8000/v1"


class TestRAGSearchFunctionality:
    """RAG 검색 기능 단위 테스트"""
    
    @pytest.fixture
    def mock_rag_results(self):
        return {
            "results": [
                {"question": "정규화 1차 정규형은?", "metadata": {"score": 0.8}},
                {"question": "BCNF와 3차 정규형 차이는?", "metadata": {"score": 0.7}}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_perform_rag_search_success(self, mock_rag_results):
        """RAG 검색 성공 케이스"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever:
            
            mock_retriever.return_value = mock_rag_results
            
            result = await perform_rag_search("데이터베이스 정규화", "데이터베이스")
            
            assert result["metadata"]["query_processed"] is True
            assert len(result["retrieved_questions"]) == 2
            assert "정규화 1차 정규형은?" in result["retrieved_questions"]
            mock_retriever.assert_called_once_with("데이터베이스 정규화", "데이터베이스")
    
    @pytest.mark.asyncio
    async def test_perform_rag_search_vector_db_unavailable(self):
        """Vector DB 사용 불가 시"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', False):
            result = await perform_rag_search("test query", "test keyword")
            
            assert result["metadata"]["query_processed"] is False
            assert result["metadata"]["reason"] == "Vector DB not available"
            assert len(result["retrieved_questions"]) == 0
    
    @pytest.mark.asyncio
    async def test_perform_rag_search_exception_handling(self):
        """RAG 검색 예외 처리"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever:
            
            mock_retriever.side_effect = Exception("Database connection failed")
            
            result = await perform_rag_search("test query", "test keyword")
            
            assert result["metadata"]["query_processed"] is False
            assert "error" in result["metadata"]
            assert "Database connection failed" in result["metadata"]["error"]


class TestContextPreparation:
    """컨텍스트 준비 기능 테스트"""
    
    @pytest.fixture
    def sample_request(self):
        from app.api.question_generator.question_generator_schema import FollowupRequest
        return FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화에 대해 설명해주세요",
            keyword="데이터베이스",
            passed_questions=["SQL이란?", "인덱스란?"]
        )
    
    def test_prepare_context_full_data(self, sample_request):
        """모든 데이터가 있는 컨텍스트 준비"""
        from app.api.question_generator.question_generator_api import prepare_context
        
        rag_results = {
            "retrieved_questions": ["유사 질문1", "유사 질문2"],
            "metadata": {"total_results": 2, "search_time": 0.1}
        }
        
        context = prepare_context(sample_request, rag_results)
        
        assert context["selected_question"] == sample_request.selected_question
        assert context["keyword"] == sample_request.keyword
        assert "[이전 질문 목록]" in context["passed_questions"]
        assert "SQL이란?" in context["passed_questions"]
        assert "[유사한 기존 질문]" in context["retrieved_questions"]
        assert "유사 질문1" in context["retrieved_questions"]
        assert context["rag_metadata"]["total_results"] == 2
    
    def test_prepare_context_no_passed_questions(self):
        """이전 질문이 없는 경우"""
        from app.api.question_generator.question_generator_schema import FollowupRequest
        from app.api.question_generator.question_generator_api import prepare_context
        
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            keyword="테스트"
        )
        
        rag_results = {
            "retrieved_questions": [],
            "metadata": {"total_results": 0}
        }
        
        context = prepare_context(request, rag_results)
        
        assert context["passed_questions"] == ""
        assert context["retrieved_questions"] == ""
    
    def test_prepare_context_max_history_limit(self):
        """최대 히스토리 제한 테스트"""
        from app.api.question_generator.question_generator_schema import FollowupRequest
        from app.api.question_generator.question_generator_api import prepare_context
        
        # 많은 이전 질문 생성
        many_questions = [f"질문 {i}" for i in range(20)]
        
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            passed_questions=many_questions
        )
        
        rag_results = {"retrieved_questions": [], "metadata": {"total_results": 0}}
        
        with patch('app.api.question_generator.question_generator_api.MAX_HISTORY_QUESTIONS', 5):
            context = prepare_context(request, rag_results)
            
            # 마지막 5개만 포함되어야 함
            for i in range(15, 20):
                assert f"질문 {i}" in context["passed_questions"]
            
            # 처음 질문들은 포함되지 않아야 함
            assert "질문 0" not in context["passed_questions"]


class TestQuestionGenerationLogic:
    """질문 생성 로직 테스트"""
    
    @pytest.fixture
    def sample_context(self):
        return {
            "selected_question": "데이터베이스 정규화에 대해 설명해주세요",
            "keyword": "데이터베이스",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 0}
        }
    
    @pytest.fixture
    def sample_request(self):
        from app.api.question_generator.question_generator_schema import FollowupRequest
        return FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화에 대해 설명해주세요"
        )
    
    @pytest.mark.asyncio
    async def test_generate_questions_vllm_sufficient(self, sample_request, sample_context):
        """vLLM으로 충분한 질문 생성"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        
        mock_prompt = Mock()
        mock_prompt.compile.return_value = "compiled prompt"
        
        expected_questions = ["질문1", "질문2", "질문3"]
        
        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse:
            
            mock_get_prompt.return_value = mock_prompt
            mock_call_llm.return_value = "vllm response"
            mock_parse.return_value = expected_questions
            
            result = await generate_questions_with_fallback(sample_request, sample_context)
            
            assert result == expected_questions
            assert len(result) == 3
            mock_call_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_openai_fallback(self, sample_request, sample_context):
        """vLLM 부족 시 OpenAI 백업 사용"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        
        mock_prompt = Mock()
        mock_prompt.compile.return_value = "compiled prompt"
        
        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.call_openai_api') as mock_call_openai, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse:
            
            mock_get_prompt.return_value = mock_prompt
            mock_call_llm.return_value = "vllm response"
            mock_call_openai.return_value = "openai response"
            
            # vLLM에서 1개만, OpenAI에서 2개 추가
            mock_parse.side_effect = [
                ["vLLM 질문"],  # vLLM 결과
                ["OpenAI 질문1", "OpenAI 질문2"]  # OpenAI 결과
            ]
            
            result = await generate_questions_with_fallback(sample_request, sample_context)
            
            assert len(result) == 3
            assert "vLLM 질문" in result
            assert "OpenAI 질문1" in result
            assert "OpenAI 질문2" in result
    
    @pytest.mark.asyncio
    async def test_generate_questions_prompt_missing(self, sample_request, sample_context):
        """프롬프트 템플릿이 없는 경우"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        
        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt:
            mock_get_prompt.return_value = None
            
            with pytest.raises(Exception) as exc_info:
                await generate_questions_with_fallback(sample_request, sample_context)
            
            # HTTPException 또는 다른 예외가 발생해야 함
            assert exc_info.value is not None


class TestCachedPrompt:
    """프롬프트 캐시 테스트"""
    
    def test_get_cached_prompt_first_call(self):
        """첫 번째 호출 시 프롬프트 로드"""
        from app.api.question_generator.question_generator_api import get_cached_prompt, _prompt_cache
        
        # 캐시 초기화
        _prompt_cache.clear()
        
        mock_prompt = Mock()
        
        with patch('app.api.question_generator.question_generator_api.langfuse') as mock_langfuse:
            mock_langfuse.get_prompt.return_value = mock_prompt
            
            result = get_cached_prompt("test_prompt")
            
            assert result == mock_prompt
            mock_langfuse.get_prompt.assert_called_once_with("test_prompt")
            assert "test_prompt" in _prompt_cache
    
    def test_get_cached_prompt_cached_call(self):
        """캐시된 프롬프트 반환"""
        from app.api.question_generator.question_generator_api import get_cached_prompt, _prompt_cache
        
        # 캐시에 미리 저장
        mock_prompt = Mock()
        _prompt_cache["cached_prompt"] = mock_prompt
        
        with patch('app.api.question_generator.question_generator_api.langfuse') as mock_langfuse:
            result = get_cached_prompt("cached_prompt")
            
            assert result == mock_prompt
            # 이미 캐시되어 있으므로 langfuse 호출되지 않아야 함
            mock_langfuse.get_prompt.assert_not_called()
    
    def test_get_cached_prompt_no_langfuse(self):
        """Langfuse가 없는 경우"""
        from app.api.question_generator.question_generator_api import get_cached_prompt, _prompt_cache
        
        _prompt_cache.clear()
        
        with patch('app.api.question_generator.question_generator_api.langfuse', None):
            result = get_cached_prompt("test_prompt")
            
            assert result is None


class TestUtilityFunctions:
    """유틸리티 함수 테스트"""
    
    def test_question_parsing_numbered_list(self):
        """번호가 있는 질문 리스트 파싱"""
        from app.api.question_generator.question_generator_parser import parse_questions
        
        response = """
        1. 첫 번째 질문입니다
        2. 두 번째 질문입니다
        3. 세 번째 질문입니다
        """
        
        questions = parse_questions(response)
        
        assert len(questions) >= 3
        assert any("첫 번째 질문" in q for q in questions)
        assert any("두 번째 질문" in q for q in questions)
        assert any("세 번째 질문" in q for q in questions)
    
    def test_question_parsing_bullet_points(self):
        """불릿 포인트 질문 리스트 파싱"""
        from app.api.question_generator.question_generator_parser import parse_questions
        
        response = """
        - 첫 번째 질문입니다
        - 두 번째 질문입니다
        - 세 번째 질문입니다
        """
        
        questions = parse_questions(response)
        
        assert len(questions) >= 3
    
    def test_question_parsing_empty_input(self):
        """빈 입력 처리"""
        from app.api.question_generator.question_generator_parser import parse_questions
        
        assert parse_questions("") == []
        assert parse_questions("   ") == []
        assert parse_questions("\n\n\n") == []
    
    def test_question_parsing_malformed_input(self):
        """잘못된 형식 입력 처리"""
        from app.api.question_generator.question_generator_parser import parse_questions
        
        malformed = "이것은 질문 형식이 아닙니다."
        result = parse_questions(malformed)
        
        # 파싱 실패 시에도 예외가 발생하지 않아야 함
        assert isinstance(result, list)

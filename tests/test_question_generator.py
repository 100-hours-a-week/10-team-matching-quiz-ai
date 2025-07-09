import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import json

from app.api.question_generator.question_generator_schema import FollowupRequest, FollowupResponse


class TestFollowupQuestionGeneration:
    """꼬리질문 생성 기능 테스트"""
    
    @pytest.fixture
    def sample_request(self):
        """테스트용 요청 데이터"""
        return FollowupRequest(
            interview_id="test_interview_123",
            selected_question="데이터베이스 정규화에 대해 설명해주세요",
            keyword="데이터베이스",
            passed_questions=["SQL이란 무엇인가요?", "인덱스의 역할을 설명해주세요"]
        )
    
    @pytest.fixture
    def mock_rag_results(self):
        """테스트용 RAG 검색 결과"""
        return {
            "results": [
                {"question": "데이터베이스 정규화의 1차 정규형은 무엇인가요?"},
                {"question": "BCNF와 3차 정규형의 차이점은 무엇인가요?"}
            ],
            "retrieved_questions": [
                "데이터베이스 정규화의 1차 정규형은 무엇인가요?",
                "BCNF와 3차 정규형의 차이점은 무엇인가요?"
            ],
            "metadata": {
                "total_results": 2,
                "search_time": 0.1,
                "query_processed": True
            }
        }
    
    @pytest.fixture
    def mock_generated_questions(self):
        """테스트용 생성된 질문 목록"""
        return [
            "데이터베이스 정규화의 각 단계별 특징을 설명해주세요",
            "비정규화를 사용하는 경우는 언제인가요?",
            "함수 종속성이란 무엇이며, 정규화에서 어떤 역할을 하나요?"
        ]


class TestRAGSearch:
    """RAG 검색 기능 테스트"""
    
    @pytest.mark.asyncio
    async def test_perform_rag_search_success(self, mock_rag_results):
        """RAG 검색 성공 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever:
            
            mock_retriever.return_value = mock_rag_results
            
            result = await perform_rag_search("데이터베이스 정규화", "데이터베이스")
            
            assert result["metadata"]["query_processed"] is True
            assert len(result["retrieved_questions"]) == 2
            assert result["metadata"]["total_results"] == 2
    
    @pytest.mark.asyncio
    async def test_perform_rag_search_vector_db_unavailable(self):
        """Vector DB 사용 불가 시 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', False):
            result = await perform_rag_search("test query", "test keyword")
            
            assert result["metadata"]["query_processed"] is False
            assert result["metadata"]["reason"] == "Vector DB not available"
            assert len(result["retrieved_questions"]) == 0
    
    @pytest.mark.asyncio
    async def test_perform_rag_search_exception(self):
        """RAG 검색 예외 처리 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever:
            
            mock_retriever.side_effect = Exception("RAG search failed")
            
            result = await perform_rag_search("test query", "test keyword")
            
            assert result["metadata"]["query_processed"] is False
            assert "error" in result["metadata"]
            assert result["metadata"]["error"] == "RAG search failed"


class TestContextPreparation:
    """컨텍스트 준비 테스트"""
    
    def test_prepare_context_with_rag_results(self, sample_request, mock_rag_results):
        """RAG 결과가 있는 컨텍스트 준비 테스트"""
        from app.api.question_generator.question_generator_api import prepare_context
        
        context = prepare_context(sample_request, mock_rag_results)
        
        assert context["selected_question"] == sample_request.selected_question
        assert context["keyword"] == sample_request.keyword
        assert "[이전 질문 목록]" in context["passed_questions"]
        assert "[유사한 기존 질문]" in context["retrieved_questions"]
        assert context["rag_metadata"]["total_results"] == 2
    
    def test_prepare_context_without_passed_questions(self, mock_rag_results):
        """이전 질문이 없는 경우 테스트"""
        from app.api.question_generator.question_generator_api import prepare_context
        
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            keyword="테스트",
            passed_questions=None
        )
        
        context = prepare_context(request, mock_rag_results)
        
        assert context["passed_questions"] == ""
        assert context["selected_question"] == "테스트 질문"
    
    def test_prepare_context_without_rag_results(self, sample_request):
        """RAG 결과가 없는 경우 테스트"""
        from app.api.question_generator.question_generator_api import prepare_context
        
        empty_rag_results = {
            "retrieved_questions": [],
            "metadata": {"total_results": 0}
        }
        
        context = prepare_context(sample_request, empty_rag_results)
        
        assert context["retrieved_questions"] == ""
        assert context["rag_metadata"]["total_results"] == 0


class TestQuestionGeneration:
    """질문 생성 테스트"""
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_fallback_success(self, sample_request, mock_generated_questions):
        """질문 생성 성공 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        
        context = {
            "selected_question": sample_request.selected_question,
            "keyword": sample_request.keyword,
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 0}
        }
        
        mock_prompt = Mock()
        mock_prompt.compile.return_value = "test prompt"
        
        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse:
            
            mock_get_prompt.return_value = mock_prompt
            mock_call_llm.return_value = "generated response"
            mock_parse.return_value = mock_generated_questions
            
            result = await generate_questions_with_fallback(sample_request, context)
            
            assert len(result) == 3
            assert result == mock_generated_questions
            mock_call_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_openai_fallback(self, sample_request):
        """OpenAI 백업 사용 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        
        context = {
            "selected_question": sample_request.selected_question,
            "keyword": sample_request.keyword,
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 0}
        }
        
        mock_prompt = Mock()
        mock_prompt.compile.return_value = "test prompt"
        
        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.call_openai_api') as mock_call_openai, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse:
            
            mock_get_prompt.return_value = mock_prompt
            mock_call_llm.return_value = "vllm response"
            mock_call_openai.return_value = "openai response"
            
            # vLLM에서 1개만 생성, OpenAI에서 2개 추가 생성
            mock_parse.side_effect = [
                ["vLLM 생성 질문"],  # vLLM 결과
                ["OpenAI 질문1", "OpenAI 질문2"]  # OpenAI 결과
            ]
            
            result = await generate_questions_with_fallback(sample_request, context)
            
            assert len(result) == 3
            assert "vLLM 생성 질문" in result
            assert "OpenAI 질문1" in result
            assert "OpenAI 질문2" in result
    
    @pytest.mark.asyncio
    async def test_generate_questions_prompt_not_found(self, sample_request):
        """프롬프트 템플릿을 찾을 수 없는 경우 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from fastapi import HTTPException
        
        context = {}
        
        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt:
            mock_get_prompt.return_value = None
            
            with pytest.raises(HTTPException) as exc_info:
                await generate_questions_with_fallback(sample_request, context)
            
            assert exc_info.value.status_code == 500
            assert "프롬프트 템플릿을 로드할 수 없습니다" in str(exc_info.value.detail)


class TestAPIEndpoints:
    """API 엔드포인트 테스트"""
    
    @pytest.fixture
    def client(self):
        """테스트 클라이언트"""
        from app.main import app
        return TestClient(app)
    
    def test_followup_questions_endpoint_success(self, client, sample_request, mock_generated_questions):
        """꼬리질문 생성 엔드포인트 성공 테스트"""
        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_api.perform_rag_search') as mock_rag, \
             patch('app.api.question_generator.question_generator_api.generate_questions_with_fallback') as mock_generate:
            
            mock_model_available.return_value = True
            mock_rag.return_value = {
                "retrieved_questions": [],
                "metadata": {"total_results": 0}
            }
            mock_generate.return_value = mock_generated_questions
            
            response = client.post(
                "/api/question-generator/followup-questions",
                json=sample_request.dict()
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "followup_questions_generated"
            assert data["interview_id"] == sample_request.interview_id
            assert len(data["followup_questions"]) == 3
    
    def test_followup_questions_endpoint_model_unavailable(self, client, sample_request):
        """모델 사용 불가 시 테스트"""
        with patch('app.main.is_model_available') as mock_model_available:
            mock_model_available.return_value = False
            
            response = client.post(
                "/api/question-generator/followup-questions",
                json=sample_request.dict()
            )
            
            assert response.status_code == 503
            assert "질문 생성 모델이 사용할 수 없습니다" in response.json()["detail"]
    
    def test_followup_questions_endpoint_invalid_input(self, client):
        """잘못된 입력 데이터 테스트"""
        invalid_request = {
            "interview_id": "",
            "selected_question": "",
            "keyword": "test"
        }
        
        with patch('app.main.is_model_available') as mock_model_available:
            mock_model_available.return_value = True
            
            response = client.post(
                "/api/question-generator/followup-questions",
                json=invalid_request
            )
            
            assert response.status_code == 400
    
    def test_status_endpoint(self, client):
        """상태 확인 엔드포인트 테스트"""
        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_model.check_vllm_health') as mock_health:
            
            mock_model_available.return_value = True
            mock_health.return_value = True
            
            response = client.get("/api/question-generator/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "question_generator"
            assert data["model_available"] is True
            assert data["vllm_api_healthy"] is True
            assert data["status"] == "healthy"


class TestQuestionParser:
    """질문 파싱 테스트"""
    
    def test_parse_questions_success(self):
        """질문 파싱 성공 테스트"""
        from app.api.question_generator.question_generator_parser import parse_questions
        
        raw_response = """
        1. 데이터베이스 정규화의 각 단계별 특징을 설명해주세요
        2. 비정규화를 사용하는 경우는 언제인가요?
        3. 함수 종속성이란 무엇이며, 정규화에서 어떤 역할을 하나요?
        """
        
        questions = parse_questions(raw_response)
        
        assert len(questions) == 3
        assert "데이터베이스 정규화의 각 단계별 특징을 설명해주세요" in questions
        assert "비정규화를 사용하는 경우는 언제인가요?" in questions
    
    def test_parse_questions_empty_response(self):
        """빈 응답 파싱 테스트"""
        from app.api.question_generator.question_generator_parser import parse_questions
        
        questions = parse_questions("")
        assert len(questions) == 0
        
        questions = parse_questions("   \n\n  ")
        assert len(questions) == 0
    
    def test_parse_questions_malformed_response(self):
        """잘못된 형식 응답 파싱 테스트"""
        from app.api.question_generator.question_generator_parser import parse_questions
        
        malformed_response = "이것은 질문이 아닙니다."
        questions = parse_questions(malformed_response)
        
        # 파싱 실패 시 빈 리스트나 원본 텍스트 반환
        assert isinstance(questions, list)

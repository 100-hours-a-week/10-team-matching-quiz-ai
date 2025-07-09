import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestQuestionGeneratorAPIFunctions:
    """question_generator_api.py 함수별 직접 테스트"""
    
    @pytest.mark.asyncio
    async def test_perform_rag_search_direct(self):
        """perform_rag_search 함수 직접 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search
        
        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever:
            
            mock_retriever.return_value = {
                "results": [
                    {"question": "테스트 질문 1"},
                    {"question": "테스트 질문 2"}
                ]
            }
            
            result = await perform_rag_search("테스트 쿼리", "테스트 키워드")
            
            assert "retrieved_questions" in result
            assert len(result["retrieved_questions"]) == 2
            assert "테스트 질문 1" in result["retrieved_questions"]
            assert result["metadata"]["query_processed"] is True
    
    def test_prepare_context_direct(self):
        """prepare_context 함수 직접 테스트"""
        from app.api.question_generator.question_generator_api import prepare_context
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        # 테스트 요청 생성
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="Python GIL에 대해 설명해주세요",
            keyword="Python",
            passed_questions=["Python이란?", "변수 선언 방법은?"]
        )
        
        # RAG 결과 Mock
        rag_results = {
            "retrieved_questions": ["관련 질문 1", "관련 질문 2"],
            "metadata": {"total_results": 2}
        }
        
        # 함수 실행
        context = prepare_context(request, rag_results)
        
        # 결과 검증
        assert context["selected_question"] == "Python GIL에 대해 설명해주세요"
        assert context["keyword"] == "Python"
        assert "[이전 질문 목록]" in context["passed_questions"]
        assert "Python이란?" in context["passed_questions"]
        assert "[유사한 기존 질문]" in context["retrieved_questions"]
        assert "관련 질문 1" in context["retrieved_questions"]
    
    def test_get_cached_prompt_direct(self):
        """get_cached_prompt 함수 직접 테스트"""
        from app.api.question_generator.question_generator_api import get_cached_prompt, _prompt_cache
        
        # 캐시 초기화
        _prompt_cache.clear()
        
        # Mock Langfuse
        mock_prompt = Mock()
        mock_prompt.compile = Mock(return_value="컴파일된 프롬프트")
        
        with patch('app.api.question_generator.question_generator_api.langfuse') as mock_langfuse:
            mock_langfuse.get_prompt.return_value = mock_prompt
            
            # 첫 번째 호출 (langfuse에서 로드)
            result1 = get_cached_prompt("test_prompt")
            
            # 두 번째 호출 (캐시에서 로드)
            result2 = get_cached_prompt("test_prompt")
            
            assert result1 == mock_prompt
            assert result2 == mock_prompt
            # langfuse는 한 번만 호출되어야 함
            mock_langfuse.get_prompt.assert_called_once_with("test_prompt")
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_fallback_direct(self):
        """generate_questions_with_fallback 함수 직접 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest
        
        # 테스트 요청
        request = FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화에 대해 설명해주세요"
        )
        
        # 테스트 컨텍스트
        context = {
            "selected_question": "데이터베이스 정규화에 대해 설명해주세요",
            "keyword": "데이터베이스",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 0}
        }
        
        # Mock 설정
        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"
        
        generated_questions = [
            "정규화의 1차, 2차, 3차 정규형의 차이점은?",
            "BCNF(Boyce-Codd Normal Form)란 무엇인가요?",
            "역정규화가 필요한 경우는 언제인가요?"
        ]
        
        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse:
            
            mock_get_prompt.return_value = mock_prompt
            mock_call_llm.return_value = "vLLM 생성 응답"
            mock_parse.return_value = generated_questions
            
            # 함수 실행
            result = await generate_questions_with_fallback(request, context)
            
            # 결과 검증
            assert len(result) == 3
            assert result == generated_questions
            mock_call_llm.assert_called_once()
            mock_parse.assert_called_once_with("vLLM 생성 응답")


class TestQuestionGeneratorAPIImports:
    """API 모듈 임포트 및 기본 구조 테스트"""
    
    def test_api_module_imports(self):
        """API 모듈 기본 임포트 테스트"""
        # 함수들이 정상적으로 임포트되는지 확인
        from app.api.question_generator.question_generator_api import (
            router,
            perform_rag_search,
            prepare_context,
            generate_questions_with_fallback,
            get_cached_prompt
        )
        
        # Router가 FastAPI APIRouter 인스턴스인지 확인
        from fastapi import APIRouter
        assert isinstance(router, APIRouter)
        
        # 함수들이 호출 가능한지 확인
        assert callable(perform_rag_search)
        assert callable(prepare_context)
        assert callable(generate_questions_with_fallback)
        assert callable(get_cached_prompt)
    
    def test_api_constants(self):
        """API 상수들 테스트"""
        from app.api.question_generator.question_generator_api import (
            VECTOR_DB_AVAILABLE,
            GENERATE_COUNT,
            MAX_HISTORY_QUESTIONS
        )
        
        # 상수들이 정의되어 있는지 확인
        assert isinstance(VECTOR_DB_AVAILABLE, bool)
        assert isinstance(GENERATE_COUNT, int)
        assert isinstance(MAX_HISTORY_QUESTIONS, int)
        assert GENERATE_COUNT > 0
        assert MAX_HISTORY_QUESTIONS > 0
    
    def test_langfuse_initialization(self):
        """Langfuse 초기화 테스트"""
        from app.api.question_generator.question_generator_api import langfuse
        
        # langfuse가 None이거나 Langfuse 인스턴스여야 함
        if langfuse is not None:
            from langfuse import Langfuse
            assert isinstance(langfuse, Langfuse)


class TestQuestionGeneratorAPIEndpoint:
    """FastAPI 엔드포인트 테스트"""
    
    @pytest.fixture
    def client(self):
        """FastAPI 테스트 클라이언트"""
        from fastapi import FastAPI
        from app.api.question_generator.question_generator_api import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/question-generator")
        
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_status_endpoint(self, client):
        """상태 확인 엔드포인트 테스트"""
        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_model.check_vllm_health') as mock_health:
            
            mock_model_available.return_value = True
            mock_health.return_value = True
            
            response = client.get("/api/question-generator/status")
            
            assert response.status_code == 200
            data = response.json()
            assert "service" in data
            assert "model_available" in data
            assert "vllm_api_healthy" in data
            assert data["service"] == "question_generator"
    
    def test_followup_questions_endpoint_basic(self, client):
        """꼬리질문 생성 엔드포인트 기본 테스트"""
        request_data = {
            "interview_id": "test_123",
            "selected_question": "Python GIL에 대해 설명해주세요",
            "keyword": "Python"
        }
        
        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_api.perform_rag_search') as mock_rag, \
             patch('app.api.question_generator.question_generator_api.generate_questions_with_fallback') as mock_generate:
            
            mock_model_available.return_value = True
            mock_rag.return_value = {
                "retrieved_questions": [],
                "metadata": {"total_results": 0}
            }
            mock_generate.return_value = [
                "GIL의 장단점은 무엇인가요?",
                "멀티쓰레딩에서 GIL의 영향은?",
                "GIL을 우회하는 방법들을 설명해주세요"
            ]
            
            response = client.post("/api/question-generator/followup-questions", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "followup_questions_generated"
            assert data["interview_id"] == "test_123"
            assert len(data["followup_questions"]) == 3

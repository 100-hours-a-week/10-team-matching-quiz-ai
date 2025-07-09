"""
question_generator_api.py 모듈에 대한 완전한 테스트 커버리지
모든 함수와 엔드포인트를 모킹과 함께 테스트
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import uuid


class TestQuestionGeneratorAPIFunctions:
    """API 함수들의 직접 테스트"""

    @pytest.mark.asyncio
    async def test_perform_rag_search_success(self):
        """RAG 검색 성공 케이스"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        mock_retriever_result = {
            "results": [
                {"question": "데이터베이스의 ACID 속성이란?", "score": 0.9},
                {"question": "정규화의 목적은 무엇인가요?", "score": 0.8}
            ]
        }

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever, \
             patch('app.api.question_generator.question_generator_api.langfuse', None):

            mock_retriever.return_value = mock_retriever_result

            result = await perform_rag_search(
                query="데이터베이스 정규화",
                keyword="데이터베이스",
                trace_id="test_trace"
            )

            assert result["metadata"]["total_results"] == 2
            assert result["metadata"]["query_processed"] is True
            assert len(result["retrieved_questions"]) == 2
            assert "데이터베이스의 ACID 속성이란?" in result["retrieved_questions"]

    @pytest.mark.asyncio
    async def test_perform_rag_search_unavailable(self):
        """RAG 검색 불가능한 경우"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', False), \
             patch('app.api.question_generator.question_generator_api.langfuse', None):

            result = await perform_rag_search(
                query="데이터베이스 정규화",
                keyword="데이터베이스"
            )

            assert result["metadata"]["total_results"] == 0
            assert result["metadata"]["query_processed"] is False
            assert result["metadata"]["reason"] == "Vector DB not available"

    @pytest.mark.asyncio
    async def test_perform_rag_search_exception(self):
        """RAG 검색 예외 발생"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever, \
             patch('app.api.question_generator.question_generator_api.langfuse', None):

            mock_retriever.side_effect = Exception("RAG 검색 오류")

            result = await perform_rag_search(
                query="데이터베이스 정규화",
                keyword="데이터베이스"
            )

            assert result["metadata"]["total_results"] == 0
            assert result["metadata"]["query_processed"] is False
            assert "error" in result["metadata"]

    def test_prepare_context(self):
        """컨텍스트 준비 함수 테스트"""
        from app.api.question_generator.question_generator_api import prepare_context
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화란?",
            keyword="데이터베이스",
            passed_questions=["SQL이란?", "테이블 설계 방법은?"]
        )

        rag_results = {
            "retrieved_questions": ["ACID 속성이란?", "인덱스의 역할은?"],
            "metadata": {"total_results": 2}
        }

        with patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3), \
             patch('app.api.question_generator.question_generator_api.MAX_HISTORY_QUESTIONS', 10):

            context = prepare_context(req, rag_results)

            assert context["selected_question"] == "데이터베이스 정규화란?"
            assert context["keyword"] == "데이터베이스"
            assert "SQL이란?" in context["passed_questions"]
            assert "ACID 속성이란?" in context["retrieved_questions"]
            assert context["num_questions"] == 3

    def test_get_cached_prompt_with_langfuse(self):
        """프롬프트 캐시 테스트 (Langfuse 있음)"""
        from app.api.question_generator.question_generator_api import get_cached_prompt

        mock_langfuse = Mock()
        mock_prompt = Mock()
        mock_langfuse.get_prompt.return_value = mock_prompt

        with patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse), \
             patch('app.api.question_generator.question_generator_api._prompt_cache', {}):

            result = get_cached_prompt("test_prompt")

            assert result == mock_prompt
            mock_langfuse.get_prompt.assert_called_once_with("test_prompt")

    def test_get_cached_prompt_without_langfuse(self):
        """프롬프트 캐시 테스트 (Langfuse 없음)"""
        from app.api.question_generator.question_generator_api import get_cached_prompt

        with patch('app.api.question_generator.question_generator_api.langfuse', None), \
             patch('app.api.question_generator.question_generator_api._prompt_cache', {}):

            result = get_cached_prompt("test_prompt")

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_questions_with_fallback_success(self):
        """질문 생성 성공 케이스"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화란?"
        )

        context = {
            "selected_question": "데이터베이스 정규화란?",
            "keyword": "데이터베이스",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 0}
        }

        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"

        generated_questions = [
            "정규화의 1차, 2차, 3차 정규형의 차이점은?",
            "BCNF(Boyce-Codd Normal Form)란 무엇인가요?",
            "역정규화가 필요한 경우는 언제인가요?"
        ]

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse, \
             patch('app.api.question_generator.question_generator_api.langfuse', None), \
             patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3):

            mock_get_prompt.return_value = mock_prompt
            mock_call_llm.return_value = "vLLM 생성 응답"
            mock_parse.return_value = generated_questions

            result = await generate_questions_with_fallback(req, context)

            assert result == generated_questions
            mock_get_prompt.assert_called_once_with("followup_questions_generator")
            mock_call_llm.assert_called_once()
            mock_parse.assert_called_once_with("vLLM 생성 응답")

    @pytest.mark.asyncio
    async def test_generate_questions_with_fallback_insufficient(self):
        """질문 부족으로 OpenAI 호출하는 케이스"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화란?"
        )

        context = {
            "selected_question": "데이터베이스 정규화란?",
            "keyword": "데이터베이스",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 0}
        }

        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"
        
        mock_api_prompt = Mock()
        mock_api_prompt.compile.return_value = "API 프롬프트"

        vllm_questions = ["질문1"]  # 부족한 질문 수
        api_questions = ["질문2", "질문3"]

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm') as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.call_openai_api') as mock_call_api, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse, \
             patch('app.api.question_generator.question_generator_api.langfuse', None), \
             patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3):

            def mock_get_prompt_side_effect(name):
                if name == "followup_questions_generator":
                    return mock_prompt
                elif name == "followup_questions_generator_api":
                    return mock_api_prompt
                return None

            mock_get_prompt.side_effect = mock_get_prompt_side_effect
            mock_call_llm.return_value = "vLLM 응답"
            mock_call_api.return_value = "OpenAI 응답"
            
            def mock_parse_side_effect(response):
                if response == "vLLM 응답":
                    return vllm_questions
                elif response == "OpenAI 응답":
                    return api_questions
                return []

            mock_parse.side_effect = mock_parse_side_effect

            result = await generate_questions_with_fallback(req, context)

            assert len(result) == 3
            assert "질문1" in result
            assert "질문2" in result
            assert "질문3" in result
            mock_call_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_questions_with_fallback_no_prompt(self):
        """프롬프트가 없는 경우 예외 발생"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="데이터베이스 정규화란?"
        )

        context = {"selected_question": "테스트"}

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.langfuse', None):

            mock_get_prompt.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await generate_questions_with_fallback(req, context)

            assert exc_info.value.status_code == 500
            assert "프롬프트 템플릿" in exc_info.value.detail


class TestQuestionGeneratorAPIEndpoints:
    """FastAPI 엔드포인트 테스트"""

    @pytest.fixture
    def client(self):
        """테스트 클라이언트 설정"""
        from fastapi import FastAPI
        from app.api.question_generator.question_generator_api import router

        app = FastAPI()
        app.include_router(router, prefix="/api/question-generator")
        return TestClient(app)

    def test_status_endpoint_healthy(self, client):
        """상태 엔드포인트 테스트 (정상)"""
        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_model.check_vllm_health') as mock_health:

            mock_model_available.return_value = True
            # 직접 True 값을 반환하는 코루틴 모킹
            mock_health.return_value = True

            response = client.get("/api/question-generator/status")

            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "question_generator"
            assert data["model_available"] is True

    def test_status_endpoint_unhealthy(self, client):
        """상태 엔드포인트 테스트 (비정상)"""
        with patch('app.main.is_model_available') as mock_model_available:

            mock_model_available.return_value = False

            response = client.get("/api/question-generator/status")

            assert response.status_code == 200
            data = response.json()
            assert data["model_available"] is False
            assert data["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_followup_questions_endpoint_success(self, client):
        """꼬리질문 생성 엔드포인트 성공 케이스"""
        request_data = {
            "interview_id": "test_123",
            "selected_question": "Python GIL에 대해 설명해주세요",
            "keyword": "Python"
        }

        mock_questions = [
            "GIL의 장단점은 무엇인가요?",
            "멀티쓰레딩에서 GIL의 영향은?",
            "GIL을 우회하는 방법들을 설명해주세요"
        ]

        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_api.perform_rag_search') as mock_rag, \
             patch('app.api.question_generator.question_generator_api.generate_questions_with_fallback') as mock_generate, \
             patch('app.api.question_generator.question_generator_api.langfuse', None):

            mock_model_available.return_value = True
            mock_rag.return_value = {
                "retrieved_questions": [],
                "metadata": {"total_results": 0}
            }
            mock_generate.return_value = mock_questions

            response = client.post("/api/question-generator/followup-questions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "followup_questions_generated"
            assert data["interview_id"] == "test_123"
            assert len(data["followup_questions"]) == 3

    def test_followup_questions_endpoint_model_unavailable(self, client):
        """모델 사용 불가능한 경우"""
        request_data = {
            "interview_id": "test_123",
            "selected_question": "Python GIL에 대해 설명해주세요"
        }

        with patch('app.main.is_model_available') as mock_model_available:
            mock_model_available.return_value = False

            response = client.post("/api/question-generator/followup-questions", json=request_data)

            assert response.status_code == 503
            assert "질문 생성 모델이 사용할 수 없습니다" in response.json()["detail"]

    def test_followup_questions_endpoint_invalid_input(self, client):
        """잘못된 입력 데이터"""
        with patch('app.main.is_model_available') as mock_model_available:
            mock_model_available.return_value = True

            # 빈 질문
            response = client.post("/api/question-generator/followup-questions", json={
                "interview_id": "test_123",
                "selected_question": ""
            })
            assert response.status_code == 400
            assert "메인 질문은 필수입니다" in response.json()["detail"]

            # 빈 interview_id
            response = client.post("/api/question-generator/followup-questions", json={
                "interview_id": "",
                "selected_question": "테스트 질문"
            })
            assert response.status_code == 400
            assert "interview_id는 필수입니다" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_followup_questions_endpoint_generation_error(self, client):
        """질문 생성 중 오류 발생"""
        request_data = {
            "interview_id": "test_123",
            "selected_question": "Python GIL에 대해 설명해주세요"
        }

        with patch('app.main.is_model_available') as mock_model_available, \
             patch('app.api.question_generator.question_generator_api.perform_rag_search') as mock_rag, \
             patch('app.api.question_generator.question_generator_api.generate_questions_with_fallback') as mock_generate, \
             patch('app.api.question_generator.question_generator_api.langfuse', None):

            mock_model_available.return_value = True
            mock_rag.return_value = {
                "retrieved_questions": [],
                "metadata": {"total_results": 0}
            }
            mock_generate.side_effect = Exception("질문 생성 실패")

            response = client.post("/api/question-generator/followup-questions", json=request_data)

            assert response.status_code == 500
            assert "질문 생성 실패" in response.json()["detail"]


class TestQuestionGeneratorAPIModuleLevel:
    """모듈 레벨 테스트"""

    def test_module_imports(self):
        """모듈 임포트 테스트"""
        import app.api.question_generator.question_generator_api as api_module
        
        # 주요 객체들이 정의되어 있는지 확인
        assert hasattr(api_module, 'router')
        assert hasattr(api_module, 'logger')
        assert hasattr(api_module, 'langfuse')
        assert hasattr(api_module, 'GENERATE_COUNT')
        assert hasattr(api_module, 'MAX_HISTORY_QUESTIONS')

    def test_module_constants(self):
        """모듈 상수 테스트"""
        from app.api.question_generator.question_generator_api import GENERATE_COUNT, MAX_HISTORY_QUESTIONS
        
        assert isinstance(GENERATE_COUNT, int)
        assert isinstance(MAX_HISTORY_QUESTIONS, int)
        assert GENERATE_COUNT > 0
        assert MAX_HISTORY_QUESTIONS > 0

    def test_vector_db_availability(self):
        """Vector DB 가용성 확인"""
        from app.api.question_generator.question_generator_api import VECTOR_DB_AVAILABLE
        
        assert isinstance(VECTOR_DB_AVAILABLE, bool)

    def test_langfuse_initialization(self):
        """Langfuse 초기화 테스트"""
        # 모듈을 직접 테스트하여 langfuse 객체 확인
        from app.api.question_generator.question_generator_api import langfuse
        
        # langfuse가 설정되어 있는지 또는 None인지 확인
        # 실제 환경에서는 설정에 따라 다를 수 있음
        assert langfuse is None or hasattr(langfuse, 'get_prompt')

    def test_prompt_cache_functionality(self):
        """프롬프트 캐시 기능 테스트"""
        from app.api.question_generator.question_generator_api import _prompt_cache, get_cached_prompt
        
        # 캐시 초기화
        _prompt_cache.clear()
        
        # Langfuse 없이 호출
        with patch('app.api.question_generator.question_generator_api.langfuse', None):
            result = get_cached_prompt("test_prompt")
            assert result is None
            assert len(_prompt_cache) == 0

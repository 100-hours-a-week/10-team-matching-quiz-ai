"""
question_generator_api.py 모듈에 대한 완전한 테스트 커버리지
모든 함수와 엔드포인트를 모킹과 함께 테스트
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import uuid
import time
import json


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

    @pytest.mark.asyncio
    async def test_perform_rag_search_with_langfuse_span_success(self):
        """Langfuse span과 함께 RAG 검색 성공 - output과 metadata 파라미터 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        mock_retriever_result = {
            "results": [
                {"question": "테스트 질문1", "score": 0.9},
                {"question": "테스트 질문2", "score": 0.8}
            ]
        }

        mock_langfuse = Mock()
        mock_span = Mock()
        mock_langfuse.span.return_value = mock_span

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever, \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse):

            mock_retriever.return_value = mock_retriever_result

            result = await perform_rag_search(
                query="테스트 쿼리",
                keyword="키워드",
                trace_id="test_trace"
            )

            # 결과 검증
            assert result["metadata"]["total_results"] == 2
            assert result["metadata"]["query_processed"] is True
            assert len(result["retrieved_questions"]) == 2

            # Langfuse span 호출 검증
            mock_langfuse.span.assert_called_once()
            mock_span.end.assert_called_once()

            # span.end() 호출 시 output과 metadata 파라미터 검증
            call_args = mock_span.end.call_args
            assert call_args[1]["output"]["retrieved_questions"] == result["retrieved_questions"]
            assert call_args[1]["output"]["total_results"] == result["metadata"]["total_results"]
            assert call_args[1]["output"]["search_successful"] == result["metadata"]["query_processed"]
            assert call_args[1]["metadata"] == result["metadata"]

    @pytest.mark.asyncio
    async def test_perform_rag_search_with_langfuse_span_exception(self):
        """Langfuse span과 함께 RAG 검색 예외 - error 파라미터 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        mock_langfuse = Mock()
        mock_span = Mock()
        mock_langfuse.span.return_value = mock_span

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever, \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse):

            mock_retriever.side_effect = Exception("RAG 오류")

            result = await perform_rag_search(
                query="테스트 쿼리",
                keyword="키워드",
                trace_id="test_trace"
            )

            # 결과 검증
            assert result["metadata"]["total_results"] == 0
            assert result["metadata"]["query_processed"] is False
            assert "error" in result["metadata"]

            # Langfuse span 호출 검증
            mock_langfuse.span.assert_called_once()
            mock_span.end.assert_called_once()

            # span.end() 호출 시 error와 output 파라미터 검증
            call_args = mock_span.end.call_args
            assert "error" in call_args[1]
            assert call_args[1]["error"]["message"] == "RAG 검색 실패: RAG 오류"
            assert call_args[1]["error"]["type"] == "Exception"
            assert call_args[1]["output"] == result

    @pytest.mark.asyncio
    async def test_perform_rag_search_vector_db_unavailable_with_span(self):
        """Vector DB 사용 불가 시 Langfuse span 처리"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        mock_langfuse = Mock()
        mock_span = Mock()
        mock_langfuse.span.return_value = mock_span

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', False), \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse):

            result = await perform_rag_search(
                query="테스트 쿼리",
                keyword="키워드",
                trace_id="test_trace"
            )

            # 결과 검증
            assert result["metadata"]["total_results"] == 0
            assert result["metadata"]["query_processed"] is False
            assert result["metadata"]["reason"] == "Vector DB not available"

            # Langfuse span 호출 검증
            mock_langfuse.span.assert_called_once()
            mock_span.end.assert_called_once()

            # span.end() 호출 시 output과 metadata 파라미터 검증
            call_args = mock_span.end.call_args
            assert call_args[1]["output"]["retrieved_questions"] == []
            assert call_args[1]["output"]["total_results"] == 0
            assert call_args[1]["output"]["search_successful"] is False
            assert call_args[1]["metadata"] == result["metadata"]

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

    @pytest.mark.asyncio
    async def test_generate_questions_with_fallback_with_prompt_span(self):
        """프롬프트 span과 함께 질문 생성 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            keyword="키워드"
        )

        context = {
            "selected_question": "테스트 질문",
            "keyword": "키워드",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 2}
        }

        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"

        mock_langfuse = Mock()
        mock_prompt_span = Mock()
        mock_langfuse.span.return_value = mock_prompt_span

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt', return_value=mock_prompt), \
             patch('app.api.question_generator.question_generator_api.call_llm', new_callable=AsyncMock) as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse, \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse), \
             patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3):

            mock_call_llm.return_value = "질문들"
            mock_parse.return_value = ["질문1", "질문2", "질문3"]

            questions = await generate_questions_with_fallback(req, context)

            assert len(questions) == 3
            assert req.keyword == "키워드"


class TestQuestionGeneratorAPILangfuseIntegration:
    """Langfuse 통합 관련 추가 테스트"""

    @pytest.mark.asyncio
    async def test_perform_rag_search_langfuse_span_output_metadata(self):
        """RAG 검색에서 langfuse span.end()의 output과 metadata 파라미터 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        mock_retriever_result = {
            "results": [
                {"question": "테스트 질문1", "score": 0.9},
                {"question": "테스트 질문2", "score": 0.8}
            ]
        }

        mock_langfuse = Mock()
        mock_span = Mock()
        mock_langfuse.span.return_value = mock_span

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', True), \
             patch('app.api.question_generator.question_generator_api.question_rag_retriever') as mock_retriever, \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse):

            mock_retriever.return_value = mock_retriever_result

            result = await perform_rag_search(
                query="테스트 쿼리",
                keyword="키워드",
                trace_id="test_trace"
            )

            # span.end() 호출 시 output과 metadata 파라미터 검증
            mock_span.end.assert_called_once()
            call_args = mock_span.end.call_args
            
            # output 파라미터 검증
            assert "output" in call_args[1]
            output = call_args[1]["output"]
            assert output["retrieved_questions"] == result["retrieved_questions"]
            assert output["total_results"] == result["metadata"]["total_results"]
            assert output["search_successful"] == result["metadata"]["query_processed"]
            
            # metadata 파라미터 검증
            assert "metadata" in call_args[1]
            assert call_args[1]["metadata"] == result["metadata"]

    @pytest.mark.asyncio
    async def test_generate_questions_prompt_span_output_metadata(self):
        """질문 생성에서 prompt span.end()의 output과 metadata 파라미터 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            keyword="키워드"
        )

        context = {
            "selected_question": "테스트 질문",
            "keyword": "키워드", 
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3,
            "rag_metadata": {"total_results": 2}
        }

        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"

        mock_langfuse = Mock()
        mock_prompt_span = Mock()
        mock_langfuse.span.return_value = mock_prompt_span

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt', return_value=mock_prompt), \
             patch('app.api.question_generator.question_generator_api.call_llm', new_callable=AsyncMock) as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse, \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse), \
             patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3):

            mock_call_llm.return_value = "질문들"
            mock_parse.return_value = ["질문1", "질문2", "질문3"]

            await generate_questions_with_fallback(req, context, trace_id="test_trace")

            # prompt span.end() 호출 시 output과 metadata 파라미터 검증
            mock_prompt_span.end.assert_called_once()
            call_args = mock_prompt_span.end.call_args
            
            # output 파라미터 검증
            assert "output" in call_args[1]
            output = call_args[1]["output"]
            assert output["compiled_prompt_length"] == len("컴파일된 프롬프트")
            
            # metadata 파라미터 검증
            assert "metadata" in call_args[1]
            metadata = call_args[1]["metadata"]
            assert metadata["selected_question_length"] == len(req.selected_question)
            assert metadata["keyword"] == req.keyword
            assert metadata["rag_results_count"] == 2

    @pytest.mark.asyncio
    async def test_generate_questions_fallback_span_output(self):
        """OpenAI fallback span.end()의 output 파라미터 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            keyword="키워드"
        )

        context = {
            "selected_question": "테스트 질문",
            "keyword": "키워드",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3
        }

        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"

        mock_api_prompt = Mock()
        mock_api_prompt.compile.return_value = "API 프롬프트"

        mock_langfuse = Mock()
        mock_prompt_span = Mock()
        mock_fallback_span = Mock()
        # span() 호출 순서에 따라 다른 mock 반환
        mock_langfuse.span.side_effect = [mock_prompt_span, mock_fallback_span]

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm', new_callable=AsyncMock) as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.call_openai_api', new_callable=AsyncMock) as mock_call_openai, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse, \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse), \
             patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3):

            mock_get_prompt.side_effect = [mock_prompt, mock_api_prompt]
            mock_call_llm.return_value = "질문들"
            mock_call_openai.return_value = "추가 질문들"
            
            # vLLM에서 1개만 생성, OpenAI에서 2개 추가
            mock_parse.side_effect = [["질문1"], ["질문2", "질문3"]]

            await generate_questions_with_fallback(req, context, trace_id="test_trace")

            # fallback span.end() 호출 시 output 파라미터 검증
            mock_fallback_span.end.assert_called_once()
            call_args = mock_fallback_span.end.call_args
            
            # output 파라미터 검증
            assert "output" in call_args[1]
            output = call_args[1]["output"]
            assert output["additional_questions_generated"] == 2
            assert output["unique_questions_added"] == 2
            assert output["final_question_count"] == 3


class TestQuestionGeneratorAPIEdgeCases:
    """Edge cases와 if-else, try-except 블록 테스트"""

    def test_vector_db_import_error_simulation(self):
        """Vector DB import 실패 시뮬레이션 테스트"""
        # 모듈 레벨 import error를 시뮬레이션하기 위해 sys.modules을 조작
        import sys
        
        # 기존 모듈 백업
        original_modules = sys.modules.copy()
        
        # vector_db 모듈 제거
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('app.vector_db')]
        for module in modules_to_remove:
            del sys.modules[module]
        
        # app.api.question_generator.question_generator_api 모듈도 제거하여 재로드 필요
        if 'app.api.question_generator.question_generator_api' in sys.modules:
            del sys.modules['app.api.question_generator.question_generator_api']
        
        try:
            # ImportError를 발생시키기 위해 모듈 경로를 차단
            import builtins
            original_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if name == 'app.vector_db.retriever':
                    raise ImportError("No module named 'app.vector_db.retriever'")
                return original_import(name, *args, **kwargs)
            
            builtins.__import__ = mock_import
            
            # 모듈 재로드
            import importlib
            import app.api.question_generator.question_generator_api as reloaded_module
            importlib.reload(reloaded_module)
            
            # except ImportError 블록이 실행되었는지 확인
            assert reloaded_module.VECTOR_DB_AVAILABLE is False
            assert reloaded_module.question_rag_retriever is None
            
        finally:
            # 원상복구
            builtins.__import__ = original_import
            sys.modules.clear()
            sys.modules.update(original_modules)

    @pytest.mark.asyncio
    async def test_perform_rag_search_no_langfuse_no_trace_id(self):
        """langfuse=None이고 trace_id=None인 경우 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', False), \
             patch('app.api.question_generator.question_generator_api.langfuse', None):

            result = await perform_rag_search(
                query="테스트 쿼리",
                keyword="키워드"
                # trace_id를 명시적으로 전달하지 않음 (None)
            )

            # langfuse span이 생성되지 않았는지 확인
            assert result["metadata"]["total_results"] == 0
            assert result["metadata"]["query_processed"] is False

    @pytest.mark.asyncio
    async def test_perform_rag_search_langfuse_but_no_trace_id(self):
        """langfuse는 있지만 trace_id=None인 경우 테스트"""
        from app.api.question_generator.question_generator_api import perform_rag_search

        mock_langfuse = Mock()

        with patch('app.api.question_generator.question_generator_api.VECTOR_DB_AVAILABLE', False), \
             patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse):

            result = await perform_rag_search(
                query="테스트 쿼리",
                keyword="키워드"
                # trace_id=None (기본값)
            )

            # langfuse.span이 호출되지 않았는지 확인
            mock_langfuse.span.assert_not_called()
            assert result["metadata"]["total_results"] == 0

    @pytest.mark.asyncio
    async def test_generate_questions_with_fallback_no_langfuse(self):
        """langfuse=None인 경우 질문 생성 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            keyword="키워드"
        )

        context = {
            "selected_question": "테스트 질문",
            "keyword": "키워드",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3
        }

        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt', return_value=mock_prompt), \
             patch('app.api.question_generator.question_generator_api.call_llm', new_callable=AsyncMock) as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse, \
             patch('app.api.question_generator.question_generator_api.langfuse', None), \
             patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3):

            mock_call_llm.return_value = "질문들"
            mock_parse.return_value = ["질문1", "질문2", "질문3"]

            questions = await generate_questions_with_fallback(req, context)

            # langfuse span이 생성되지 않았어도 질문은 정상 생성
            assert len(questions) == 3
            mock_call_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_questions_fallback_no_api_prompt(self):
        """API 프롬프트가 None인 경우 fallback 테스트"""
        from app.api.question_generator.question_generator_api import generate_questions_with_fallback
        from app.api.question_generator.question_generator_schema import FollowupRequest

        req = FollowupRequest(
            interview_id="test_123",
            selected_question="테스트 질문",
            keyword="키워드"
        )

        context = {
            "selected_question": "테스트 질문",
            "keyword": "키워드",
            "passed_questions": "",
            "retrieved_questions": "",
            "num_questions": 3
        }

        mock_prompt = Mock()
        mock_prompt.compile.return_value = "컴파일된 프롬프트"

        with patch('app.api.question_generator.question_generator_api.get_cached_prompt') as mock_get_prompt, \
             patch('app.api.question_generator.question_generator_api.call_llm', new_callable=AsyncMock) as mock_call_llm, \
             patch('app.api.question_generator.question_generator_api.call_openai_api', new_callable=AsyncMock) as mock_call_openai, \
             patch('app.api.question_generator.question_generator_api.parse_questions') as mock_parse, \
             patch('app.api.question_generator.question_generator_api.langfuse', None), \
             patch('app.api.question_generator.question_generator_api.GENERATE_COUNT', 3):

            # 첫 번째 프롬프트는 있지만, API 프롬프트는 None
            mock_get_prompt.side_effect = [mock_prompt, None]
            mock_call_llm.return_value = "질문들"
            mock_parse.return_value = ["질문1"]  # 부족한 질문 수

            # API 프롬프트가 없어서 OpenAI 호출이 안 됨
            assert len(questions) == 1
            mock_call_openai.assert_not_called()

    @pytest.mark.asyncio
    async def test_followup_questions_endpoint_empty_selected_question(self):
        """빈 선택 질문으로 엔드포인트 테스트"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.question_generator.question_generator_api import router

        app = FastAPI()
        app.include_router(router, prefix="/api/question-generator")
        client = TestClient(app)

        with patch('app.main.is_model_available', return_value=True):
            # 빈 문자열 질문
            response = client.post("/api/question-generator/followup-questions", json={
                "interview_id": "test_123",
                "selected_question": "   "  # 공백만 있는 문자열
            })
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_followup_questions_endpoint_empty_interview_id(self):
        """빈 interview_id로 엔드포인트 테스트"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.question_generator.question_generator_api import router

        app = FastAPI()
        app.include_router(router, prefix="/api/question-generator")
        client = TestClient(app)

        with patch('app.main.is_model_available', return_value=True):
            # 빈 문자열 interview_id
            response = client.post("/api/question-generator/followup-questions", json={
                "interview_id": "   ",  # 공백만 있는 문자열
                "selected_question": "테스트 질문"
            })
            assert response.status_code == 400

    def test_get_cached_prompt_cache_hit(self):
        """프롬프트 캐시 히트 테스트"""
        from app.api.question_generator.question_generator_api import get_cached_prompt

        mock_prompt = Mock()
        mock_langfuse = Mock()

        with patch('app.api.question_generator.question_generator_api.langfuse', mock_langfuse), \
             patch('app.api.question_generator.question_generator_api._prompt_cache', {"test_prompt": mock_prompt}):

            result = get_cached_prompt("test_prompt")

            # 캐시에서 가져왔으므로 langfuse.get_prompt이 호출되지 않음
            assert result == mock_prompt
            mock_langfuse.get_prompt.assert_not_called()
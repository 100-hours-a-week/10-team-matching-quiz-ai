import pytest
from unittest.mock import patch, Mock
from app.api.stt_feedback.service import generate_feedback_service

class TestFeedbackGenerateService:
    @patch('google.generativeai.GenerativeModel')
    def test_generate_feedback_gemini_success(self, mock_model):
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.text = '{"model_answer": "답안", "feedback": {"good_points": "좋음", "areas_for_improvement": "개선", "overall_score": 4, "detailed_analysis": "분석"}}'
        mock_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_instance
        result = generate_feedback_service.generate_feedback_gemini('질문', '답변')
        assert result['model_answer'] == '답안'
        assert result['feedback']['overall_score'] == 4

    @patch('google.generativeai.GenerativeModel')
    def test_generate_feedback_gemini_json_error(self, mock_model):
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.text = 'not a json'
        mock_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_instance
        result = generate_feedback_service.generate_feedback_gemini('질문', '답변')
        assert 'JSON 파싱 오류' in result['model_answer']

    @patch('google.generativeai.GenerativeModel')
    def test_generate_feedback_gemini_api_error(self, mock_model):
        mock_instance = Mock()
        mock_instance.generate_content.side_effect = Exception('API error')
        mock_model.return_value = mock_instance
        result = generate_feedback_service.generate_feedback_gemini('질문', '답변')
        assert '모범답안 생성 실패' in result['model_answer']

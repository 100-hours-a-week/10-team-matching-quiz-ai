import pytest
from unittest.mock import patch, Mock
from app.api.stt_feedback import stt_feedback_api
from app.api.stt_feedback.stt_feedback_schema import VoiceFeedbackRequest, QuestionItem

class TestFeedbackAPI:
    @patch('app.api.stt_feedback.stt_feedback_api.run_feedback_pipeline')
    def test_process_feedback_request_success(self, mock_pipeline):
        mock_pipeline.return_value.feedbackLists = [
            Mock(segment_id='seg1', question='q1', model_answer='a1', feedback={'overall_score': 5, 'detailed_analysis': '분석', 'good_points': '좋음', 'areas_for_improvement': '개선'})
        ]
        request = VoiceFeedbackRequest(
            recording_url='http://test.com/audio.mp3',
            question_lists=[QuestionItem(segment_id='seg1', start_time=0, end_time=10, question='q1')]
        )
        result = stt_feedback_api.process_feedback_request(request)
        assert 'feedback_lists' in result
        assert result['feedback_lists'][0]['model_answer'] == 'a1'

    @patch('app.api.stt_feedback.stt_feedback_api.run_feedback_pipeline')
    def test_process_feedback_request_empty_questions(self, mock_pipeline):
        request = VoiceFeedbackRequest(
            recording_url='http://test.com/audio.mp3',
            question_lists=[]
        )
        with pytest.raises(ValueError):
            stt_feedback_api.process_feedback_request(request)

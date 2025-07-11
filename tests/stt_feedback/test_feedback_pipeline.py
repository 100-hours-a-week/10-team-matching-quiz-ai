import pytest
from unittest.mock import patch, Mock
from app.api.stt_feedback.service import feedback_pipline
from app.api.stt_feedback.stt_feedback_schema import QuestionItem

class TestFeedbackPipeline:
    @patch('app.api.stt_feedback.service.feedback_pipline.download_audio')
    @patch('app.api.stt_feedback.service.feedback_pipline.cut_audio')
    @patch('app.api.stt_feedback.service.feedback_pipline.transcribe_whisperx')
    @patch('app.api.stt_feedback.service.feedback_pipline.generate_feedback_gemini')
    def test_run_feedback_pipeline_success(self, mock_gemini, mock_stt, mock_cut, mock_download):
        mock_download.return_value = '/tmp/test.mp3'
        mock_cut.return_value = Mock()
        mock_stt.return_value = 'mock transcript'
        mock_gemini.return_value = {
            'model_answer': '모범답안',
            'feedback': {
                'good_points': '좋음',
                'areas_for_improvement': '개선 필요',
                'overall_score': 5,
                'detailed_analysis': '상세 분석'
            }
        }
        questions = [
            QuestionItem(segment_id='seg1', start_time=0, end_time=10, question='질문1'),
            QuestionItem(segment_id='seg2', start_time=10, end_time=20, question='질문2')
        ]
        result = feedback_pipline.run_feedback_pipeline('http://test.com/audio.mp3', questions)
        assert len(result.feedbackLists) == 2
        assert result.feedbackLists[0].model_answer == '모범답안'
        assert result.feedbackLists[0].feedback['overall_score'] == 5

    @patch('app.api.stt_feedback.service.feedback_pipline.download_audio')
    def test_run_feedback_pipeline_download_fail(self, mock_download):
        mock_download.side_effect = Exception('Download failed')
        questions = [QuestionItem(segment_id='seg1', start_time=0, end_time=10, question='질문1')]
        with pytest.raises(Exception):
            feedback_pipline.run_feedback_pipeline('http://test.com/audio.mp3', questions)

    @patch('app.api.stt_feedback.service.feedback_pipline.download_audio')
    @patch('app.api.stt_feedback.service.feedback_pipline.cut_audio')
    @patch('app.api.stt_feedback.service.feedback_pipline.transcribe_whisperx')
    @patch('app.api.stt_feedback.service.feedback_pipline.generate_feedback_gemini')
    def test_run_feedback_pipeline_partial_fail(self, mock_gemini, mock_stt, mock_cut, mock_download):
        mock_download.return_value = '/tmp/test.mp3'
        mock_cut.side_effect = [Mock(), Exception('Cut failed')]
        mock_stt.return_value = 'mock transcript'
        mock_gemini.return_value = {
            'model_answer': '모범답안',
            'feedback': {
                'good_points': '좋음',
                'areas_for_improvement': '개선 필요',
                'overall_score': 5,
                'detailed_analysis': '상세 분석'
            }
        }
        questions = [
            QuestionItem(segment_id='seg1', start_time=0, end_time=10, question='질문1'),
            QuestionItem(segment_id='seg2', start_time=10, end_time=20, question='질문2')
        ]
        result = feedback_pipline.run_feedback_pipeline('http://test.com/audio.mp3', questions)
        assert len(result.feedbackLists) == 1
        assert result.feedbackLists[0].segment_id == 'seg1'

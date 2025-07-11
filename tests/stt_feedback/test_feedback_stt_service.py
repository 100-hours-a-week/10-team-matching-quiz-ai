import pytest
from unittest.mock import patch, Mock
from app.api.stt_feedback.service import stt_service
from pydub import AudioSegment

class TestFeedbackSTTService:
    @patch('app.api.stt_feedback.service.stt_service.WhisperXModel')
    @patch('app.api.stt_feedback.service.stt_service.AudioSegment')
    @patch('app.api.stt_feedback.service.stt_service.tempfile')
    def test_transcribe_whisperx_success(self, mock_tempfile, mock_audio_segment, mock_whisperx_model):
        mock_segment = Mock(spec=AudioSegment)
        mock_file = Mock()
        mock_file.name = '/tmp/test.mp3'
        mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_file
        mock_audio_segment.export = Mock()
        mock_whisperx_model.model.transcribe.return_value = {
            'segments': [{'text': 'hello'}, {'text': 'world'}]
        }
        with patch('app.api.stt_feedback.service.stt_service.whisperx.load_audio', return_value='audio'):
            result = stt_service.transcribe_whisperx(mock_segment)
            assert result == 'hello world'

    @patch('app.api.stt_feedback.service.stt_service.WhisperXModel')
    @patch('app.api.stt_feedback.service.stt_service.AudioSegment')
    @patch('app.api.stt_feedback.service.stt_service.tempfile')
    def test_transcribe_whisperx_failure(self, mock_tempfile, mock_audio_segment, mock_whisperx_model):
        mock_segment = Mock(spec=AudioSegment)
        mock_file = Mock()
        mock_file.name = '/tmp/test.mp3'
        mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_file
        mock_audio_segment.export = Mock()
        mock_whisperx_model.model.transcribe.side_effect = Exception('STT error')
        with patch('app.api.stt_feedback.service.stt_service.whisperx.load_audio', return_value='audio'):
            with pytest.raises(Exception):
                stt_service.transcribe_whisperx(mock_segment)

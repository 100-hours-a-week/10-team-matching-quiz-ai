import pytest
from unittest.mock import patch, Mock
from app.api.stt_feedback import stt_model_loader

class TestFeedbackModelLoader:
    @patch('app.api.stt_feedback.stt_model_loader.whisperx')
    @patch('app.api.stt_feedback.stt_model_loader.torch')
    def test_load_model_success(self, mock_torch, mock_whisperx):
        mock_torch.cuda.is_available.return_value = True
        mock_whisperx.load_model.return_value = Mock()
        stt_model_loader.WhisperXModel.model = None
        stt_model_loader.WhisperXModel.load_model()
        assert stt_model_loader.WhisperXModel.model is not None

    @patch('app.api.stt_feedback.stt_model_loader.whisperx')
    @patch('app.api.stt_feedback.stt_model_loader.torch')
    def test_load_model_failure(self, mock_torch, mock_whisperx):
        mock_torch.cuda.is_available.return_value = False
        mock_whisperx.load_model.side_effect = Exception('Model load error')
        stt_model_loader.WhisperXModel.model = None
        with pytest.raises(Exception):
            stt_model_loader.WhisperXModel.load_model()

    def test_ensure_loaded(self):
        stt_model_loader.WhisperXModel.model = None
        with patch.object(stt_model_loader.WhisperXModel, 'load_model') as mock_load:
            stt_model_loader.WhisperXModel.ensure_loaded()
            mock_load.assert_called_once()

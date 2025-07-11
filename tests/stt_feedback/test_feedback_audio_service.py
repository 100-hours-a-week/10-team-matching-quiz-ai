import pytest
from unittest.mock import patch, Mock
from app.api.stt_feedback.service import audio_service
from pydub import AudioSegment

class TestAudioService:
    def test_download_audio_success(self):
        url = "https://example.com/test.mp3"
        mock_content = b"fake audio data"
        with patch("requests.get") as mock_get, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_response = Mock()
            mock_response.content = mock_content
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            mock_file = Mock()
            mock_file.name = "/tmp/test.mp3"
            mock_tmp.return_value = mock_file
            result = audio_service.download_audio(url)
            assert result == "/tmp/test.mp3"
            mock_get.assert_called_once_with(url)
            mock_file.write.assert_called_once_with(mock_content)

    def test_download_audio_failure(self):
        url = "https://example.com/test.mp3"
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Download failed")
            with pytest.raises(Exception):
                audio_service.download_audio(url)

    def test_cut_audio_success(self):
        audio_path = "/tmp/test.mp3"
        with patch("pydub.AudioSegment.from_file") as mock_from_file:
            mock_audio = Mock(spec=AudioSegment)
            mock_from_file.return_value = mock_audio
            result = audio_service.cut_audio(audio_path, 0, 10)
            mock_from_file.assert_called_once_with(audio_path)
            assert result == mock_audio.__getitem__.return_value

    def test_cut_audio_failure(self):
        audio_path = "/tmp/test.mp3"
        with patch("pydub.AudioSegment.from_file") as mock_from_file:
            mock_from_file.side_effect = Exception("Audio load failed")
            with pytest.raises(Exception):
                audio_service.cut_audio(audio_path, 0, 10)

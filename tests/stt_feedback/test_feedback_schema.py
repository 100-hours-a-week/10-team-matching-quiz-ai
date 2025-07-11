from app.api.stt_feedback.stt_feedback_schema import QuestionItem, VoiceFeedbackRequest, StandardResponse, InvalidRequestResponse, TokenExpiredResponse, AlreadySubmittedResponse, InternalServerErrorResponse

class TestFeedbackSchema:
    def test_question_item(self):
        item = QuestionItem(segment_id='seg1', start_time=0, end_time=10, question='q1')
        assert item.segment_id == 'seg1'
        assert item.start_time == 0
        assert item.end_time == 10
        assert item.question == 'q1'

    def test_voice_feedback_request(self):
        item = QuestionItem(segment_id='seg1', start_time=0, end_time=10, question='q1')
        req = VoiceFeedbackRequest(recording_url='http://test.com/audio.mp3', question_lists=[item])
        assert req.recording_url == 'http://test.com/audio.mp3'
        assert len(req.question_lists) == 1

    def test_standard_response(self):
        resp = StandardResponse(message='ok', data={'key': 'value'})
        assert resp.message == 'ok'
        assert resp.data['key'] == 'value'

    def test_invalid_request_response(self):
        resp = InvalidRequestResponse(data={'reason': 'error'})
        assert resp.message == 'invalid_request'
        assert resp.data['reason'] == 'error'

    def test_token_expired_response(self):
        resp = TokenExpiredResponse()
        assert resp.message == 'token_expired'
        assert resp.data is None

    def test_already_submitted_response(self):
        resp = AlreadySubmittedResponse()
        assert resp.message == 'already_submit'
        assert resp.data is None

    def test_internal_server_error_response(self):
        resp = InternalServerErrorResponse()
        assert resp.message == 'internal_server_error'
        assert resp.data is None

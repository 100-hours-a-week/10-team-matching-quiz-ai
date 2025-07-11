from app.api.stt_feedback.stt_feedback_model import FeedbackItem, FeedbackResponse

class TestFeedbackModel:
    def test_feedback_item_creation(self):
        item = FeedbackItem(
            segment_id='seg1',
            question='q1',
            model_answer='answer',
            feedback={'overall_score': 5, 'detailed_analysis': '분석', 'good_points': '좋음', 'areas_for_improvement': '개선'}
        )
        assert item.segment_id == 'seg1'
        assert item.model_answer == 'answer'
        assert item.feedback['overall_score'] == 5

    def test_feedback_response_creation(self):
        item = FeedbackItem(
            segment_id='seg1',
            question='q1',
            model_answer='answer',
            feedback={'overall_score': 5}
        )
        response = FeedbackResponse(feedbackLists=[item])
        assert len(response.feedbackLists) == 1
        assert response.feedbackLists[0].segment_id == 'seg1'

from unittest.mock import Mock, AsyncMock


class MockVLLMClient:
    """Mock vLLM client for testing"""
    
    def __init__(self):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = AsyncMock()
    
    async def create_completion(self, prompt: str, **kwargs):
        """Mock completion creation"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = self._generate_mock_questions(prompt)
        return mock_response
    
    def _generate_mock_questions(self, prompt: str):
        """Generate mock questions based on prompt"""
        if "데이터베이스" in prompt:
            return """
            1. 데이터베이스 정규화의 1차 정규형에 대해 설명해주세요
            2. BCNF와 3차 정규형의 차이점은 무엇인가요?
            3. 함수 종속성이란 무엇이며, 정규화에서 어떤 역할을 하나요?
            """
        elif "알고리즘" in prompt:
            return """
            1. 시간복잡도와 공간복잡도의 차이점을 설명해주세요
            2. 동적 계획법의 핵심 원리는 무엇인가요?
            3. 그래프 탐색에서 DFS와 BFS의 차이점은 무엇인가요?
            """
        else:
            return """
            1. 첫 번째 모의 질문입니다
            2. 두 번째 모의 질문입니다
            3. 세 번째 모의 질문입니다
            """


class MockOpenAIClient:
    """Mock OpenAI client for testing"""
    
    def __init__(self):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = AsyncMock()
    
    async def create_completion(self, messages, **kwargs):
        """Mock completion creation"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = self._generate_openai_questions()
        return mock_response
    
    def _generate_openai_questions(self):
        """Generate mock OpenAI questions"""
        return """
        1. OpenAI 생성 질문 1
        2. OpenAI 생성 질문 2
        3. OpenAI 생성 질문 3
        """


class MockRAGRetriever:
    """Mock RAG retriever for testing"""
    
    def __init__(self):
        self.default_results = {
            "results": [
                {"question": "유사한 질문 1", "metadata": {"score": 0.9}},
                {"question": "유사한 질문 2", "metadata": {"score": 0.8}},
                {"question": "유사한 질문 3", "metadata": {"score": 0.7}}
            ]
        }
    
    def __call__(self, query: str, keyword: str = ""):
        """Mock RAG search"""
        if "데이터베이스" in query.lower():
            return {
                "results": [
                    {"question": "데이터베이스 정규화의 1차 정규형은?", "metadata": {"score": 0.9}},
                    {"question": "BCNF와 3차 정규형의 차이는?", "metadata": {"score": 0.8}}
                ]
            }
        elif "알고리즘" in query.lower():
            return {
                "results": [
                    {"question": "시간복잡도 계산 방법은?", "metadata": {"score": 0.9}},
                    {"question": "동적 계획법 예시를 설명하세요", "metadata": {"score": 0.8}}
                ]
            }
        else:
            return self.default_results


class MockLangfuse:
    """Mock Langfuse client for testing"""
    
    def __init__(self):
        self.traces = []
        self.spans = []
    
    def trace(self, **kwargs):
        """Mock trace creation"""
        mock_trace = Mock()
        mock_trace.update = Mock()
        mock_trace.id = kwargs.get('id', 'mock_trace_id')
        self.traces.append(mock_trace)
        return mock_trace
    
    def span(self, **kwargs):
        """Mock span creation"""
        mock_span = Mock()
        mock_span.end = Mock()
        mock_span.trace_id = kwargs.get('trace_id', 'mock_trace_id')
        self.spans.append(mock_span)
        return mock_span
    
    def get_prompt(self, name: str):
        """Mock prompt retrieval"""
        mock_prompt = Mock()
        mock_prompt.compile = Mock()
        
        if name == "followup_questions_generator":
            mock_prompt.compile.return_value = """
            다음 질문을 바탕으로 꼬리질문을 생성하세요:
            선택된 질문: {selected_question}
            키워드: {keyword}
            이전 질문들: {passed_questions}
            유사 질문들: {retrieved_questions}
            
            {num_questions}개의 꼬리질문을 생성하세요.
            """
        elif name == "followup_questions_generator_api":
            mock_prompt.compile.return_value = """
            OpenAI API용 꼬리질문 생성 프롬프트:
            선택된 질문: {selected_question}
            키워드: {keyword}
            
            {ungenerated_questions_num}개의 추가 질문을 생성하세요.
            """
        
        return mock_prompt


def create_mock_environment():
    """Create a complete mock environment for testing"""
    return {
        'vllm_client': MockVLLMClient(),
        'openai_client': MockOpenAIClient(),
        'rag_retriever': MockRAGRetriever(),
        'langfuse': MockLangfuse()
    }

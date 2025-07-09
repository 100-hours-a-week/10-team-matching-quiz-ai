import pytest
import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("ENABLED_MODELS", "question_generator")
    monkeypatch.setenv("VLLM_API_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_NAME", "test-model")

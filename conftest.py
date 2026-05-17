"""Root conftest — ensure the src layout is importable without editable install."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock the anthropic SDK before any test module imports it.
# The real SDK hangs on import; all Anthropic API calls are mocked in individual tests.
class _FakeAPIError(Exception):
    def __init__(self, message="", request=None, body=None):
        super().__init__(message)
        self.message = message
        self.request = request
        self.body = body

_mock_anthropic = MagicMock()
_mock_anthropic.APIError = _FakeAPIError
sys.modules["anthropic"] = _mock_anthropic

# Mock the openai SDK before any test module imports it.
# Prevents real network calls; all OpenAI API calls are mocked in individual tests.
class _FakeOpenAIAPIError(Exception):
    def __init__(self, message="", request=None, body=None):
        super().__init__(message)
        self.message = message
        self.request = request
        self.body = body

_mock_openai = MagicMock()
_mock_openai.APIError = _FakeOpenAIAPIError
sys.modules["openai"] = _mock_openai

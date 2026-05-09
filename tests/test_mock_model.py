from agentlab.models.base import LLMMessage
from agentlab.models.mock_model import MockModel


def test_mock_model_returns_default_response() -> None:
    model = MockModel()

    response = model.generate([LLMMessage(role="user", content="hello")])

    assert isinstance(response.content, str)
    assert response.content


def test_mock_model_can_match_keywords() -> None:
    model = MockModel(
        keyword_responses={
            "langgraph": "LangGraph mock",
            "autogen": "AutoGen mock",
        }
    )

    response = model.generate([LLMMessage(role="user", content="compare LangGraph and others")])

    assert response.content == "LangGraph mock"

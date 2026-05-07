from agentlab.models.mock_model import MockModel


def test_mock_model_returns_default_response() -> None:
    model = MockModel()

    text = model.generate([{"role": "user", "content": "hello"}])

    assert isinstance(text, str)
    assert text


def test_mock_model_can_match_keywords() -> None:
    model = MockModel(
        keyword_responses={
            "langgraph": "LangGraph mock",
            "autogen": "AutoGen mock",
        }
    )

    text = model.generate([{"role": "user", "content": "compare LangGraph and others"}])

    assert text == "LangGraph mock"

from pydantic import ValidationError
import pytest

from agentlab.core.message import MESSAGE_TYPES, Message


def test_message_defaults() -> None:
    msg = Message(sender="planner", receiver="searcher", content="test")

    assert msg.type == "task"
    assert msg.metadata == {}


def test_message_metadata_is_not_shared() -> None:
    first = Message(sender="a", receiver="b", content="x")
    second = Message(sender="a", receiver="b", content="y")

    first.metadata["k"] = "v"

    assert second.metadata == {}


def test_message_type_must_be_supported() -> None:
    with pytest.raises(ValidationError):
        Message(sender="a", receiver="b", content="x", type="unsupported")


@pytest.mark.parametrize("message_type", sorted(MESSAGE_TYPES))
def test_message_supports_known_types(message_type: str) -> None:
    msg = Message(sender="a", receiver="b", content="x", type=message_type)
    assert msg.type == message_type

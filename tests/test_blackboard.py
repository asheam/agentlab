import json

from agentlab.workspace.blackboard import Blackboard


def test_blackboard_write_read_and_list() -> None:
    board = Blackboard()

    board.write("plan", ["q1", "q2"], author="planner")

    assert board.read("plan") == ["q1", "q2"]
    assert board.read("missing") is None
    assert board.list() == ["plan"]


def test_blackboard_export_json(tmp_path) -> None:
    board = Blackboard()
    board.write("notes", "summary", author="reader")

    out_path = tmp_path / "workspace.json"
    board.export_json(out_path)

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["notes"]["author"] == "reader"
    assert payload["notes"]["value"] == "summary"

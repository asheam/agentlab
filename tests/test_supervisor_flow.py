import json

from agentlab.multi_agent.supervisor import build_default_supervisor


def test_supervisor_generates_outputs(tmp_path) -> None:
    supervisor = build_default_supervisor(output_dir=tmp_path)

    result = supervisor.run("研究 LangGraph、AutoGen、CrewAI 的区别")

    report_path = result["report_path"]
    trace_path = result["trace_path"]
    workspace_path = result["workspace_path"]

    assert report_path.exists()
    assert trace_path.exists()
    assert workspace_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "# Deep Research Report" in report_text
    assert "LangGraph" in report_text
    assert "## Search Mode Summary" in report_text
    assert "DuckDuckGo hits" in report_text
    assert "Wikipedia hits" in report_text

    workspace_payload = json.loads(workspace_path.read_text(encoding="utf-8"))
    for key in ["plan", "search_results", "notes", "critique", "report"]:
        assert key in workspace_payload


def test_supervisor_report_contains_framework_comparison(tmp_path) -> None:
    supervisor = build_default_supervisor(output_dir=tmp_path)

    result = supervisor.run("研究 LangGraph、AutoGen、CrewAI 的区别")
    report_text = result["report_path"].read_text(encoding="utf-8")
    workspace_payload = json.loads(result["workspace_path"].read_text(encoding="utf-8"))

    assert "mock://langgraph/overview" in report_text
    assert "mock://autogen/overview" in report_text
    assert "mock://crewai/overview" in report_text
    assert workspace_payload["critique"]["value"]["verdict"] == "acceptable"

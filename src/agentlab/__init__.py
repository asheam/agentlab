from agentlab.multi_agent.supervisor import SupervisorConfig, build_default_supervisor


__all__ = ["build_default_supervisor", "SupervisorConfig"]


def main() -> None:
    topic = "研究 LangGraph、AutoGen、CrewAI 的区别"
    supervisor = build_default_supervisor()
    outputs = supervisor.run(topic)
    print("AgentLab demo finished.")
    for key, path in outputs.items():
        print(f"{key}: {path}")

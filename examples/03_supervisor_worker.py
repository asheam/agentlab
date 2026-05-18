from __future__ import annotations

from agentlab.multi_agent.supervisor import build_default_supervisor


if __name__ == "__main__":
    topic = "比较多 Agent 框架的协作方式"
    supervisor = build_default_supervisor()
    outputs = supervisor.run(topic)

    print("Supervisor-Worker demo completed")
    for key, path in outputs.items():
        print(f"{key}: {path}")

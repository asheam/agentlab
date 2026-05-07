from __future__ import annotations

import sys

from agentlab.multi_agent.supervisor import build_default_supervisor


def main() -> None:
    topic = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "研究 LangGraph、AutoGen、CrewAI 的区别"
    )
    supervisor = build_default_supervisor(output_dir="outputs")
    outputs = supervisor.run(topic)

    print("Deep Research completed")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()

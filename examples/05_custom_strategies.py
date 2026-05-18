from __future__ import annotations

import argparse

from agentlab.multi_agent.supervisor import SupervisorConfig, build_default_supervisor
from agentlab.strategy_presets import apply_strategy_preset


DEFAULT_TOPIC = "研究 LangGraph、AutoGen、CrewAI 的区别"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep research with the built-in concise strategy preset."
    )
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="Research topic")
    parser.add_argument(
        "--output-dir",
        default="outputs/custom_strategies",
        help="Output directory for generated artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    base_config = SupervisorConfig(
        output_dir=args.output_dir,
        search_mode="mock",
        critic_mode="rule",
    )
    config = apply_strategy_preset(base_config, "concise")

    supervisor = build_default_supervisor(config=config)
    outputs = supervisor.run(args.topic)

    print("Custom strategy demo completed")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

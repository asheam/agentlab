from __future__ import annotations

import argparse

from agentlab.multi_agent.supervisor import (
    SupervisorConfig,
    build_default_supervisor,
)


DEFAULT_TOPIC = "研究 LangGraph、AutoGen、CrewAI 的区别"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentLab deep research demo.")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="Research topic")
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI-compatible API model instead of offline mock flow.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for report.md, trace.json, workspace.json",
    )
    parser.add_argument(
        "--search-mode",
        choices=["mock", "real"],
        default="mock",
        help="Search backend mode. 'mock' is offline-safe, 'real' uses web requests with fallback.",
    )
    parser.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Disable fallback to mock search when real search fails or returns empty.",
    )
    parser.add_argument(
        "--search-providers",
        default="duckduckgo,wikipedia,tavily",
        help=(
            "Comma-separated providers for real search mode. "
            "Supported: duckduckgo,wikipedia,tavily"
        ),
    )
    parser.add_argument(
        "--critic-mode",
        choices=["auto", "rule", "llm"],
        default="auto",
        help="Critic strategy: auto tries LLM then falls back to rule, rule is deterministic only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    search_providers = [item.strip() for item in args.search_providers.split(",") if item.strip()]
    config = SupervisorConfig(
        output_dir=args.output_dir,
        use_openai_model=args.use_openai,
        search_mode=args.search_mode,
        allow_search_fallback=not args.no_search_fallback,
        search_providers=search_providers,
        critic_mode=args.critic_mode,
    )
    supervisor = build_default_supervisor(config=config)
    outputs = supervisor.run(args.topic)

    print("Deep Research completed")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0

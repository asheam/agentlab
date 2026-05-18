from __future__ import annotations

import argparse
from typing import Literal, cast

from agentlab.cli_config import CLIConfigError, load_cli_config
from agentlab.multi_agent.supervisor import (
    SupervisorConfig,
    build_default_supervisor,
)
from agentlab.strategy_presets import (
    StrategyPreset,
    apply_strategy_preset,
    iter_strategy_presets,
)


DEFAULT_TOPIC = "研究 LangGraph、AutoGen、CrewAI 的区别"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SEARCH_MODE = "mock"
DEFAULT_SEARCH_PROVIDERS = ["duckduckgo", "wikipedia", "tavily"]
DEFAULT_CRITIC_MODE: Literal["auto", "rule", "llm"] = "auto"
DEFAULT_STRATEGY_PRESET: StrategyPreset = "default"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentLab deep research demo.")
    parser.add_argument("topic", nargs="?", help="Research topic")
    parser.add_argument(
        "--config",
        help="Path to YAML config file (for example: agentlab.yaml).",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI-compatible API model instead of offline mock flow.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for report.md, trace.json, workspace.json",
    )
    parser.add_argument(
        "--search-mode",
        choices=["mock", "real"],
        help="Search backend mode. 'mock' is offline-safe, 'real' uses web requests with fallback.",
    )
    parser.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Disable fallback to mock search when real search fails or returns empty.",
    )
    parser.add_argument(
        "--search-providers",
        help=(
            "Comma-separated providers for real search mode. "
            "Supported: duckduckgo,wikipedia,tavily"
        ),
    )
    parser.add_argument(
        "--critic-mode",
        choices=["auto", "rule", "llm"],
        help="Critic strategy: auto tries LLM then falls back to rule, rule is deterministic only.",
    )
    parser.add_argument(
        "--strategy-preset",
        choices=["default", "concise"],
        help="Optional strategy preset. 'concise' injects custom planner/search/reader/critic/writer strategies.",
    )
    parser.add_argument(
        "--list-strategy-presets",
        action="store_true",
        help="List available strategy presets and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_strategy_presets:
        print("Available strategy presets:")
        for name, description in iter_strategy_presets():
            print(f"- {name}: {description}")
        return 0

    try:
        file_config = load_cli_config(args.config)
    except CLIConfigError as exc:
        print(f"Config error: {exc}")
        return 2

    topic = args.topic if args.topic is not None else _config_str(file_config, "topic", DEFAULT_TOPIC)
    use_openai = args.use_openai or _config_bool(file_config, "use_openai", False)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _config_str(file_config, "output_dir", DEFAULT_OUTPUT_DIR)
    )
    search_mode = cast(
        Literal["mock", "real"],
        args.search_mode
        if args.search_mode is not None
        else _config_str(file_config, "search_mode", DEFAULT_SEARCH_MODE),
    )
    no_search_fallback = args.no_search_fallback or _config_bool(
        file_config, "no_search_fallback", False
    )
    search_providers = (
        [item.strip() for item in args.search_providers.split(",") if item.strip()]
        if args.search_providers is not None
        else _config_providers(file_config, "search_providers", DEFAULT_SEARCH_PROVIDERS)
    )
    critic_mode = cast(
        Literal["auto", "rule", "llm"],
        args.critic_mode
        if args.critic_mode is not None
        else _config_str(file_config, "critic_mode", DEFAULT_CRITIC_MODE),
    )
    preset = cast(
        StrategyPreset,
        args.strategy_preset
        if args.strategy_preset is not None
        else _config_str(file_config, "strategy_preset", DEFAULT_STRATEGY_PRESET),
    )

    config = SupervisorConfig(
        output_dir=output_dir,
        use_openai_model=use_openai,
        search_mode=search_mode,
        allow_search_fallback=not no_search_fallback,
        search_providers=search_providers,
        critic_mode=critic_mode,
    )
    config = apply_strategy_preset(config, preset)

    supervisor = build_default_supervisor(config=config)
    outputs = supervisor.run(topic)

    print("Deep Research completed")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def _config_str(config: dict[str, object], key: str, default: str) -> str:
    value = config.get(key)
    if isinstance(value, str):
        return value
    return default


def _config_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key)
    if isinstance(value, bool):
        return value
    return default


def _config_providers(config: dict[str, object], key: str, default: list[str]) -> list[str]:
    value = config.get(key)
    if isinstance(value, list):
        providers = [str(item).strip() for item in value if str(item).strip()]
        if providers:
            return providers
    return list(default)

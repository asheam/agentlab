from __future__ import annotations

import argparse
import json
from typing import Literal, cast

from agentlab.cli_config import CLIConfigError, load_cli_config
from agentlab.multi_agent.supervisor import (
    RunPolicy,
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
DEFAULT_MAX_RETRIES = 0
DEFAULT_AGENT_TIMEOUT_S: float | None = None
DEFAULT_RETRY_BACKOFF_S = 0.0
DEFAULT_RETRY_ON_TIMEOUT_ONLY = False
DEFAULT_CONTINUE_ON_ERROR = False


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
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print merged runtime config (after CLI/config/default precedence) and exit.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum retry attempts per agent (RunPolicy.max_retries).",
    )
    parser.add_argument(
        "--agent-timeout-s",
        type=float,
        help="Soft timeout threshold in seconds per agent (RunPolicy.agent_timeout_s).",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        help="Backoff sleep seconds before retry (RunPolicy.retry_backoff_s).",
    )
    parser.add_argument(
        "--retry-on-timeout-only",
        action="store_true",
        help="Retry only timeout failures, not runtime errors (RunPolicy.retry_on_timeout_only).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline after final agent failure (RunPolicy.continue_on_error).",
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
    max_retries = (
        args.max_retries
        if args.max_retries is not None
        else _config_int(file_config, "max_retries", DEFAULT_MAX_RETRIES)
    )
    agent_timeout_s = (
        args.agent_timeout_s
        if args.agent_timeout_s is not None
        else _config_optional_float(
            file_config,
            "agent_timeout_s",
            DEFAULT_AGENT_TIMEOUT_S,
        )
    )
    retry_backoff_s = (
        args.retry_backoff_s
        if args.retry_backoff_s is not None
        else _config_float(file_config, "retry_backoff_s", DEFAULT_RETRY_BACKOFF_S)
    )
    retry_on_timeout_only = args.retry_on_timeout_only or _config_bool(
        file_config,
        "retry_on_timeout_only",
        DEFAULT_RETRY_ON_TIMEOUT_ONLY,
    )
    continue_on_error = args.continue_on_error or _config_bool(
        file_config,
        "continue_on_error",
        DEFAULT_CONTINUE_ON_ERROR,
    )

    effective_config = {
        "topic": topic,
        "use_openai": use_openai,
        "output_dir": output_dir,
        "search_mode": search_mode,
        "no_search_fallback": no_search_fallback,
        "search_providers": search_providers,
        "critic_mode": critic_mode,
        "strategy_preset": preset,
        "max_retries": max_retries,
        "agent_timeout_s": agent_timeout_s,
        "retry_backoff_s": retry_backoff_s,
        "retry_on_timeout_only": retry_on_timeout_only,
        "continue_on_error": continue_on_error,
    }

    if args.print_effective_config:
        print(json.dumps(effective_config, ensure_ascii=False, indent=2))
        return 0

    try:
        run_policy = RunPolicy(
            max_retries=max_retries,
            agent_timeout_s=agent_timeout_s,
            continue_on_error=continue_on_error,
            retry_backoff_s=retry_backoff_s,
            retry_on_timeout_only=retry_on_timeout_only,
        )
    except ValueError as exc:
        print(f"Config error: {exc}")
        return 2

    config = SupervisorConfig(
        output_dir=output_dir,
        use_openai_model=use_openai,
        search_mode=search_mode,
        allow_search_fallback=not no_search_fallback,
        search_providers=search_providers,
        critic_mode=critic_mode,
        run_policy=run_policy,
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


def _config_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _config_optional_float(
    config: dict[str, object],
    key: str,
    default: float | None,
) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _config_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()

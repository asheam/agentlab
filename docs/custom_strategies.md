# Custom Strategies

This document shows how to replace the default planning and writing logic in AgentLab.

## Why

`SupervisorConfig` supports strategy injection so you can customize behavior without forking runtime or agent orchestration code.

## Interfaces

Planner strategy:

```python
from agentlab.models.base import BaseModel

class MyPlannerStrategy:
    def build_plan(self, topic: str, model: BaseModel | None) -> list[str]:
        ...
```

Writer strategy:

```python
from agentlab.agents.writer_agent import WriterStrategyInput

class MyWriterStrategy:
    def build_report(self, request: WriterStrategyInput) -> str:
        ...
```

## Wiring Through SupervisorConfig

```python
from agentlab.multi_agent.supervisor import SupervisorConfig, build_default_supervisor

config = SupervisorConfig(
    planner_strategy=MyPlannerStrategy(),
    writer_strategy=MyWriterStrategy(),
    search_mode="mock",
    critic_mode="rule",
)

supervisor = build_default_supervisor(config=config)
outputs = supervisor.run("研究 LangGraph、AutoGen、CrewAI 的区别")
```

## Data Available in WriterStrategyInput

`WriterStrategyInput` provides:

- `topic`
- `plan`
- `search_results`
- `notes`
- `critique`
- `model`
- `context`

This makes writer customization straightforward for report format, verbosity, and evidence selection.

## Runnable Example

Use the bundled example:

```bash
uv run python examples/05_custom_strategies.py "研究 LangGraph、AutoGen、CrewAI 的区别"
```

Outputs include `report.md`, `trace.json`, `workspace.json`, and `run_summary.json` under the chosen output directory.

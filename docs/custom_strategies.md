# Custom Strategies

This document shows how to replace default planning, searching, reading, critique, and writing logic in AgentLab.

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

Search strategy:

```python
from agentlab.agents.search_agent import SearchStrategyInput, SearchStrategyOutput

class MySearchStrategy:
    def collect(self, request: SearchStrategyInput) -> SearchStrategyOutput:
        ...
```

Reader strategy:

```python
from agentlab.agents.reader_agent import ReaderStrategyInput
from agentlab.workspace.research_workspace import NotesPayload

class MyReaderStrategy:
    def build_notes(self, request: ReaderStrategyInput) -> NotesPayload:
        ...
```

Critic strategy:

```python
from agentlab.agents.critic_agent import CriticStrategyInput

class MyCriticStrategy:
    def build_critique(self, request: CriticStrategyInput) -> dict[str, object]:
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
    search_strategy=MySearchStrategy(),
    reader_strategy=MyReaderStrategy(),
    critic_strategy=MyCriticStrategy(),
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

This example injects all five strategy slots in one run.

Outputs include `report.md`, `trace.json`, `workspace.json`, and `run_summary.json` under the chosen output directory.

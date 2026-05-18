from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentlab.core.runtime import AgentRuntime
    from agentlab.tools.registry import ToolRegistry
    from agentlab.tracing.recorder import TraceRecorder
    from agentlab.workspace.artifacts import ArtifactStore
    from agentlab.workspace.blackboard import Blackboard


@dataclass
class RuntimeContext:
    runtime: AgentRuntime | None = None
    tool_registry: ToolRegistry | None = None
    blackboard: Blackboard | None = None
    trace_recorder: TraceRecorder | None = None
    artifacts: ArtifactStore | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

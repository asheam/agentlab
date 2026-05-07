from __future__ import annotations

from agentlab.core.agent import Agent
from agentlab.core.runtime import AgentRuntime


class AgentTeam:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self._agents: dict[str, Agent] = {}

    def add(self, agent: Agent) -> None:
        self._agents[agent.name] = agent
        self.runtime.register_agent(agent)

    def add_many(self, agents: list[Agent]) -> None:
        for agent in agents:
            self.add(agent)

    def get(self, name: str) -> Agent:
        return self._agents[name]

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

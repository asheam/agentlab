from __future__ import annotations

from agentlab.core.agent import Agent
from agentlab.core.context import RuntimeContext
from agentlab.core.message import Message
from agentlab.core.runtime import AgentRuntime


class EchoAgent(Agent):
    def run(self, message: Message, context: RuntimeContext) -> Message:
        return Message(
            sender=self.name,
            receiver=message.sender,
            content=f"[{self.role}] {message.content}",
            type="response",
        )


if __name__ == "__main__":
    runtime = AgentRuntime()
    runtime.register_agent(
        EchoAgent(
            name="assistant",
            role="demo",
            system_prompt="Respond by echoing the input.",
        )
    )
    response = runtime.send(
        Message(sender="user", receiver="assistant", content="Hello AgentLab", type="task")
    )
    print(response.content)

from __future__ import annotations

from typing import Any

from .agent_registry import AgentRegistry
from .runtime import AgentRuntime

from .coding_agent import CodingAgent
from .developer_agent import DeveloperAgent
from .debugger_agent import DebuggerAgent


class AgentHub:

    VERSION = "1.0.0"

    def __init__(
        self,
        tool_engine: Any = None,
        planner: Any = None,
        reasoning: Any = None,
        memory: Any = None,
    ) -> None:

        self.runtime = AgentRuntime(
            tool_engine=tool_engine,
            planner=planner,
            reasoning=reasoning,
            memory=memory,
        )

        self.registry = AgentRegistry(
            runtime=self.runtime
        )

        self._register_builtin_agents()

    def _register_builtin_agents(
        self,
    ) -> None:

        self.registry.register(
            CodingAgent(
                runtime=self.runtime
            )
        )

        self.registry.register(
            DeveloperAgent(
                runtime=self.runtime
            )
        )

        self.registry.register(
            DebuggerAgent(
                runtime=self.runtime
            )
        )

    async def execute(
        self,
        agent: str,
        task: str,
        context: dict[str, Any] | None = None,
    ):

        return await self.runtime.execute(
            agent_name=agent,
            task=task,
            context=context,
        )

    def status(self) -> dict[str, Any]:

        return {
            "hub": "AgentHub",
            "version": self.VERSION,
            "registry": self.registry.summary(),
            "status": "READY",
        }
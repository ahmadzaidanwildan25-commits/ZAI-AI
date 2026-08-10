"""
Super ZAI Agent Loop.

Executes plans through ToolEngine and records observations.
"""

from __future__ import annotations

from typing import Any, Dict

from .reasoning import ReasoningEngine


class AgentLoop:

    def __init__(
        self,
        tool_engine: Any,
        reasoning_engine: ReasoningEngine | None = None,
        max_steps: int = 8,
    ) -> None:

        self.tool_engine = tool_engine

        self.reasoning_engine = (
            reasoning_engine
            or ReasoningEngine()
        )

        self.max_steps = max(1, max_steps)

    def execute(
        self,
        context: Any,
    ) -> Any:

        plan = getattr(
            context,
            "plan",
            [],
        )

        executed = 0

        for step in plan:

            if executed >= self.max_steps:
                break

            action = step.get("action")

            if action != "execute_tool":
                continue

            tool = step.get("tool")

            if not tool:
                continue

            arguments = step.get(
                "arguments",
                {},
            )

            query = arguments.get(
                "query",
                context.user_message,
            )

            result = self.tool_engine.execute(
                tool,
                query,
            )

            observation = (
                self.reasoning_engine
                .analyze_tool_result(
                    tool=tool,
                    result=result,
                )
            )

            context.add_observation(
                tool=tool,
                success=observation["success"],
                response=observation["response"],
                data=observation.get("data", {}),
                error=observation.get("error"),
            )

            executed += 1

        context.metadata[
            "agent_steps_executed"
        ] = executed

        return context
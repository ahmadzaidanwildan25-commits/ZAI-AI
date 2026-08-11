from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent
from .agent_result import AgentResult


class DebuggerAgent(BaseAgent):

    name = "debugger_agent"
    version = "1.0.0"

    description = (
        "Agent untuk diagnosis error, "
        "trace analysis, dan perbaikan."
    )

    capabilities = (
        "error_analysis",
        "traceback_analysis",
        "root_cause_analysis",
        "bug_fixing",
    )

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:

        return AgentResult(
            success=True,
            agent=self.name,
            task=task,
            status="ready",
            response=(
                "Debugger Agent menerima "
                "task diagnosis."
            ),
            steps=1,
            observations=[
                {
                    "type": "debug_task",
                    "task": task,
                }
            ],
        )
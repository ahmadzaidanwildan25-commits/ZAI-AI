from __future__ import annotations

from typing import Any

from .agent_result import AgentResult
from .base_agent import BaseAgent


class GeneralAgent(BaseAgent):
    """
    Agent umum ZAI.

    Digunakan sebagai fallback ketika belum tersedia
    agent khusus untuk suatu task.
    """

    name = "general_agent"

    version = "1.1.0"

    description = (
        "General-purpose AI agent "
        "untuk menangani task umum ZAI."
    )

    capabilities = (
        "general_task",
        "text_processing",
        "basic_reasoning",
        "task_response",
    )

    async def run(
        self,
        task: str,
        result: AgentResult,
        **kwargs: Any,
    ) -> AgentResult:

        result.add_observation(
            "general_agent_started",
            agent=self.name,
            task_length=len(task),
        )

        normalized_task = (
            task.strip()
        )

        result.add_observation(
            "task_normalized",
            original_length=len(task),
            normalized_length=len(
                normalized_task
            ),
        )

        if not normalized_task:
            return result.fail(
                "Task tidak boleh kosong.",
                response=(
                    "ZAI menerima task kosong."
                ),
                task_type="general",
            )

        response = (
            f"ZAI menerima task Anda: "
            f"{normalized_task}"
        )

        result.add_observation(
            "response_generated",
            response_length=len(response),
        )

        return result.complete(
            response,
            task_type="general",
            task_length=len(
                normalized_task
            ),
        )
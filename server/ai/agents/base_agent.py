from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from .agent_result import AgentResult


class BaseAgent(ABC):
    """
    Base class seluruh agent ZAI.
    """

    name: str = "base_agent"
    version: str = "1.0.0"
    description: str = "Base ZAI agent."

    capabilities: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

    # ============================================================
    # INFO
    # ============================================================

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "status": "READY",
        }

    # ============================================================
    # EXECUTE
    # ============================================================

    async def execute(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:

        self.execution_count += 1

        started = perf_counter()

        result = AgentResult(
            success=True,
            agent=self.name,
            response="",
            task=task,
            status="running",
        )

        result.set_metadata(
            "agent",
            self.name,
        )

        result.set_metadata(
            "agent_version",
            self.version,
        )

        result.add_observation(
            "task_received",
            task=task,
        )

        try:
            result = await self.run(
                task=task,
                result=result,
                **kwargs,
            )

            if result.success:
                self.success_count += 1
            else:
                self.failure_count += 1

        except Exception as exc:
            self.failure_count += 1

            result.add_error(
                f"{type(exc).__name__}: {exc}"
            )

            result.response = (
                f"{self.name} gagal menjalankan task."
            )

        latency_ms = round(
            (perf_counter() - started) * 1000,
            2,
        )

        result.metadata.update(
            {
                "agent_version": self.version,
                "latency_ms": latency_ms,
            }
        )

        if result.success:
            result.status = "completed"

        return result

    # ============================================================
    # STATISTICS
    # ============================================================

    @property
    def success_rate(self) -> float:
        if self.execution_count == 0:
            return 0.0

        return round(
            (
                self.success_count
                / self.execution_count
            )
            * 100,
            2,
        )

    def statistics(self) -> dict[str, Any]:
        return {
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
        }

    # ============================================================
    # ABSTRACT EXECUTION
    # ============================================================

    @abstractmethod
    async def run(
        self,
        task: str,
        result: AgentResult,
        **kwargs: Any,
    ) -> AgentResult:
        raise NotImplementedError
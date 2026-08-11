from __future__ import annotations

from time import perf_counter
from typing import Any

from .agent_registry import AgentRegistry
from .agent_result import AgentResult
from .agent_router import AgentRouter


class AgentRuntime:
    """
    Runtime utama sistem multi-agent ZAI.

    Bertugas:
    - mengelola AgentRegistry
    - memilih agent
    - menjalankan agent
    - menangani error
    - mengukur latency
    - menyimpan statistik runtime
    - mendukung automatic routing
    """

    VERSION = "2.3.0"

    def __init__(
        self,
        registry: AgentRegistry | None = None,
    ) -> None:

        self.registry = (
            registry
            or AgentRegistry()
        )

        self.router = AgentRouter(
            self.registry
        )

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

    # ============================================================
    # INFO
    # ============================================================

    def info(self) -> dict[str, Any]:

        return {
            "runtime": self.__class__.__name__,
            "version": self.VERSION,
            "status": "READY",
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "router": self.router.info(),
            "registry": self.registry.summary(),
        }

    # ============================================================
    # HEALTH
    # ============================================================

    def health(self) -> dict[str, Any]:

        registry_status = (
            self.registry.summary()
            .get("status", "UNKNOWN")
        )

        runtime_status = (
            "HEALTHY"
            if registry_status == "READY"
            else "DEGRADED"
        )

        return {
            "runtime": self.__class__.__name__,
            "version": self.VERSION,
            "status": runtime_status,
            "registry_status": registry_status,
            "total_agents": self.registry.count,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
        }

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

    # ============================================================
    # REGISTER AGENT
    # ============================================================

    def register_agent(
        self,
        agent: Any,
    ) -> None:

        self.registry.register(agent)

    # ============================================================
    # UNREGISTER AGENT
    # ============================================================

    def unregister_agent(
        self,
        name: str,
    ) -> bool:

        return self.registry.unregister(
            name
        )

    # ============================================================
    # EXECUTE MANUAL
    # ============================================================

    async def execute(
        self,
        agent_name: str,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:

        self.execution_count += 1

        started = perf_counter()

        # --------------------------------------------------------
        # Agent lookup
        # --------------------------------------------------------

        try:
            agent = self.registry.get(
                agent_name
            )

        except KeyError as exc:

            self.failure_count += 1

            result = AgentResult(
                success=False,
                agent=agent_name,
                response="",
                task=task,
                status="failed",
            )

            result.add_error(
                str(exc)
            )

            result.metadata.update(
                {
                    "runtime_version": self.VERSION,
                    "runtime_latency_ms": round(
                        (
                            perf_counter()
                            - started
                        )
                        * 1000,
                        2,
                    ),
                }
            )

            return result

        # --------------------------------------------------------
        # Execute agent
        # --------------------------------------------------------

        try:

            result = await agent.execute(
                task,
                **kwargs,
            )

            if result.success:
                self.success_count += 1
            else:
                self.failure_count += 1

        except Exception as exc:

            self.failure_count += 1

            result = AgentResult(
                success=False,
                agent=agent_name,
                response="",
                task=task,
                status="failed",
            )

            result.add_error(
                f"{type(exc).__name__}: {exc}"
            )

            result.response = (
                f"Runtime gagal menjalankan "
                f"agent '{agent_name}'."
            )

        # --------------------------------------------------------
        # Runtime metrics
        # --------------------------------------------------------

        runtime_latency_ms = round(
            (
                perf_counter()
                - started
            )
            * 1000,
            2,
        )

        result.metadata.update(
            {
                "runtime_version": self.VERSION,
                "runtime_latency_ms": (
                    runtime_latency_ms
                ),
                "runtime_execution_count": (
                    self.execution_count
                ),
                "runtime_success_count": (
                    self.success_count
                ),
                "runtime_failure_count": (
                    self.failure_count
                ),
                "runtime_success_rate": (
                    self.success_rate
                ),
            }
        )

        return result

    # ============================================================
    # EXECUTE AUTOMATIC
    # ============================================================

    async def execute_auto(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:

        routing = (
            self.router.route_with_details(
                task
            )
        )

        agent_name = routing[
            "selected_agent"
        ]

        result = await self.execute(
            agent_name,
            task,
            **kwargs,
        )

        result.metadata.update(
            {
                "routing_mode": "automatic",
                "selected_agent": agent_name,
                "router_version": (
                    self.router.VERSION
                ),
                "routing_fallback": (
                    routing["fallback"]
                ),
            }
        )

        result.add_observation(
            "automatic_routing",
            selected_agent=agent_name,
            fallback=routing["fallback"],
        )

        return result

    # ============================================================
    # RESET STATISTICS
    # ============================================================

    def reset_statistics(self) -> None:

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

        self.router.reset_statistics()

    # ============================================================
    # REPRESENTATION
    # ============================================================

    def __repr__(self) -> str:

        return (
            f"AgentRuntime("
            f"version='{self.VERSION}', "
            f"agents={self.registry.names()}"
            f")"
        )
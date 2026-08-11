from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable, Sequence
from uuid import uuid4

from .agent_result import AgentResult
from .base_agent import BaseAgent
from .agent_registry import AgentRegistry
from .agent_router import AgentRouter
from .runtime import AgentRuntime
from .orchestrator import AgentOrchestrator


@dataclass(slots=True)
class AgentExecutionRecord:
    """
    Record historis setiap eksekusi agent.

    Class ini sengaja dibuat terpisah dari AgentResult agar:
    - AgentResult fokus pada hasil task.
    - ExecutionRecord fokus pada lifecycle manager.
    """

    execution_id: str
    task: str
    requested_agent: str | None
    selected_agent: str | None
    success: bool
    status: str
    started_at: str
    completed_at: str | None
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task": self.task,
            "requested_agent": self.requested_agent,
            "selected_agent": self.selected_agent,
            "success": self.success,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AgentManagerConfig:
    """
    Configuration untuk AgentManager.

    Default dibuat konservatif agar aman untuk local development.
    """

    version: str = "1.0.0"

    max_history: int = 500

    default_agent: str = "general_agent"

    enable_history: bool = True

    enable_statistics: bool = True

    auto_register: bool = True

    allow_duplicate_registration: bool = False

    allow_unhealthy_agents: bool = False

    execution_timeout_seconds: float | None = None

    batch_concurrency: int = 4

    strict_agent_validation: bool = True

    def __post_init__(self) -> None:
        if self.max_history < 1:
            raise ValueError(
                "max_history harus lebih besar dari 0."
            )

        if self.batch_concurrency < 1:
            raise ValueError(
                "batch_concurrency harus lebih besar dari 0."
            )

        if self.execution_timeout_seconds is not None:
            if self.execution_timeout_seconds <= 0:
                raise ValueError(
                    "execution_timeout_seconds harus lebih besar dari 0."
                )


class AgentManager:
    """
    High-level manager untuk seluruh agent ZAI.

    Tanggung jawab utama:

    1. Registration agent.
    2. Unregistration agent.
    3. Discovery agent.
    4. Health monitoring.
    5. Single execution.
    6. Batch execution.
    7. History.
    8. Statistics.
    9. Agent lifecycle.
    10. Runtime integration.
    11. Orchestrator integration.
    12. Safe error handling.

    Arsitektur:

        AgentManager
             |
             +---- AgentRegistry
             |
             +---- AgentRouter
             |
             +---- AgentRuntime
             |
             +---- AgentOrchestrator
             |
             +---- Agents
                    |
                    +---- GeneralAgent
                    +---- CodingAgent
                    +---- ResearchAgent
                    +---- SystemAgent
                    +---- future agents
    """

    VERSION = "1.0.0"

    NAME = "AgentManager"

    STATUS_READY = "READY"

    STATUS_HEALTHY = "HEALTHY"

    STATUS_DEGRADED = "DEGRADED"

    STATUS_ERROR = "ERROR"

    def __init__(
        self,
        *,
        registry: AgentRegistry | None = None,
        router: AgentRouter | None = None,
        runtime: AgentRuntime | None = None,
        orchestrator: AgentOrchestrator | None = None,
        config: AgentManagerConfig | None = None,
    ) -> None:

        self.config = config or AgentManagerConfig()

        self.registry = registry or AgentRegistry()

        self.router = router

        self.runtime = runtime or AgentRuntime()

        self.orchestrator = (
            orchestrator
            or AgentOrchestrator()
        )

        self.created_at = self._now()

        self._history: list[AgentExecutionRecord] = []

        self._execution_count = 0

        self._success_count = 0

        self._failure_count = 0

        self._timeout_count = 0

        self._registration_count = 0

        self._unregistration_count = 0

        self._manager_id = str(uuid4())

        self._started = True

        self._last_execution_id: str | None = None

        self._last_error: str | None = None

        self._agent_locks: dict[str, asyncio.Lock] = {}

    # ============================================================
    # INTERNAL UTILITIES
    # ============================================================

    @staticmethod
    def _now() -> str:
        """
        Return UTC timestamp ISO-8601.
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _normalize_task(task: str) -> str:
        """
        Normalize task input.
        """

        if not isinstance(task, str):
            raise TypeError(
                "Task harus berupa string."
            )

        normalized = task.strip()

        if not normalized:
            raise ValueError(
                "Task tidak boleh kosong."
            )

        return normalized

    def _record_history(
        self,
        record: AgentExecutionRecord,
    ) -> None:

        if not self.config.enable_history:
            return

        self._history.append(record)

        if len(self._history) > self.config.max_history:
            overflow = (
                len(self._history)
                - self.config.max_history
            )

            del self._history[:overflow]

    def _calculate_success_rate(self) -> float:
        if self._execution_count == 0:
            return 0.0

        return round(
            (
                self._success_count
                / self._execution_count
            )
            * 100,
            2,
        )

    def _get_agent_lock(
        self,
        agent_name: str,
    ) -> asyncio.Lock:

        lock = self._agent_locks.get(
            agent_name
        )

        if lock is None:
            lock = asyncio.Lock()

            self._agent_locks[
                agent_name
            ] = lock

        return lock

    # ============================================================
    # AGENT REGISTRATION
    # ============================================================

    def register_agent(
        self,
        agent: BaseAgent,
    ) -> dict[str, Any]:

        if not isinstance(agent, BaseAgent):
            raise TypeError(
                "Agent harus merupakan turunan BaseAgent."
            )

        if not agent.name:
            raise ValueError(
                "Agent harus memiliki name."
            )

        if (
            self.registry.has(agent.name)
            and not self.config.allow_duplicate_registration
        ):
            raise ValueError(
                f"Agent '{agent.name}' sudah terdaftar."
            )

        self.registry.register(agent)

        self._registration_count += 1

        self._get_agent_lock(agent.name)

        if self.config.auto_register:
            try:
                self.orchestrator.register_agent(
                    agent
                )
            except Exception:
                # Orchestrator integration tidak boleh
                # menggagalkan registry registration.
                pass

        return {
            "success": True,
            "agent": agent.name,
            "version": agent.version,
            "status": "REGISTERED",
            "timestamp": self._now(),
        }

    def register_agents(
        self,
        agents: Iterable[BaseAgent],
    ) -> list[dict[str, Any]]:

        results: list[dict[str, Any]] = []

        for agent in agents:
            try:
                results.append(
                    self.register_agent(agent)
                )
            except Exception as exc:
                results.append(
                    {
                        "success": False,
                        "agent": getattr(
                            agent,
                            "name",
                            None,
                        ),
                        "status": "FAILED",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "timestamp": self._now(),
                    }
                )

        return results

    def unregister_agent(
        self,
        name: str,
    ) -> bool:

        if not isinstance(name, str):
            raise TypeError(
                "Nama agent harus berupa string."
            )

        if not self.registry.has(name):
            return False

        agents = getattr(
            self.registry,
            "_agents",
            None,
        )

        if isinstance(agents, dict):
            agents.pop(name, None)

        self._agent_locks.pop(
            name,
            None,
        )

        self._unregistration_count += 1

        return True

    # ============================================================
    # AGENT DISCOVERY
    # ============================================================

    def has_agent(
        self,
        name: str,
    ) -> bool:

        return self.registry.has(name)

    def get_agent(
        self,
        name: str,
    ) -> BaseAgent:

        return self.registry.get(name)

    def list_agents(
        self,
    ) -> list[dict[str, Any]]:

        return self.registry.active()

    def agent_names(
        self,
    ) -> list[str]:

        return self.registry.names()

    def count_agents(
        self,
    ) -> int:

        return len(
            self.registry.names()
        )

    def find_agents_by_capability(
        self,
        capability: str,
    ) -> list[dict[str, Any]]:

        if not isinstance(
            capability,
            str,
        ):
            raise TypeError(
                "Capability harus berupa string."
            )

        target = capability.strip().lower()

        if not target:
            return []

        results: list[dict[str, Any]] = []

        for agent in self.registry._agents.values():

            capabilities = {
                str(item).lower()
                for item in agent.capabilities
            }

            if target in capabilities:
                results.append(
                    agent.info()
                )

        return results

    # ============================================================
    # HEALTH
    # ============================================================

    def health(
        self,
    ) -> dict[str, Any]:

        unhealthy: list[str] = []

        agent_health: list[dict[str, Any]] = []

        for name in self.agent_names():

            try:
                agent = self.get_agent(name)

                if hasattr(agent, "health"):
                    health_data = agent.health()
                else:
                    health_data = {
                        "agent": name,
                        "status": "HEALTHY",
                    }

                status = str(
                    health_data.get(
                        "status",
                        "UNKNOWN",
                    )
                ).upper()

                if status not in {
                    "HEALTHY",
                    "READY",
                }:
                    unhealthy.append(name)

                agent_health.append(
                    health_data
                )

            except Exception as exc:
                unhealthy.append(name)

                agent_health.append(
                    {
                        "agent": name,
                        "status": "ERROR",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

        if not unhealthy:
            manager_status = self.STATUS_HEALTHY
        elif len(unhealthy) < self.count_agents():
            manager_status = self.STATUS_DEGRADED
        else:
            manager_status = self.STATUS_ERROR

        return {
            "manager": self.NAME,
            "version": self.VERSION,
            "manager_id": self._manager_id,
            "status": manager_status,
            "registered_agents": self.count_agents(),
            "unhealthy_agents": unhealthy,
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "timeout_count": self._timeout_count,
            "success_rate": self._calculate_success_rate(),
            "agents": agent_health,
            "timestamp": self._now(),
        }

    # ============================================================
    # INFO
    # ============================================================

    def info(
        self,
    ) -> dict[str, Any]:

        return {
            "manager": self.NAME,
            "version": self.VERSION,
            "manager_id": self._manager_id,
            "status": self.STATUS_READY,
            "created_at": self.created_at,
            "started": self._started,
            "default_agent": self.config.default_agent,
            "registered_agents": self.count_agents(),
            "agent_names": self.agent_names(),
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "timeout_count": self._timeout_count,
            "success_rate": self._calculate_success_rate(),
            "history_size": len(self._history),
            "registration_count": self._registration_count,
            "unregistration_count": self._unregistration_count,
            "runtime": (
                self.runtime.info()
                if hasattr(
                    self.runtime,
                    "info",
                )
                else None
            ),
            "registry": self.registry.summary(),
            "config": {
                "max_history": self.config.max_history,
                "enable_history": (
                    self.config.enable_history
                ),
                "enable_statistics": (
                    self.config.enable_statistics
                ),
                "default_agent": (
                    self.config.default_agent
                ),
                "batch_concurrency": (
                    self.config.batch_concurrency
                ),
            },
        }

    # ============================================================
    # EXECUTION
    # ============================================================

    async def execute(
        self,
        task: str,
        *,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:

        normalized_task = self._normalize_task(task)

        requested_agent = agent_name

        started = perf_counter()

        started_at = self._now()

        execution_id = str(uuid4())

        self._last_execution_id = execution_id

        selected_agent_name = agent_name

        result: AgentResult | None = None

        try:

            if agent_name is not None:

                if not self.has_agent(agent_name):
                    raise KeyError(
                        f"Agent '{agent_name}' tidak terdaftar."
                    )

                selected_agent_name = agent_name

            else:

                if (
                    self.config.default_agent
                    and self.has_agent(
                        self.config.default_agent
                    )
                ):
                    selected_agent_name = (
                        self.config.default_agent
                    )

            if selected_agent_name is None:

                names = self.agent_names()

                if not names:
                    raise RuntimeError(
                        "Tidak ada agent yang terdaftar."
                    )

                selected_agent_name = names[0]

            agent = self.get_agent(
                selected_agent_name
            )

            if (
                not self.config.allow_unhealthy_agents
                and hasattr(agent, "health")
            ):

                health_data = agent.health()

                status = str(
                    health_data.get(
                        "status",
                        "UNKNOWN",
                    )
                ).upper()

                if status in {
                    "ERROR",
                    "UNHEALTHY",
                }:
                    raise RuntimeError(
                        f"Agent '{agent.name}' tidak sehat."
                    )

            lock = self._get_agent_lock(
                selected_agent_name
            )

            async with lock:

                if self.config.execution_timeout_seconds:
                    result = await asyncio.wait_for(
                        self.runtime.execute(
                            selected_agent_name,
                            normalized_task,
                            **kwargs,
                        ),
                        timeout=(
                            self.config
                            .execution_timeout_seconds
                        ),
                    )
                else:
                    result = await self.runtime.execute(
                        selected_agent_name,
                        normalized_task,
                        **kwargs,
                    )

            if result is None:
                raise RuntimeError(
                    "Runtime mengembalikan result kosong."
                )

            self._execution_count += 1

            if result.success:
                self._success_count += 1
            else:
                self._failure_count += 1

            completed_at = self._now()

            latency_ms = round(
                (
                    perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            result.metadata.update(
                {
                    "manager_version": self.VERSION,
                    "manager_execution_id": execution_id,
                    "manager_latency_ms": latency_ms,
                }
            )

            record = AgentExecutionRecord(
                execution_id=execution_id,
                task=normalized_task,
                requested_agent=requested_agent,
                selected_agent=selected_agent_name,
                success=result.success,
                status=result.status,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                metadata={
                    "agent": selected_agent_name,
                },
            )

            self._record_history(record)

            return result

        except asyncio.TimeoutError:

            self._execution_count += 1

            self._failure_count += 1

            self._timeout_count += 1

            self._last_error = (
                "Agent execution timeout."
            )

            latency_ms = round(
                (
                    perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            fallback_agent = (
                selected_agent_name
                or requested_agent
                or "unknown"
            )

            result = AgentResult(
                success=False,
                agent=fallback_agent,
                response=(
                    "Task dihentikan karena "
                    "melewati execution timeout."
                ),
                task=normalized_task,
                status="failed",
            )

            result.add_error(
                "Agent execution timeout."
            )

            result.metadata.update(
                {
                    "manager_version": self.VERSION,
                    "manager_execution_id": execution_id,
                    "manager_latency_ms": latency_ms,
                    "timeout": True,
                }
            )

            self._record_history(
                AgentExecutionRecord(
                    execution_id=execution_id,
                    task=normalized_task,
                    requested_agent=requested_agent,
                    selected_agent=selected_agent_name,
                    success=False,
                    status="failed",
                    started_at=started_at,
                    completed_at=self._now(),
                    latency_ms=latency_ms,
                    error="timeout",
                )
            )

            return result

        except Exception as exc:

            self._execution_count += 1

            self._failure_count += 1

            self._last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            latency_ms = round(
                (
                    perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            fallback_agent = (
                selected_agent_name
                or requested_agent
                or "manager"
            )

            result = AgentResult(
                success=False,
                agent=fallback_agent,
                response=(
                    "AgentManager gagal "
                    "menjalankan task."
                ),
                task=normalized_task,
                status="failed",
            )

            result.add_error(
                f"{type(exc).__name__}: {exc}"
            )

            result.metadata.update(
                {
                    "manager_version": self.VERSION,
                    "manager_execution_id": execution_id,
                    "manager_latency_ms": latency_ms,
                }
            )

            self._record_history(
                AgentExecutionRecord(
                    execution_id=execution_id,
                    task=normalized_task,
                    requested_agent=requested_agent,
                    selected_agent=selected_agent_name,
                    success=False,
                    status="failed",
                    started_at=started_at,
                    completed_at=self._now(),
                    latency_ms=latency_ms,
                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

            return result

    # ============================================================
    # BATCH EXECUTION
    # ============================================================

    async def execute_batch(
        self,
        tasks: Sequence[str],
        *,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> list[AgentResult]:

        if not isinstance(
            tasks,
            Sequence,
        ):
            raise TypeError(
                "tasks harus berupa sequence."
            )

        if not tasks:
            return []

        semaphore = asyncio.Semaphore(
            self.config.batch_concurrency
        )

        async def run_one(
            task: str,
        ) -> AgentResult:

            async with semaphore:

                return await self.execute(
                    task,
                    agent_name=agent_name,
                    **kwargs,
                )

        return await asyncio.gather(
            *[
                run_one(task)
                for task in tasks
            ]
        )

    # ============================================================
    # MULTI AGENT EXECUTION
    # ============================================================

    async def execute_multi_agent(
        self,
        task: str,
        agent_names: Sequence[str],
        **kwargs: Any,
    ) -> list[AgentResult]:

        normalized_task = self._normalize_task(
            task
        )

        if not agent_names:
            raise ValueError(
                "agent_names tidak boleh kosong."
            )

        unique_names = list(
            dict.fromkeys(agent_names)
        )

        for name in unique_names:

            if not self.has_agent(name):
                raise KeyError(
                    f"Agent '{name}' tidak terdaftar."
                )

        return await self.execute_batch(
            [normalized_task] * len(unique_names),
            **kwargs,
        )

    # ============================================================
    # HISTORY
    # ============================================================

    def history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        records = self._history

        if limit is not None:

            if limit < 0:
                raise ValueError(
                    "limit tidak boleh negatif."
                )

            records = records[-limit:]

        return [
            record.to_dict()
            for record in records
        ]

    def clear_history(
        self,
    ) -> int:

        count = len(
            self._history
        )

        self._history.clear()

        return count

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(
        self,
    ) -> dict[str, Any]:

        agent_counts: dict[str, int] = {}

        agent_success: dict[str, int] = {}

        agent_failures: dict[str, int] = {}

        for record in self._history:

            name = (
                record.selected_agent
                or "unknown"
            )

            agent_counts[name] = (
                agent_counts.get(name, 0)
                + 1
            )

            if record.success:
                agent_success[name] = (
                    agent_success.get(name, 0)
                    + 1
                )
            else:
                agent_failures[name] = (
                    agent_failures.get(name, 0)
                    + 1
                )

        return {
            "manager": self.NAME,
            "version": self.VERSION,
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "timeout_count": self._timeout_count,
            "success_rate": self._calculate_success_rate(),
            "history_size": len(self._history),
            "agent_execution_counts": agent_counts,
            "agent_success_counts": agent_success,
            "agent_failure_counts": agent_failures,
            "last_execution_id": (
                self._last_execution_id
            ),
            "last_error": self._last_error,
        }

    def reset_statistics(
        self,
    ) -> None:

        self._execution_count = 0

        self._success_count = 0

        self._failure_count = 0

        self._timeout_count = 0

        self._last_execution_id = None

        self._last_error = None

    # ============================================================
    # SEARCH / DISCOVERY
    # ============================================================

    def search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query harus berupa string."
            )

        normalized = query.strip().lower()

        if not normalized:
            return []

        results: list[dict[str, Any]] = []

        for agent in self.registry._agents.values():

            score = 0.0

            reasons: list[str] = []

            name = agent.name.lower()

            description = (
                agent.description.lower()
            )

            capabilities = [
                str(cap).lower()
                for cap in agent.capabilities
            ]

            if normalized in name:
                score += 1.0
                reasons.append(
                    "name_match"
                )

            if normalized in description:
                score += 0.75
                reasons.append(
                    "description_match"
                )

            for capability in capabilities:

                if normalized in capability:
                    score += 0.5
                    reasons.append(
                        "capability_match"
                    )

            if score > 0:

                results.append(
                    {
                        "agent": agent.name,
                        "score": round(
                            score,
                            4,
                        ),
                        "reasons": reasons,
                        "info": agent.info(),
                    }
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results

    # ============================================================
    # LIFECYCLE
    # ============================================================

    def start(
        self,
    ) -> dict[str, Any]:

        self._started = True

        return {
            "success": True,
            "status": "STARTED",
            "manager": self.NAME,
            "timestamp": self._now(),
        }

    def stop(
        self,
    ) -> dict[str, Any]:

        self._started = False

        return {
            "success": True,
            "status": "STOPPED",
            "manager": self.NAME,
            "timestamp": self._now(),
        }

    def restart(
        self,
    ) -> dict[str, Any]:

        self.stop()

        self.start()

        return {
            "success": True,
            "status": "RESTARTED",
            "manager": self.NAME,
            "timestamp": self._now(),
        }

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate(
        self,
    ) -> dict[str, Any]:

        errors: list[str] = []

        warnings: list[str] = []

        if not self._started:
            warnings.append(
                "AgentManager belum aktif."
            )

        if self.count_agents() == 0:
            warnings.append(
                "Belum ada agent yang terdaftar."
            )

        try:
            self.registry.summary()
        except Exception as exc:
            errors.append(
                f"Registry error: {exc}"
            )

        try:
            self.runtime.info()
        except Exception as exc:
            errors.append(
                f"Runtime error: {exc}"
            )

        status = (
            "VALID"
            if not errors
            else "INVALID"
        )

        return {
            "valid": not errors,
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "agent_count": self.count_agents(),
            "timestamp": self._now(),
        }

    # ============================================================
    # SNAPSHOT
    # ============================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:

        return {
            "manager": self.NAME,
            "version": self.VERSION,
            "manager_id": self._manager_id,
            "timestamp": self._now(),
            "info": self.info(),
            "health": self.health(),
            "statistics": self.statistics(),
            "history": self.history(
                limit=20
            ),
        }

    # ============================================================
    # STRING REPRESENTATION
    # ============================================================

    def __repr__(self) -> str:

        return (
            f"<{self.NAME} "
            f"version={self.VERSION!r} "
            f"agents={self.count_agents()} "
            f"executions={self._execution_count}>"
        )


__all__ = [
    "AgentExecutionRecord",
    "AgentManagerConfig",
    "AgentManager",
]
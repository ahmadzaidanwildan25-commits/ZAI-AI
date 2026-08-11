from __future__ import annotations

"""
ZAI Agent Orchestrator
======================

Orchestrator pusat untuk sistem multi-agent ZAI.

Tanggung jawab utama:

1. Registrasi agent
2. Routing task
3. Eksekusi single-agent
4. Eksekusi multi-agent
5. Sequential execution
6. Parallel execution
7. Retry
8. Timeout
9. Fallback
10. Context propagation
11. Execution history
12. Health monitoring
13. Runtime statistics
14. Error isolation
15. Result normalization
16. Agent capability discovery

Arsitektur:

    User Task
        |
        v
    AgentOrchestrator
        |
        +---- AgentRegistry
        |
        +---- AgentRouter
        |
        +---- AgentRuntime
        |
        +---- Agent(s)
        |
        v
    AgentResult

File ini sengaja dibuat tidak bergantung pada FastAPI,
database, HTTP client, atau library eksternal.

Dengan demikian orchestrator dapat digunakan untuk:

- unit test
- CLI
- FastAPI
- background worker
- desktop ZAI
- mobile bridge
- future distributed agent architecture
"""

import asyncio
import inspect
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .agent_registry import AgentRegistry
from .agent_result import AgentResult
from .agent_router import AgentRouter
from .base_agent import BaseAgent


# ============================================================
# CONSTANTS
# ============================================================

ORCHESTRATOR_NAME = "AgentOrchestrator"
ORCHESTRATOR_VERSION = "2.0.0"

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRY_COUNT = 0
DEFAULT_HISTORY_SIZE = 500

MAX_TIMEOUT_SECONDS = 600.0
MAX_RETRY_COUNT = 5


# ============================================================
# HELPERS
# ============================================================


def utc_now() -> str:
    """
    Return current UTC timestamp in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def clamp_int(
    value: Any,
    minimum: int,
    maximum: int,
) -> int:
    """
    Safely clamp integer values.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = minimum

    return max(minimum, min(number, maximum))


def clamp_float(
    value: Any,
    minimum: float,
    maximum: float,
) -> float:
    """
    Safely clamp floating-point values.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum

    return max(minimum, min(number, maximum))


def safe_task_text(task: Any) -> str:
    """
    Normalize task into a safe string.
    """
    if task is None:
        return ""

    if isinstance(task, str):
        return task.strip()

    return str(task).strip()


# ============================================================
# ORCHESTRATION REQUEST
# ============================================================


@dataclass
class OrchestrationRequest:
    """
    Configuration untuk satu orchestration request.
    """

    task: str

    agent_name: str | None = None

    mode: str = "auto"

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    retry_count: int = DEFAULT_RETRY_COUNT

    allow_fallback: bool = True

    include_history: bool = False

    context: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.task = safe_task_text(self.task)

        self.timeout_seconds = clamp_float(
            self.timeout_seconds,
            0.1,
            MAX_TIMEOUT_SECONDS,
        )

        self.retry_count = clamp_int(
            self.retry_count,
            0,
            MAX_RETRY_COUNT,
        )

        if not self.mode:
            self.mode = "auto"

        self.mode = self.mode.lower().strip()


# ============================================================
# ORCHESTRATION RECORD
# ============================================================


@dataclass
class OrchestrationRecord:
    """
    Record internal untuk history orchestration.
    """

    execution_id: str

    task: str

    mode: str

    status: str

    selected_agent: str | None

    started_at: str

    completed_at: str | None = None

    latency_ms: float = 0.0

    success: bool = False

    fallback_used: bool = False

    retry_count: int = 0

    agent_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task": self.task,
            "mode": self.mode,
            "status": self.status,
            "selected_agent": self.selected_agent,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "fallback_used": self.fallback_used,
            "retry_count": self.retry_count,
            "agent_count": self.agent_count,
            "metadata": dict(self.metadata),
        }


# ============================================================
# AGENT ORCHESTRATOR
# ============================================================


class AgentOrchestrator:
    """
    Central multi-agent coordinator untuk ZAI.

    Contoh:

        orchestrator = AgentOrchestrator()

        orchestrator.register_agent(
            GeneralAgent()
        )

        result = await orchestrator.execute(
            "Halo ZAI"
        )

    Or explicit agent:

        result = await orchestrator.execute(
            "Analisis kode ini",
            agent_name="coding_agent",
        )
    """

    NAME = ORCHESTRATOR_NAME
    VERSION = ORCHESTRATOR_VERSION

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        router: AgentRouter | None = None,
        runtime: Any | None = None,
        history_size: int = DEFAULT_HISTORY_SIZE,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        default_retry_count: int = DEFAULT_RETRY_COUNT,
    ) -> None:

        self.registry = (
            registry
            if registry is not None
            else AgentRegistry()
        )

        self.router = router

        self.runtime = runtime

        self.history_size = clamp_int(
            history_size,
            1,
            10_000,
        )

        self.default_timeout = clamp_float(
            default_timeout,
            0.1,
            MAX_TIMEOUT_SECONDS,
        )

        self.default_retry_count = clamp_int(
            default_retry_count,
            0,
            MAX_RETRY_COUNT,
        )

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.fallback_count = 0
        self.retry_count = 0
        self.timeout_count = 0

        self.started_at = utc_now()

        self._history: deque[
            OrchestrationRecord
        ] = deque(
            maxlen=self.history_size
        )

        self._lock = asyncio.Lock()

    # ========================================================
    # AGENT REGISTRATION
    # ========================================================

    def register_agent(
        self,
        agent: BaseAgent,
    ) -> None:
        """
        Register one agent.
        """

        if not isinstance(agent, BaseAgent):
            raise TypeError(
                "Agent harus merupakan turunan BaseAgent."
            )

        self.registry.register(agent)

    def register_agents(
        self,
        agents: Iterable[BaseAgent],
    ) -> None:
        """
        Register multiple agents.
        """

        for agent in agents:
            self.register_agent(agent)

    def unregister_agent(
        self,
        name: str,
    ) -> bool:
        """
        Remove agent jika registry mendukung operasi remove.

        Return False jika registry versi sekarang tidak
        menyediakan remove.
        """

        remove_method = getattr(
            self.registry,
            "unregister",
            None,
        )

        if callable(remove_method):
            remove_method(name)
            return True

        remove_method = getattr(
            self.registry,
            "remove",
            None,
        )

        if callable(remove_method):
            remove_method(name)
            return True

        return False

    # ========================================================
    # AGENT DISCOVERY
    # ========================================================

    def get_agent(
        self,
        name: str,
    ) -> BaseAgent:
        return self.registry.get(name)

    def has_agent(
        self,
        name: str,
    ) -> bool:
        return self.registry.has(name)

    def agent_names(self) -> list[str]:
        return self.registry.names()

    def agents(self) -> list[BaseAgent]:
        result: list[BaseAgent] = []

        for name in self.agent_names():
            try:
                result.append(
                    self.registry.get(name)
                )
            except KeyError:
                continue

        return result

    # ========================================================
    # ROUTER INITIALIZATION
    # ========================================================

    def _ensure_router(self) -> AgentRouter:
        """
        Lazy router initialization.

        Kompatibel dengan beberapa versi AgentRouter.
        """

        if self.router is not None:
            return self.router

        try:
            self.router = AgentRouter(
                self.registry
            )
        except TypeError:
            try:
                self.router = AgentRouter(
                    registry=self.registry
                )
            except TypeError:
                self.router = AgentRouter()

        return self.router

    # ========================================================
    # RUNTIME INITIALIZATION
    # ========================================================

    def _get_runtime(self) -> Any | None:
        """
        Return runtime jika tersedia.
        """

        return self.runtime

    # ========================================================
    # RESULT HELPERS
    # ========================================================

    def _create_failure_result(
        self,
        task: str,
        agent_name: str,
        message: str,
        execution_id: str | None = None,
    ) -> AgentResult:

        result = AgentResult(
            success=False,
            agent=agent_name,
            response=message,
            task=task,
            status="failed",
        )

        if execution_id:
            result.set_metadata(
                "orchestration_execution_id",
                execution_id,
            )

        result.add_error(message)

        return result

    def _attach_orchestration_metadata(
        self,
        result: AgentResult,
        execution_id: str,
        latency_ms: float,
        mode: str,
        fallback_used: bool,
        retry_count: int,
    ) -> AgentResult:

        result.set_metadata(
            "orchestration_execution_id",
            execution_id,
        )

        result.set_metadata(
            "orchestrator_version",
            self.VERSION,
        )

        result.set_metadata(
            "orchestration_mode",
            mode,
        )

        result.set_metadata(
            "orchestration_latency_ms",
            latency_ms,
        )

        result.set_metadata(
            "orchestration_fallback_used",
            fallback_used,
        )

        result.set_metadata(
            "orchestration_retry_count",
            retry_count,
        )

        return result

    # ========================================================
    # EXECUTION
    # ========================================================

    async def execute(
        self,
        task: str,
        agent_name: str | None = None,
        *,
        mode: str = "auto",
        timeout_seconds: float | None = None,
        retry_count: int | None = None,
        allow_fallback: bool = True,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:

        task = safe_task_text(task)

        execution_id = str(
            uuid.uuid4()
        )

        started = time.perf_counter()

        self.execution_count += 1

        timeout = (
            self.default_timeout
            if timeout_seconds is None
            else clamp_float(
                timeout_seconds,
                0.1,
                MAX_TIMEOUT_SECONDS,
            )
        )

        retries = (
            self.default_retry_count
            if retry_count is None
            else clamp_int(
                retry_count,
                0,
                MAX_RETRY_COUNT,
            )
        )

        normalized_mode = (
            mode.lower().strip()
            if mode
            else "auto"
        )

        record = OrchestrationRecord(
            execution_id=execution_id,
            task=task,
            mode=normalized_mode,
            status="running",
            selected_agent=agent_name,
            started_at=utc_now(),
        )

        try:
            if not task:
                result = self._create_failure_result(
                    task=task,
                    agent_name=(
                        agent_name
                        or "orchestrator"
                    ),
                    message=(
                        "Task ZAI tidak boleh kosong."
                    ),
                    execution_id=execution_id,
                )

                return await self._finish_execution(
                    result,
                    record,
                    started,
                )

            if normalized_mode in {
                "multi",
                "parallel",
            }:
                result = await self.execute_parallel(
                    task,
                    timeout_seconds=timeout,
                    retry_count=retries,
                    allow_fallback=allow_fallback,
                    context=context,
                    metadata=metadata,
                    **kwargs,
                )

                record.agent_count = (
                    len(self.agent_names())
                )

                return await self._finish_execution(
                    result,
                    record,
                    started,
                )

            if normalized_mode in {
                "sequential",
                "pipeline",
            }:
                result = await self.execute_sequential(
                    task,
                    timeout_seconds=timeout,
                    retry_count=retries,
                    allow_fallback=allow_fallback,
                    context=context,
                    metadata=metadata,
                    **kwargs,
                )

                record.agent_count = (
                    len(self.agent_names())
                )

                return await self._finish_execution(
                    result,
                    record,
                    started,
                )

            if agent_name:
                result = await self._execute_named_agent(
                    agent_name=agent_name,
                    task=task,
                    timeout_seconds=timeout,
                    retry_count=retries,
                    context=context,
                    metadata=metadata,
                    execution_id=execution_id,
                    **kwargs,
                )

                record.selected_agent = (
                    agent_name
                )

            else:
                result = await self._execute_routed(
                    task=task,
                    timeout_seconds=timeout,
                    retry_count=retries,
                    allow_fallback=allow_fallback,
                    context=context,
                    metadata=metadata,
                    execution_id=execution_id,
                    **kwargs,
                )

                record.selected_agent = (
                    result.agent
                )

            return await self._finish_execution(
                result,
                record,
                started,
            )

        except Exception as exc:

            self.failure_count += 1

            result = self._create_failure_result(
                task=task,
                agent_name=(
                    agent_name
                    or "orchestrator"
                ),
                message=(
                    "Orchestration gagal: "
                    f"{type(exc).__name__}: {exc}"
                ),
                execution_id=execution_id,
            )

            return await self._finish_execution(
                result,
                record,
                started,
            )

    # ========================================================
    # ROUTED EXECUTION
    # ========================================================

    async def _execute_routed(
        self,
        task: str,
        timeout_seconds: float,
        retry_count: int,
        allow_fallback: bool,
        context: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        execution_id: str,
        **kwargs: Any,
    ) -> AgentResult:

        agents = self.agents()

        if not agents:
            return self._create_failure_result(
                task=task,
                agent_name="orchestrator",
                message=(
                    "Tidak ada agent terdaftar "
                    "di AgentRegistry."
                ),
                execution_id=execution_id,
            )

        router = self._ensure_router()

        route_result = None

        try:
            route_method = getattr(
                router,
                "route",
                None,
            )

            if callable(route_method):
                route_result = route_method(
                    task,
                    agents,
                )

        except Exception:
            route_result = None

        selected_name = None

        if route_result is not None:
            selected_name = getattr(
                route_result,
                "selected_agent",
                None,
            )

            if selected_name is None:
                try:
                    route_dict = route_result.to_dict()

                    selected_name = route_dict.get(
                        "selected_agent"
                    )
                except Exception:
                    selected_name = None

        if selected_name:
            try:
                result = await self._execute_named_agent(
                    agent_name=selected_name,
                    task=task,
                    timeout_seconds=timeout_seconds,
                    retry_count=retry_count,
                    context=context,
                    metadata=metadata,
                    execution_id=execution_id,
                    **kwargs,
                )

                if result.success:
                    return result

                if not allow_fallback:
                    return result

            except Exception:
                if not allow_fallback:
                    raise

        if allow_fallback:
            self.fallback_count += 1

            fallback = await self._execute_fallback(
                task=task,
                agents=agents,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
                context=context,
                metadata=metadata,
                execution_id=execution_id,
                **kwargs,
            )

            return fallback

        return self._create_failure_result(
            task=task,
            agent_name=(
                selected_name
                or "orchestrator"
            ),
            message=(
                "Router tidak menemukan "
                "agent yang dapat menjalankan task."
            ),
            execution_id=execution_id,
        )

    # ========================================================
    # NAMED AGENT EXECUTION
    # ========================================================

    async def _execute_named_agent(
        self,
        agent_name: str,
        task: str,
        timeout_seconds: float,
        retry_count: int,
        context: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        execution_id: str,
        **kwargs: Any,
    ) -> AgentResult:

        try:
            agent = self.registry.get(
                agent_name
            )
        except KeyError as exc:
            return self._create_failure_result(
                task=task,
                agent_name=agent_name,
                message=str(exc),
                execution_id=execution_id,
            )

        payload = dict(kwargs)

        if context is not None:
            payload["context"] = dict(
                context
            )

        if metadata is not None:
            payload["metadata"] = dict(
                metadata
            )

        payload[
            "orchestration_execution_id"
        ] = execution_id

        return await self._execute_with_retry(
            agent=agent,
            task=task,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            payload=payload,
            execution_id=execution_id,
        )

    # ========================================================
    # RETRY
    # ========================================================

    async def _execute_with_retry(
        self,
        agent: BaseAgent,
        task: str,
        timeout_seconds: float,
        retry_count: int,
        payload: dict[str, Any],
        execution_id: str,
    ) -> AgentResult:

        attempts = retry_count + 1

        last_result: AgentResult | None = None

        for attempt in range(attempts):

            if attempt > 0:
                self.retry_count += 1

                await asyncio.sleep(
                    min(
                        0.1 * attempt,
                        1.0,
                    )
                )

            try:
                result = await self._execute_agent(
                    agent=agent,
                    task=task,
                    timeout_seconds=timeout_seconds,
                    payload=payload,
                )

                result.set_metadata(
                    "attempt",
                    attempt + 1,
                )

                result.set_metadata(
                    "max_attempts",
                    attempts,
                )

                result.set_metadata(
                    "retry_used",
                    attempt > 0,
                )

                if result.success:
                    return result

                last_result = result

            except asyncio.TimeoutError:

                self.timeout_count += 1

                last_result = (
                    self._create_failure_result(
                        task=task,
                        agent_name=agent.name,
                        message=(
                            f"Agent '{agent.name}' "
                            f"timeout setelah "
                            f"{timeout_seconds} detik."
                        ),
                        execution_id=execution_id,
                    )
                )

            except Exception as exc:

                last_result = (
                    self._create_failure_result(
                        task=task,
                        agent_name=agent.name,
                        message=(
                            f"Agent '{agent.name}' "
                            f"gagal: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                        execution_id=execution_id,
                    )
                )

        if last_result is not None:
            return last_result

        return self._create_failure_result(
            task=task,
            agent_name=agent.name,
            message=(
                "Agent gagal tanpa menghasilkan result."
            ),
            execution_id=execution_id,
        )

    # ========================================================
    # LOW LEVEL AGENT EXECUTION
    # ========================================================

    async def _execute_agent(
        self,
        agent: BaseAgent,
        task: str,
        timeout_seconds: float,
        payload: dict[str, Any],
    ) -> AgentResult:

        execute_method = getattr(
            agent,
            "execute",
            None,
        )

        if not callable(execute_method):
            raise TypeError(
                f"Agent '{agent.name}' "
                "tidak memiliki execute()."
            )

        result = execute_method(
            task,
            **payload,
        )

        if inspect.isawaitable(result):

            return await asyncio.wait_for(
                result,
                timeout=timeout_seconds,
            )

        return result

    # ========================================================
    # FALLBACK
    # ========================================================

    async def _execute_fallback(
        self,
        task: str,
        agents: Sequence[BaseAgent],
        timeout_seconds: float,
        retry_count: int,
        context: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        execution_id: str,
        **kwargs: Any,
    ) -> AgentResult:

        ordered_agents = list(agents)

        if not ordered_agents:
            return self._create_failure_result(
                task=task,
                agent_name="orchestrator",
                message=(
                    "Fallback gagal: "
                    "tidak ada agent tersedia."
                ),
                execution_id=execution_id,
            )

        for agent in ordered_agents:

            result = await self._execute_named_agent(
                agent_name=agent.name,
                task=task,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
                context=context,
                metadata=metadata,
                execution_id=execution_id,
                **kwargs,
            )

            result.set_metadata(
                "fallback_used",
                True,
            )

            if result.success:
                return result

        return self._create_failure_result(
            task=task,
            agent_name=ordered_agents[0].name,
            message=(
                "Semua fallback agent gagal "
                "menjalankan task."
            ),
            execution_id=execution_id,
        )

    # ========================================================
    # PARALLEL EXECUTION
    # ========================================================

    async def execute_parallel(
        self,
        task: str,
        agent_names: Sequence[str] | None = None,
        *,
        timeout_seconds: float | None = None,
        retry_count: int | None = None,
        allow_fallback: bool = True,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:

        task = safe_task_text(task)

        timeout = (
            self.default_timeout
            if timeout_seconds is None
            else clamp_float(
                timeout_seconds,
                0.1,
                MAX_TIMEOUT_SECONDS,
            )
        )

        retries = (
            self.default_retry_count
            if retry_count is None
            else clamp_int(
                retry_count,
                0,
                MAX_RETRY_COUNT,
            )
        )

        if agent_names:
            agents = []

            for name in agent_names:
                try:
                    agents.append(
                        self.registry.get(name)
                    )
                except KeyError:
                    continue
        else:
            agents = self.agents()

        if not agents:
            return self._create_failure_result(
                task=task,
                agent_name="orchestrator",
                message=(
                    "Tidak ada agent untuk "
                    "parallel execution."
                ),
            )

        execution_id = str(
            uuid.uuid4()
        )

        async def run_one(
            agent: BaseAgent,
        ) -> AgentResult:

            return await self._execute_named_agent(
                agent_name=agent.name,
                task=task,
                timeout_seconds=timeout,
                retry_count=retries,
                context=context,
                metadata=metadata,
                execution_id=execution_id,
                **kwargs,
            )

        results = await asyncio.gather(
            *[
                run_one(agent)
                for agent in agents
            ],
            return_exceptions=True,
        )

        successful: list[AgentResult] = []
        failed: list[Any] = []

        for item in results:

            if isinstance(
                item,
                AgentResult,
            ):
                if item.success:
                    successful.append(item)
                else:
                    failed.append(item)

            else:
                failed.append(item)

        if successful:

            primary = successful[0]

            primary.set_metadata(
                "parallel_mode",
                True,
            )

            primary.set_metadata(
                "parallel_agent_count",
                len(agents),
            )

            primary.set_metadata(
                "parallel_success_count",
                len(successful),
            )

            primary.set_metadata(
                "parallel_failure_count",
                len(failed),
            )

            parallel_results = []

            for result in successful:
                try:
                    parallel_results.append(
                        result.to_dict()
                    )
                except Exception:
                    parallel_results.append(
                        {
                            "agent": getattr(
                                result,
                                "agent",
                                None,
                            ),
                            "success": getattr(
                                result,
                                "success",
                                False,
                            ),
                        }
                    )

            primary.set_metadata(
                "parallel_results",
                parallel_results,
            )

            return primary

        return self._create_failure_result(
            task=task,
            agent_name="orchestrator",
            message=(
                "Semua agent pada parallel "
                "execution gagal."
            ),
        )

    # ========================================================
    # SEQUENTIAL EXECUTION
    # ========================================================

    async def execute_sequential(
        self,
        task: str,
        agent_names: Sequence[str] | None = None,
        *,
        timeout_seconds: float | None = None,
        retry_count: int | None = None,
        allow_fallback: bool = True,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        stop_on_failure: bool = False,
        **kwargs: Any,
    ) -> AgentResult:

        task = safe_task_text(task)

        timeout = (
            self.default_timeout
            if timeout_seconds is None
            else clamp_float(
                timeout_seconds,
                0.1,
                MAX_TIMEOUT_SECONDS,
            )
        )

        retries = (
            self.default_retry_count
            if retry_count is None
            else clamp_int(
                retry_count,
                0,
                MAX_RETRY_COUNT,
            )
        )

        if agent_names:
            agents = []

            for name in agent_names:
                try:
                    agents.append(
                        self.registry.get(name)
                    )
                except KeyError:
                    continue

        else:
            agents = self.agents()

        if not agents:
            return self._create_failure_result(
                task=task,
                agent_name="orchestrator",
                message=(
                    "Tidak ada agent untuk "
                    "sequential execution."
                ),
            )

        execution_id = str(
            uuid.uuid4()
        )

        current_task = task

        current_context: dict[str, Any] = {}

        if context:
            current_context.update(
                dict(context)
            )

        sequence_results: list[
            dict[str, Any]
        ] = []

        final_result: AgentResult | None = None

        for index, agent in enumerate(
            agents,
            start=1,
        ):

            result = await self._execute_named_agent(
                agent_name=agent.name,
                task=current_task,
                timeout_seconds=timeout,
                retry_count=retries,
                context=current_context,
                metadata=metadata,
                execution_id=execution_id,
                **kwargs,
            )

            result.set_metadata(
                "sequence_index",
                index,
            )

            result.set_metadata(
                "sequence_total",
                len(agents),
            )

            try:
                sequence_results.append(
                    result.to_dict()
                )
            except Exception:
                sequence_results.append(
                    {
                        "agent": agent.name,
                        "success": result.success,
                        "response": result.response,
                    }
                )

            final_result = result

            if not result.success:
                if stop_on_failure:
                    break

                if allow_fallback:
                    continue

            response = getattr(
                result,
                "response",
                "",
            )

            if response:
                current_context[
                    f"{agent.name}_response"
                ] = response

                current_task = (
                    f"{task}\n\n"
                    f"Previous agent "
                    f"({agent.name}) output:\n"
                    f"{response}"
                )

        if final_result is None:
            return self._create_failure_result(
                task=task,
                agent_name="orchestrator",
                message=(
                    "Sequential execution "
                    "tidak menghasilkan result."
                ),
            )

        final_result.set_metadata(
            "sequential_mode",
            True,
        )

        final_result.set_metadata(
            "sequential_agent_count",
            len(agents),
        )

        final_result.set_metadata(
            "sequential_results",
            sequence_results,
        )

        return final_result

    # ========================================================
    # FINISH EXECUTION
    # ========================================================

    async def _finish_execution(
        self,
        result: AgentResult,
        record: OrchestrationRecord,
        started: float,
    ) -> AgentResult:

        latency_ms = round(
            (time.perf_counter() - started)
            * 1000,
            2,
        )

        record.completed_at = utc_now()
        record.latency_ms = latency_ms
        record.success = bool(
            getattr(
                result,
                "success",
                False,
            )
        )

        record.status = (
            "completed"
            if record.success
            else "failed"
        )

        record.retry_count = int(
            getattr(
                result,
                "metadata",
                {},
            ).get(
                "attempt",
                1,
            )
        ) - 1

        record.fallback_used = bool(
            getattr(
                result,
                "metadata",
                {},
            ).get(
                "fallback_used",
                False,
            )
        )

        try:
            result.set_metadata(
                "orchestrator_execution_id",
                record.execution_id,
            )

            result.set_metadata(
                "orchestrator_version",
                self.VERSION,
            )

            result.set_metadata(
                "orchestrator_latency_ms",
                latency_ms,
            )

        except Exception:
            pass

        if record.success:
            self.success_count += 1
        else:
            self.failure_count += 1

        async with self._lock:
            self._history.append(
                record
            )

        return result

    # ========================================================
    # HEALTH
    # ========================================================

    def health(self) -> dict[str, Any]:
        """
        Return health status orchestrator.
        """

        unhealthy_agents: list[str] = []

        for agent in self.agents():

            try:
                health_method = getattr(
                    agent,
                    "health",
                    None,
                )

                if not callable(
                    health_method
                ):
                    continue

                health = health_method()

                if isinstance(
                    health,
                    Mapping,
                ):
                    status = str(
                        health.get(
                            "status",
                            "UNKNOWN",
                        )
                    ).upper()

                    if status not in {
                        "HEALTHY",
                        "READY",
                        "OK",
                    }:
                        unhealthy_agents.append(
                            agent.name
                        )

            except Exception:
                unhealthy_agents.append(
                    agent.name
                )

        status = (
            "HEALTHY"
            if not unhealthy_agents
            else "DEGRADED"
        )

        return {
            "orchestrator": self.NAME,
            "version": self.VERSION,
            "status": status,
            "registered_agents": len(
                self.agent_names()
            ),
            "unhealthy_agents": unhealthy_agents,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
            "fallback_count": self.fallback_count,
            "retry_count": self.retry_count,
            "timeout_count": self.timeout_count,
            "registry_status": self._registry_status(),
        }

    # ========================================================
    # REGISTRY STATUS
    # ========================================================

    def _registry_status(self) -> str:
        try:
            summary = self.registry.summary()

            return str(
                summary.get(
                    "status",
                    "UNKNOWN",
                )
            )

        except Exception:
            return "UNKNOWN"

    # ========================================================
    # SUCCESS RATE
    # ========================================================

    def success_rate(self) -> float:
        if self.execution_count <= 0:
            return 0.0

        return round(
            (
                self.success_count
                / self.execution_count
            )
            * 100,
            2,
        )

    # ========================================================
    # INFO
    # ========================================================

    def info(self) -> dict[str, Any]:
        """
        Detailed orchestrator information.
        """

        router_info: dict[str, Any] = {}

        if self.router is not None:

            try:
                info_method = getattr(
                    self.router,
                    "info",
                    None,
                )

                if callable(
                    info_method
                ):
                    value = info_method()

                    if isinstance(
                        value,
                        Mapping,
                    ):
                        router_info = dict(
                            value
                        )

                if not router_info:
                    summary_method = getattr(
                        self.router,
                        "summary",
                        None,
                    )

                    if callable(
                        summary_method
                    ):
                        value = summary_method()

                        if isinstance(
                            value,
                            Mapping,
                        ):
                            router_info = dict(
                                value
                            )

            except Exception:
                router_info = {}

        try:
            registry_info = self.registry.summary()
        except Exception:
            registry_info = {
                "status": "UNKNOWN"
            }

        return {
            "orchestrator": self.NAME,
            "version": self.VERSION,
            "status": "READY",
            "started_at": self.started_at,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
            "fallback_count": self.fallback_count,
            "retry_count": self.retry_count,
            "timeout_count": self.timeout_count,
            "default_timeout": self.default_timeout,
            "default_retry_count": (
                self.default_retry_count
            ),
            "history_size": len(
                self._history
            ),
            "registered_agents": self.agent_names(),
            "router": router_info,
            "registry": registry_info,
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> dict[str, Any]:
        """
        Lightweight summary.
        """

        return {
            "name": self.NAME,
            "version": self.VERSION,
            "status": "READY",
            "agents": len(
                self.agent_names()
            ),
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
            "fallback_count": self.fallback_count,
            "retry_count": self.retry_count,
            "timeout_count": self.timeout_count,
        }

    # ========================================================
    # HISTORY
    # ========================================================

    def history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        records = list(
            self._history
        )

        if limit is not None:

            limit = clamp_int(
                limit,
                1,
                self.history_size,
            )

            records = records[-limit:]

        return [
            record.to_dict()
            for record in records
        ]

    def clear_history(self) -> None:
        self._history.clear()

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(self) -> dict[str, Any]:

        return {
            "orchestrator": self.NAME,
            "version": self.VERSION,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
            "fallback_count": self.fallback_count,
            "retry_count": self.retry_count,
            "timeout_count": self.timeout_count,
            "history_count": len(
                self._history
            ),
            "agent_count": len(
                self.agent_names()
            ),
        }

    # ========================================================
    # RESET STATISTICS
    # ========================================================

    def reset_statistics(
        self,
        clear_history: bool = False,
    ) -> None:

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.fallback_count = 0
        self.retry_count = 0
        self.timeout_count = 0

        if clear_history:
            self.clear_history()

    # ========================================================
    # AGENT HEALTH MAP
    # ========================================================

    def agent_health(
        self,
    ) -> dict[str, Any]:

        output: dict[str, Any] = {}

        for agent in self.agents():

            try:

                method = getattr(
                    agent,
                    "health",
                    None,
                )

                if callable(method):
                    output[
                        agent.name
                    ] = method()
                else:
                    output[
                        agent.name
                    ] = {
                        "agent": agent.name,
                        "status": "UNKNOWN",
                    }

            except Exception as exc:

                output[
                    agent.name
                ] = {
                    "agent": agent.name,
                    "status": "UNHEALTHY",
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

        return output

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, list[str]]:

        output: dict[
            str,
            list[str],
        ] = {}

        for agent in self.agents():

            capabilities = getattr(
                agent,
                "capabilities",
                (),
            )

            output[
                agent.name
            ] = list(
                capabilities
            )

        return output

    # ========================================================
    # FIND AGENTS BY CAPABILITY
    # ========================================================

    def find_by_capability(
        self,
        capability: str,
    ) -> list[str]:

        capability = (
            safe_task_text(
                capability
            ).lower()
        )

        if not capability:
            return []

        matches: list[str] = []

        for agent in self.agents():

            capabilities = getattr(
                agent,
                "capabilities",
                (),
            )

            normalized = {
                str(item).lower()
                for item in capabilities
            }

            if capability in normalized:
                matches.append(
                    agent.name
                )

        return matches

    # ========================================================
    # TASK CLASSIFICATION
    # ========================================================

    def classify_task(
        self,
        task: str,
    ) -> dict[str, Any]:

        normalized = safe_task_text(
            task
        ).lower()

        if not normalized:
            return {
                "task_type": "empty",
                "confidence": 1.0,
            }

        coding_keywords = {
            "code",
            "coding",
            "python",
            "dart",
            "flutter",
            "javascript",
            "typescript",
            "debug",
            "bug",
            "program",
            "programming",
            "compile",
            "error",
        }

        research_keywords = {
            "riset",
            "research",
            "penelitian",
            "cari informasi",
            "sumber",
            "referensi",
            "analisis teknologi",
            "bandingkan",
        }

        system_keywords = {
            "system",
            "sistem",
            "server",
            "runtime",
            "health",
            "status",
            "diagnostic",
            "diagnostik",
            "cpu",
            "memory",
            "ram",
        }

        coding_score = sum(
            1
            for keyword in coding_keywords
            if keyword in normalized
        )

        research_score = sum(
            1
            for keyword in research_keywords
            if keyword in normalized
        )

        system_score = sum(
            1
            for keyword in system_keywords
            if keyword in normalized
        )

        scores = {
            "coding": coding_score,
            "research": research_score,
            "system": system_score,
        }

        selected = max(
            scores,
            key=scores.get,
        )

        maximum = scores[selected]

        if maximum <= 0:
            return {
                "task_type": "general",
                "confidence": 0.5,
                "scores": scores,
            }

        confidence = min(
            0.99,
            0.5 + (
                maximum * 0.1
            ),
        )

        return {
            "task_type": selected,
            "confidence": round(
                confidence,
                2,
            ),
            "scores": scores,
        }

    # ========================================================
    # PLAN
    # ========================================================

    def plan(
        self,
        task: str,
    ) -> dict[str, Any]:

        classification = (
            self.classify_task(
                task
            )
        )

        task_type = classification[
            "task_type"
        ]

        recommended_agent = None

        mapping = {
            "coding": "coding_agent",
            "research": "research_agent",
            "system": "system_agent",
            "general": "general_agent",
        }

        candidate = mapping.get(
            task_type
        )

        if (
            candidate
            and self.has_agent(candidate)
        ):
            recommended_agent = candidate

        if recommended_agent is None:

            candidates = (
                self.find_by_capability(
                    task_type
                )
            )

            if candidates:
                recommended_agent = (
                    candidates[0]
                )

        return {
            "task": safe_task_text(
                task
            ),
            "task_type": task_type,
            "confidence": classification[
                "confidence"
            ],
            "recommended_agent": (
                recommended_agent
            ),
            "available_agents": (
                self.agent_names()
            ),
            "classification": classification,
        }

    # ========================================================
    # EXECUTE PLAN
    # ========================================================

    async def execute_plan(
        self,
        task: str,
        *,
        timeout_seconds: float | None = None,
        retry_count: int | None = None,
        allow_fallback: bool = True,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:

        plan = self.plan(
            task
        )

        agent_name = plan.get(
            "recommended_agent"
        )

        if agent_name:
            return await self.execute(
                task,
                agent_name=agent_name,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
                allow_fallback=allow_fallback,
                context=context,
                metadata=metadata,
                **kwargs,
            )

        return await self.execute(
            task,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            allow_fallback=allow_fallback,
            context=context,
            metadata=metadata,
            **kwargs,
        )

    # ========================================================
    # SAFE EXECUTION
    # ========================================================

    async def safe_execute(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:

        try:
            return await self.execute(
                task,
                **kwargs,
            )

        except Exception as exc:

            return self._create_failure_result(
                task=safe_task_text(
                    task
                ),
                agent_name="orchestrator",
                message=(
                    f"Safe execution error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )


# ============================================================
# FACTORY
# ============================================================


def create_orchestrator(
    registry: AgentRegistry | None = None,
    router: AgentRouter | None = None,
    runtime: Any | None = None,
) -> AgentOrchestrator:
    """
    Factory sederhana untuk ZAI.
    """

    return AgentOrchestrator(
        registry=registry,
        router=router,
        runtime=runtime,
    )


# ============================================================
# MODULE SELF TEST
# ============================================================


async def _self_test() -> None:
    """
    Internal smoke test.

    Tidak dijalankan saat module di-import.
    """

    orchestrator = AgentOrchestrator()

    assert (
        orchestrator.NAME
        == "AgentOrchestrator"
    )

    assert (
        orchestrator.VERSION
        == "2.0.0"
    )

    assert (
        orchestrator.execution_count
        == 0
    )

    health = orchestrator.health()

    assert health[
        "registered_agents"
    ] == 0

    summary = orchestrator.summary()

    assert summary[
        "status"
    ] == "READY"

    print(
        "ORCHESTRATOR_SELF_TEST_OK"
    )


if __name__ == "__main__":

    asyncio.run(
        _self_test()
    )
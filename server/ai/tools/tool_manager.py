from __future__ import annotations

"""
ZAI Tool Manager
================

High-level orchestration layer untuk ToolRegistry.

Tanggung jawab utama:
    1. Mengelola lifecycle tool.
    2. Menyediakan facade API di atas ToolRegistry.
    3. Menjalankan tool secara sync.
    4. Menjalankan tool secara async.
    5. Menjalankan batch tool.
    6. Menangani permission.
    7. Menangani whitelist.
    8. Menyediakan health check.
    9. Menyediakan statistics.
    10. Menyediakan execution history.
    11. Menyediakan tool discovery.
    12. Menyediakan capability summary.
    13. Menyediakan safe execution metadata.
    14. Menjadi fondasi integrasi AgentOrchestrator.
    15. Menjadi fondasi integrasi ZAI Planner/Executor.

Design goals:
    - Backward compatible dengan ToolRegistry.
    - Tidak menjalankan arbitrary code.
    - Tidak mengubah kontrak ToolRegistry.
    - Thread-safe pada operasi state sederhana.
    - Async-friendly.
    - Mudah dikembangkan.
    - Tidak bergantung pada FastAPI.
    - Tidak bergantung pada database.
    - Tidak bergantung pada LLM.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from time import perf_counter
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from .tool_registry import (
    AsyncToolCallable,
    ToolCallable,
    ToolDefinition,
    ToolExecutionRecord,
    ToolExecutionStatus,
    ToolRegistry,
    ToolResult,
)


# ============================================================================
# CONSTANTS
# ============================================================================

MANAGER_NAME = "ToolManager"
MANAGER_VERSION = "1.0.0"

STATUS_READY = "READY"
STATUS_HEALTHY = "HEALTHY"
STATUS_DEGRADED = "DEGRADED"
STATUS_FAILED = "FAILED"

DEFAULT_BATCH_CONCURRENCY = 8
DEFAULT_HISTORY_LIMIT = 100
DEFAULT_DISCOVERY_LIMIT = 100


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def utc_now() -> datetime:
    """
    Return timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    """
    Return ISO-8601 UTC timestamp.
    """
    return utc_now().isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value to float without raising.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_name(value: str) -> str:
    """
    Normalize tool name.

    Example:
        " Add " -> "add"
    """
    if not isinstance(value, str):
        raise TypeError("Nama tool harus berupa string.")

    normalized = value.strip().lower()

    if not normalized:
        raise ValueError("Nama tool tidak boleh kosong.")

    return normalized


def normalize_arguments(
    arguments: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Normalize tool arguments menjadi dictionary.
    """
    if arguments is None:
        return {}

    if not isinstance(arguments, Mapping):
        raise TypeError("arguments harus berupa mapping/dictionary.")

    return dict(arguments)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class ToolManagerConfig:
    """
    Configuration untuk ToolManager.
    """

    max_history: int = DEFAULT_HISTORY_LIMIT
    max_discovery_results: int = DEFAULT_DISCOVERY_LIMIT
    batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY
    allow_unknown_tools: bool = False
    auto_whitelist_registered_tools: bool = False
    strict_health_check: bool = False

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history harus > 0.")

        if self.max_discovery_results <= 0:
            raise ValueError(
                "max_discovery_results harus > 0."
            )

        if self.batch_concurrency <= 0:
            raise ValueError(
                "batch_concurrency harus > 0."
            )


@dataclass(slots=True)
class ToolManagerEvent:
    """
    Event internal ToolManager.

    Event ini bukan execution record dari ToolRegistry.
    Event ini digunakan untuk observability lifecycle manager.
    """

    event_id: str
    event: str
    created_at: str
    tool: str | None = None
    execution_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event": self.event,
            "created_at": self.created_at,
            "tool": self.tool,
            "execution_id": self.execution_id,
            "data": dict(self.data),
        }


@dataclass(slots=True)
class ToolBatchItem:
    """
    Representasi satu item batch.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any],
    ) -> "ToolBatchItem":
        if not isinstance(value, Mapping):
            raise TypeError(
                "Batch item harus berupa mapping."
            )

        tool = normalize_name(
            str(value.get("tool", ""))
        )

        arguments = normalize_arguments(
            value.get("arguments")
        )

        permissions_value = value.get(
            "permissions",
            [],
        )

        if permissions_value is None:
            permissions_value = []

        if isinstance(
            permissions_value,
            str,
        ):
            permissions = [permissions_value]
        else:
            permissions = [
                str(item)
                for item in permissions_value
            ]

        metadata_value = value.get(
            "metadata",
            {},
        )

        metadata = (
            dict(metadata_value)
            if isinstance(
                metadata_value,
                Mapping,
            )
            else {}
        )

        return cls(
            tool=tool,
            arguments=arguments,
            permissions=permissions,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "permissions": list(self.permissions),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ToolBatchResult:
    """
    Hasil eksekusi batch.
    """

    success: bool
    batch_id: str
    status: str
    results: list[ToolResult]
    total: int
    completed: int
    failed: int
    denied: int
    latency_ms: float
    created_at: str
    completed_at: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "batch_id": self.batch_id,
            "status": self.status,
            "results": [
                item.to_dict()
                for item in self.results
            ],
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "denied": self.denied,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# TOOL MANAGER
# ============================================================================


class ToolManager:
    """
    High-level manager untuk seluruh tool ZAI.

    ToolManager tidak menggantikan ToolRegistry.

    Architecture:

        Agent
          |
          v
        ToolManager
          |
          v
        ToolRegistry
          |
          v
        ToolCallable

    Dengan struktur ini agent dapat menggunakan:

        manager.execute(...)
        manager.execute_async(...)
        manager.execute_batch(...)
        manager.discover(...)
        manager.health()
        manager.statistics()
    """

    name = MANAGER_NAME
    version = MANAGER_VERSION

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        config: ToolManagerConfig | None = None,
    ) -> None:
        self.registry = (
            registry
            if registry is not None
            else ToolRegistry()
        )

        self.config = (
            config
            if config is not None
            else ToolManagerConfig()
        )

        self._lock = RLock()

        self._events: list[ToolManagerEvent] = []

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.denied_count = 0

        self.created_at = utc_iso()

        self._emit(
            "manager_initialized",
            data={
                "version": self.version,
                "registry_version": getattr(
                    self.registry,
                    "VERSION",
                    "unknown",
                ),
            },
        )

    # ========================================================================
    # INTERNAL EVENT SYSTEM
    # ========================================================================

    def _emit(
        self,
        event: str,
        *,
        tool: str | None = None,
        execution_id: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> ToolManagerEvent:
        """
        Emit internal manager event.
        """

        item = ToolManagerEvent(
            event_id=str(uuid4()),
            event=event,
            created_at=utc_iso(),
            tool=tool,
            execution_id=execution_id,
            data=dict(data or {}),
        )

        with self._lock:
            self._events.append(item)

            if (
                len(self._events)
                > self.config.max_history
            ):
                overflow = (
                    len(self._events)
                    - self.config.max_history
                )

                del self._events[:overflow]

        return item

    def events(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return manager lifecycle events.
        """

        with self._lock:
            items = list(self._events)

        if limit is not None:
            if limit <= 0:
                return []

            items = items[-limit:]

        return [
            item.to_dict()
            for item in items
        ]

    # ========================================================================
    # REGISTRATION
    # ========================================================================

    def register(
        self,
        tool: ToolDefinition,
    ) -> ToolDefinition:
        """
        Register ToolDefinition ke registry.
        """

        if not isinstance(
            tool,
            ToolDefinition,
        ):
            raise TypeError(
                "tool harus berupa ToolDefinition."
            )

        name = normalize_name(tool.name)

        self.registry.register(tool)

        if (
            self.config
            .auto_whitelist_registered_tools
        ):
            self.registry.whitelist(name)

        self._emit(
            "tool_registered",
            tool=name,
            data={
                "auto_whitelisted": (
                    self.config
                    .auto_whitelist_registered_tools
                ),
            },
        )

        return tool

    def register_function(
        self,
        name: str,
        function: ToolCallable | AsyncToolCallable,
        *,
        description: str = "",
        category: str = "general",
        permissions: Iterable[str] | None = None,
        schema: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: str = "1.0.0",
    ) -> ToolDefinition:
        """
        Convenience API untuk register function.
        """

        normalized_name = normalize_name(name)

        permission_list = list(
            permissions or []
        )

        schema_dict = dict(
            schema or {}
        )

        metadata_dict = dict(
            metadata or {}
        )

        definition = ToolDefinition(
            name=normalized_name,
            function=function,
            description=description,
            category=category,
            permissions=permission_list,
            schema=schema_dict,
            metadata=metadata_dict,
            version=version,
        )

        self.register(definition)

        return definition

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Unregister tool jika registry mendukung API tersebut.
        """

        normalized_name = normalize_name(name)

        unregister_method = getattr(
            self.registry,
            "unregister",
            None,
        )

        if unregister_method is None:
            return False

        result = bool(
            unregister_method(
                normalized_name
            )
        )

        if result:
            self._emit(
                "tool_unregistered",
                tool=normalized_name,
            )

        return result

    # ========================================================================
    # DISCOVERY
    # ========================================================================

    def has_tool(
        self,
        name: str,
    ) -> bool:
        """
        Check apakah tool tersedia.
        """

        normalized_name = normalize_name(name)

        method = getattr(
            self.registry,
            "has",
            None,
        )

        if callable(method):
            return bool(
                method(normalized_name)
            )

        try:
            self.registry.get(
                normalized_name
            )
            return True
        except Exception:
            return False

    def get_tool(
        self,
        name: str,
    ) -> ToolDefinition | Any:
        """
        Ambil definisi tool dari registry.
        """

        normalized_name = normalize_name(name)

        return self.registry.get(
            normalized_name
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """
        Return semua tool aktif.
        """

        method = getattr(
            self.registry,
            "active",
            None,
        )

        if callable(method):
            return list(
                method()
            )

        definitions = getattr(
            self.registry,
            "_tools",
            {},
        )

        result: list[dict[str, Any]] = []

        if isinstance(
            definitions,
            Mapping,
        ):
            for item in definitions.values():
                if hasattr(
                    item,
                    "to_dict",
                ):
                    result.append(
                        item.to_dict()
                    )
                else:
                    result.append(
                        {
                            "name": getattr(
                                item,
                                "name",
                                None,
                            ),
                            "description": getattr(
                                item,
                                "description",
                                "",
                            ),
                        }
                    )

        return result

    def discover(
        self,
        query: str = "",
        *,
        category: str | None = None,
        capability: str | None = None,
        permission: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Discover tool berdasarkan query.

        Matching:
            - name
            - description
            - category
            - capabilities
            - permissions
        """

        query_normalized = (
            query.strip().lower()
            if isinstance(
                query,
                str,
            )
            else ""
        )

        category_normalized = (
            category.strip().lower()
            if isinstance(
                category,
                str,
            )
            else None
        )

        capability_normalized = (
            capability.strip().lower()
            if isinstance(
                capability,
                str,
            )
            else None
        )

        permission_normalized = (
            permission.strip().lower()
            if isinstance(
                permission,
                str,
            )
            else None
        )

        maximum = (
            limit
            if limit is not None
            else self.config.max_discovery_results
        )

        if maximum <= 0:
            return []

        tools = self.list_tools()

        ranked: list[
            tuple[int, dict[str, Any]]
        ] = []

        for tool in tools:
            name = str(
                tool.get(
                    "name",
                    "",
                )
            ).lower()

            description = str(
                tool.get(
                    "description",
                    "",
                )
            ).lower()

            tool_category = str(
                tool.get(
                    "category",
                    "",
                )
            ).lower()

            permissions = [
                str(item).lower()
                for item in tool.get(
                    "permissions",
                    [],
                )
                or []
            ]

            capabilities = [
                str(item).lower()
                for item in tool.get(
                    "capabilities",
                    [],
                )
                or []
            ]

            score = 0

            if query_normalized:
                if query_normalized == name:
                    score += 100

                if query_normalized in name:
                    score += 50

                if (
                    query_normalized
                    in description
                ):
                    score += 25

                if (
                    query_normalized
                    in tool_category
                ):
                    score += 20

                if any(
                    query_normalized in item
                    for item in capabilities
                ):
                    score += 20

            else:
                score = 1

            if category_normalized:
                if (
                    tool_category
                    != category_normalized
                ):
                    continue

                score += 20

            if capability_normalized:
                if (
                    capability_normalized
                    not in capabilities
                ):
                    continue

                score += 20

            if permission_normalized:
                if (
                    permission_normalized
                    not in permissions
                ):
                    continue

                score += 20

            if score > 0:
                item = dict(tool)
                item["_discovery_score"] = score

                ranked.append(
                    (
                        score,
                        item,
                    )
                )

        ranked.sort(
            key=lambda value: (
                -value[0],
                str(
                    value[1].get(
                        "name",
                        "",
                    )
                ),
            )
        )

        results = [
            item
            for _, item in ranked[:maximum]
        ]

        self._emit(
            "tool_discovery",
            data={
                "query": query_normalized,
                "category": category_normalized,
                "capability": capability_normalized,
                "permission": permission_normalized,
                "result_count": len(results),
            },
        )

        return results

    # ========================================================================
    # WHITELIST
    # ========================================================================

    def whitelist(
        self,
        name: str,
    ) -> Any:
        """
        Whitelist tool.
        """

        normalized_name = normalize_name(name)

        result = self.registry.whitelist(
            normalized_name
        )

        self._emit(
            "tool_whitelisted",
            tool=normalized_name,
        )

        return result

    def revoke_whitelist(
        self,
        name: str,
    ) -> Any:
        """
        Remove tool dari whitelist.
        """

        normalized_name = normalize_name(name)

        method = getattr(
            self.registry,
            "revoke_whitelist",
            None,
        )

        if method is None:
            method = getattr(
                self.registry,
                "unwhitelist",
                None,
            )

        if method is None:
            raise AttributeError(
                "ToolRegistry tidak menyediakan "
                "API revoke whitelist."
            )

        result = method(
            normalized_name
        )

        self._emit(
            "tool_whitelist_revoked",
            tool=normalized_name,
        )

        return result

    # ========================================================================
    # PERMISSION
    # ========================================================================

    def grant_permission(
        self,
        name: str,
        permission: str,
    ) -> Any:
        """
        Grant permission ke tool jika registry mendukung.
        """

        normalized_name = normalize_name(name)

        method = getattr(
            self.registry,
            "grant_permission",
            None,
        )

        if method is None:
            raise AttributeError(
                "ToolRegistry tidak menyediakan "
                "API grant_permission."
            )

        result = method(
            normalized_name,
            permission,
        )

        self._emit(
            "permission_granted",
            tool=normalized_name,
            data={
                "permission": permission,
            },
        )

        return result

    def revoke_permission(
        self,
        name: str,
        permission: str,
    ) -> Any:
        """
        Revoke permission dari tool.
        """

        normalized_name = normalize_name(name)

        method = getattr(
            self.registry,
            "revoke_permission",
            None,
        )

        if method is None:
            raise AttributeError(
                "ToolRegistry tidak menyediakan "
                "API revoke_permission."
            )

        result = method(
            normalized_name,
            permission,
        )

        self._emit(
            "permission_revoked",
            tool=normalized_name,
            data={
                "permission": permission,
            },
        )

        return result

    # ========================================================================
    # EXECUTION - SYNC
    # ========================================================================

    def execute(
        self,
        tool: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        permissions: Iterable[str] | None = None,
        timeout: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """
        Execute tool secara synchronous.

        timeout disediakan sebagai metadata/control parameter.
        Enforcement timeout tetap berada pada executor/registry
        jika registry mendukung timeout.
        """

        normalized_name = normalize_name(tool)
        normalized_arguments = normalize_arguments(
            arguments
        )

        execution_id = str(uuid4())

        self._emit(
            "tool_execution_started",
            tool=normalized_name,
            execution_id=execution_id,
            data={
                "argument_count": len(
                    normalized_arguments
                ),
            },
        )

        started = perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "arguments": normalized_arguments,
            }

            if permissions is not None:
                kwargs["permissions"] = list(
                    permissions
                )

            if timeout is not None:
                kwargs["timeout"] = timeout

            if metadata is not None:
                kwargs["metadata"] = dict(
                    metadata
                )

            result = self.registry.execute(
                normalized_name,
                **kwargs,
            )

        except TypeError:
            result = self.registry.execute(
                normalized_name,
                arguments=normalized_arguments,
            )

        except Exception as exc:
            latency_ms = round(
                (
                    perf_counter()
                    - started
                )
                * 1000,
                4,
            )

            self._record_failure()

            self._emit(
                "tool_execution_failed",
                tool=normalized_name,
                execution_id=execution_id,
                data={
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    "latency_ms": latency_ms,
                },
            )

            raise

        latency_ms = round(
            (
                perf_counter()
                - started
            )
            * 1000,
            4,
        )

        self._update_counters(
            result
        )

        self._emit(
            "tool_execution_completed",
            tool=normalized_name,
            execution_id=(
                getattr(
                    result,
                    "execution_id",
                    None,
                )
                or execution_id
            ),
            data={
                "success": bool(
                    getattr(
                        result,
                        "success",
                        False,
                    )
                ),
                "latency_ms": latency_ms,
            },
        )

        return result

    # ========================================================================
    # EXECUTION - ASYNC
    # ========================================================================

    async def execute_async(
        self,
        tool: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        permissions: Iterable[str] | None = None,
        timeout: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """
        Execute tool melalui async registry API jika tersedia.

        Fallback:
            synchronous registry.execute(...)
        """

        normalized_name = normalize_name(tool)
        normalized_arguments = normalize_arguments(
            arguments
        )

        async_method = getattr(
            self.registry,
            "execute_async",
            None,
        )

        if callable(async_method):
            execution_id = str(uuid4())

            self._emit(
                "async_tool_execution_started",
                tool=normalized_name,
                execution_id=execution_id,
                data={
                    "argument_count": len(
                        normalized_arguments
                    ),
                },
            )

            started = perf_counter()

            try:
                kwargs: dict[str, Any] = {
                    "arguments": normalized_arguments,
                }

                if permissions is not None:
                    kwargs["permissions"] = list(
                        permissions
                    )

                if timeout is not None:
                    kwargs["timeout"] = timeout

                if metadata is not None:
                    kwargs["metadata"] = dict(
                        metadata
                    )

                try:
                    result = await async_method(
                        normalized_name,
                        **kwargs,
                    )
                except TypeError:
                    result = await async_method(
                        normalized_name,
                        arguments=normalized_arguments,
                    )

            except Exception as exc:
                self._record_failure()

                self._emit(
                    "async_tool_execution_failed",
                    tool=normalized_name,
                    execution_id=execution_id,
                    data={
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    },
                )

                raise

            latency_ms = round(
                (
                    perf_counter()
                    - started
                )
                * 1000,
                4,
            )

            self._update_counters(
                result
            )

            self._emit(
                "async_tool_execution_completed",
                tool=normalized_name,
                execution_id=(
                    getattr(
                        result,
                        "execution_id",
                        None,
                    )
                    or execution_id
                ),
                data={
                    "success": bool(
                        getattr(
                            result,
                            "success",
                            False,
                        )
                    ),
                    "latency_ms": latency_ms,
                },
            )

            return result

        return self.execute(
            normalized_name,
            arguments=normalized_arguments,
            permissions=permissions,
            timeout=timeout,
            metadata=metadata,
        )

    # ========================================================================
    # EXECUTION - BATCH
    # ========================================================================

    async def execute_batch(
        self,
        items: Sequence[
            Mapping[str, Any] | ToolBatchItem
        ],
        *,
        concurrent: bool = True,
        stop_on_failure: bool = False,
        max_concurrency: int | None = None,
    ) -> ToolBatchResult:
        """
        Execute banyak tool.

        concurrent=True:
            menggunakan asyncio.gather jika tersedia.

        stop_on_failure:
            jika True, batch akan berhenti secara logis setelah failure.
            Tool yang sudah dimulai tetap dapat selesai.
        """

        import asyncio

        batch_id = str(uuid4())
        created_at = utc_iso()
        started = perf_counter()

        parsed_items: list[
            ToolBatchItem
        ] = []

        for item in items:
            if isinstance(
                item,
                ToolBatchItem,
            ):
                parsed_items.append(item)
            else:
                parsed_items.append(
                    ToolBatchItem.from_value(
                        item
                    )
                )

        total = len(parsed_items)

        self._emit(
            "batch_started",
            data={
                "batch_id": batch_id,
                "total": total,
                "concurrent": concurrent,
                "stop_on_failure": stop_on_failure,
            },
        )

        if total == 0:
            latency_ms = round(
                (
                    perf_counter()
                    - started
                )
                * 1000,
                4,
            )

            return ToolBatchResult(
                success=True,
                batch_id=batch_id,
                status="completed",
                results=[],
                total=0,
                completed=0,
                failed=0,
                denied=0,
                latency_ms=latency_ms,
                created_at=created_at,
                completed_at=utc_iso(),
                metadata={
                    "empty_batch": True,
                },
            )

        semaphore = asyncio.Semaphore(
            max(
                1,
                max_concurrency
                or self.config.batch_concurrency,
            )
        )

        async def execute_item(
            item: ToolBatchItem,
        ) -> ToolResult:
            async with semaphore:
                return await self.execute_async(
                    item.tool,
                    arguments=item.arguments,
                    permissions=item.permissions,
                    metadata=item.metadata,
                )

        results: list[ToolResult] = []

        if concurrent:
            gathered = await asyncio.gather(
                *[
                    execute_item(item)
                    for item in parsed_items
                ],
                return_exceptions=True,
            )

            for item in gathered:
                if isinstance(
                    item,
                    Exception,
                ):
                    result = self._exception_to_result(
                        item
                    )
                else:
                    result = item

                results.append(result)

                if (
                    stop_on_failure
                    and not result.success
                ):
                    break

        else:
            for item in parsed_items:
                result = await execute_item(
                    item
                )

                results.append(result)

                if (
                    stop_on_failure
                    and not result.success
                ):
                    break

        completed = sum(
            1
            for result in results
            if result.success
        )

        failed = sum(
            1
            for result in results
            if not result.success
            and getattr(
                result,
                "status",
                "",
            )
            not in {
                "denied",
                "blocked",
            }
        )

        denied = sum(
            1
            for result in results
            if getattr(
                result,
                "status",
                "",
            )
            in {
                "denied",
                "blocked",
            }
        )

        success = (
            failed == 0
            and denied == 0
            and len(results) == total
        )

        status = (
            "completed"
            if success
            else "partial"
            if completed > 0
            else "failed"
        )

        latency_ms = round(
            (
                perf_counter()
                - started
            )
            * 1000,
            4,
        )

        result = ToolBatchResult(
            success=success,
            batch_id=batch_id,
            status=status,
            results=results,
            total=total,
            completed=completed,
            failed=failed,
            denied=denied,
            latency_ms=latency_ms,
            created_at=created_at,
            completed_at=utc_iso(),
            metadata={
                "requested_total": total,
                "returned_total": len(results),
                "concurrent": concurrent,
                "stop_on_failure": stop_on_failure,
                "max_concurrency": (
                    max_concurrency
                    or self.config.batch_concurrency
                ),
            },
        )

        self._emit(
            "batch_completed",
            data={
                "batch_id": batch_id,
                "success": success,
                "status": status,
                "total": total,
                "completed": completed,
                "failed": failed,
                "denied": denied,
                "latency_ms": latency_ms,
            },
        )

        return result

    # ========================================================================
    # BATCH SYNC WRAPPER
    # ========================================================================

    def execute_batch_sync(
        self,
        items: Sequence[
            Mapping[str, Any] | ToolBatchItem
        ],
        *,
        concurrent: bool = False,
        stop_on_failure: bool = False,
        max_concurrency: int | None = None,
    ) -> ToolBatchResult:
        """
        Sync wrapper untuk execute_batch.

        Catatan:
            Jangan dipanggil dari running event loop.
        """

        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.execute_batch(
                    items,
                    concurrent=concurrent,
                    stop_on_failure=stop_on_failure,
                    max_concurrency=max_concurrency,
                )
            )

        raise RuntimeError(
            "execute_batch_sync() tidak dapat "
            "dipanggil dari running event loop. "
            "Gunakan await execute_batch()."
        )

    # ========================================================================
    # RESULT / COUNTERS
    # ========================================================================

    def _update_counters(
        self,
        result: ToolResult,
    ) -> None:
        """
        Update manager counters berdasarkan result registry.
        """

        with self._lock:
            self.execution_count += 1

            if bool(
                getattr(
                    result,
                    "success",
                    False,
                )
            ):
                self.success_count += 1
            else:
                self.failure_count += 1

                status = str(
                    getattr(
                        result,
                        "status",
                        "",
                    )
                ).lower()

                if status in {
                    "denied",
                    "blocked",
                }:
                    self.denied_count += 1

    def _record_failure(self) -> None:
        with self._lock:
            self.execution_count += 1
            self.failure_count += 1

    def _exception_to_result(
        self,
        exc: Exception,
    ) -> ToolResult:
        """
        Convert exception menjadi ToolResult jika constructor
        registry mendukung parameter standar.
        """

        try:
            result = ToolResult(
                success=False,
                tool="batch",
                status="failed",
                response=None,
                data=None,
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            return result

        except Exception:
            raise exc

    # ========================================================================
    # HISTORY
    # ========================================================================

    def history(
        self,
        *,
        limit: int | None = None,
        tool: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return execution history dari ToolRegistry.
        """

        history_method = getattr(
            self.registry,
            "history",
            None,
        )

        if not callable(history_method):
            return []

        kwargs: dict[str, Any] = {}

        if limit is not None:
            kwargs["limit"] = limit

        if tool is not None:
            kwargs["tool"] = normalize_name(
                tool
            )

        try:
            records = history_method(
                **kwargs
            )
        except TypeError:
            records = history_method()

        output: list[dict[str, Any]] = []

        for record in records:
            if hasattr(
                record,
                "to_dict",
            ):
                output.append(
                    record.to_dict()
                )
            elif isinstance(
                record,
                Mapping,
            ):
                output.append(
                    dict(record)
                )
            else:
                output.append(
                    {
                        "value": str(
                            record
                        )
                    }
                )

        return output

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Return combined statistics ToolManager + ToolRegistry.
        """

        with self._lock:
            execution_count = (
                self.execution_count
            )
            success_count = (
                self.success_count
            )
            failure_count = (
                self.failure_count
            )
            denied_count = (
                self.denied_count
            )

        success_rate = (
            round(
                (
                    success_count
                    / execution_count
                )
                * 100,
                2,
            )
            if execution_count
            else 0.0
        )

        registry_statistics_method = getattr(
            self.registry,
            "statistics",
            None,
        )

        registry_statistics: dict[
            str,
            Any,
        ] = {}

        if callable(
            registry_statistics_method
        ):
            try:
                registry_statistics = dict(
                    registry_statistics_method()
                )
            except Exception:
                registry_statistics = {}

        return {
            "manager": self.name,
            "version": self.version,
            "execution_count": execution_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "denied_count": denied_count,
            "success_rate": success_rate,
            "registry": registry_statistics,
        }

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health(self) -> dict[str, Any]:
        """
        Health check manager dan registry.
        """

        registry_health_method = getattr(
            self.registry,
            "health",
            None,
        )

        registry_health: dict[str, Any] = {}

        if callable(
            registry_health_method
        ):
            try:
                registry_health = dict(
                    registry_health_method()
                )
            except Exception as exc:
                registry_health = {
                    "status": STATUS_FAILED,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

        registry_status = str(
            registry_health.get(
                "status",
                STATUS_READY,
            )
        ).upper()

        if registry_status in {
            STATUS_FAILED,
            "UNHEALTHY",
        }:
            manager_status = STATUS_DEGRADED
        else:
            manager_status = STATUS_HEALTHY

        if (
            self.config.strict_health_check
            and not self.list_tools()
        ):
            manager_status = STATUS_DEGRADED

        statistics = self.statistics()

        return {
            "manager": self.name,
            "version": self.version,
            "status": manager_status,
            "registry_status": registry_status,
            "registered_tools": len(
                self.list_tools()
            ),
            "execution_count": statistics[
                "execution_count"
            ],
            "success_count": statistics[
                "success_count"
            ],
            "failure_count": statistics[
                "failure_count"
            ],
            "success_rate": statistics[
                "success_rate"
            ],
            "registry_health": registry_health,
        }

    # ========================================================================
    # INFO
    # ========================================================================

    def info(self) -> dict[str, Any]:
        """
        Return complete manager information.
        """

        tools = self.list_tools()
        statistics = self.statistics()

        return {
            "manager": self.name,
            "version": self.version,
            "status": STATUS_READY,
            "created_at": self.created_at,
            "registered_tools": len(
                tools
            ),
            "tools": tools,
            "statistics": statistics,
            "configuration": {
                "max_history": (
                    self.config.max_history
                ),
                "max_discovery_results": (
                    self.config.max_discovery_results
                ),
                "batch_concurrency": (
                    self.config.batch_concurrency
                ),
                "allow_unknown_tools": (
                    self.config.allow_unknown_tools
                ),
                "auto_whitelist_registered_tools": (
                    self.config
                    .auto_whitelist_registered_tools
                ),
                "strict_health_check": (
                    self.config.strict_health_check
                ),
            },
        }

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def summary(self) -> dict[str, Any]:
        """
        Compact manager summary.
        """

        statistics = self.statistics()

        return {
            "manager": self.name,
            "version": self.version,
            "status": STATUS_READY,
            "total_tools": len(
                self.list_tools()
            ),
            "execution_count": statistics[
                "execution_count"
            ],
            "success_count": statistics[
                "success_count"
            ],
            "failure_count": statistics[
                "failure_count"
            ],
            "denied_count": statistics[
                "denied_count"
            ],
            "success_rate": statistics[
                "success_rate"
            ],
        }


# ============================================================================
# FACTORY
# ============================================================================


def create_tool_manager(
    registry: ToolRegistry | None = None,
    *,
    config: ToolManagerConfig | None = None,
) -> ToolManager:
    """
    Factory resmi ZAI ToolManager.
    """

    return ToolManager(
        registry=registry,
        config=config,
    )


# ============================================================================
# DEFAULT MANAGER
# ============================================================================


_default_manager: ToolManager | None = None
_default_manager_lock = RLock()


def get_default_tool_manager() -> ToolManager:
    """
    Return singleton ToolManager.
    """

    global _default_manager

    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = ToolManager()

        return _default_manager


def reset_default_tool_manager() -> ToolManager:
    """
    Reset singleton manager.

    Berguna untuk testing.
    """

    global _default_manager

    with _default_manager_lock:
        _default_manager = ToolManager()

        return _default_manager


# ============================================================================
# PUBLIC HELPER FUNCTIONS
# ============================================================================


def register_tool(
    name: str,
    function: ToolCallable | AsyncToolCallable,
    *,
    description: str = "",
    category: str = "general",
    permissions: Iterable[str] | None = None,
    schema: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    version: str = "1.0.0",
) -> ToolDefinition:
    """
    Register tool ke default manager.
    """

    manager = get_default_tool_manager()

    return manager.register_function(
        name,
        function,
        description=description,
        category=category,
        permissions=permissions,
        schema=schema,
        metadata=metadata,
        version=version,
    )


def execute_tool(
    name: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    permissions: Iterable[str] | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ToolResult:
    """
    Execute tool melalui default manager.
    """

    manager = get_default_tool_manager()

    return manager.execute(
        name,
        arguments=arguments,
        permissions=permissions,
        timeout=timeout,
        metadata=metadata,
    )


async def execute_tool_async(
    name: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    permissions: Iterable[str] | None = None,
    timeout: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ToolResult:
    """
    Async execution helper.
    """

    manager = get_default_tool_manager()

    return await manager.execute_async(
        name,
        arguments=arguments,
        permissions=permissions,
        timeout=timeout,
        metadata=metadata,
    )


def discover_tools(
    query: str = "",
    *,
    category: str | None = None,
    capability: str | None = None,
    permission: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Discover tools melalui default manager.
    """

    manager = get_default_tool_manager()

    return manager.discover(
        query,
        category=category,
        capability=capability,
        permission=permission,
        limit=limit,
    )


def tool_manager_health() -> dict[str, Any]:
    """
    Health check default manager.
    """

    return get_default_tool_manager().health()


def tool_manager_info() -> dict[str, Any]:
    """
    Info default manager.
    """

    return get_default_tool_manager().info()


# ============================================================================
# SELF TESTS
# ============================================================================


def _self_test_sync() -> None:
    """
    Internal sync self-test.

    Tidak dijalankan otomatis ketika module di-import.
    """

    manager = ToolManager()

    manager.register_function(
        "hello",
        lambda: "Halo ZAI",
        description="Greeting test tool.",
        category="general",
    )

    manager.whitelist(
        "hello"
    )

    result = manager.execute(
        "hello"
    )

    assert result.success is True
    assert result.response == "Halo ZAI"


def _self_test_arguments() -> None:
    """
    Internal argument validation self-test.
    """

    manager = ToolManager()

    manager.register_function(
        "add",
        lambda a, b: a + b,
        description="Add two integers.",
        category="math",
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": [
                "a",
                "b",
            ],
            "additionalProperties": False,
        },
    )

    manager.whitelist(
        "add"
    )

    result = manager.execute(
        "add",
        arguments={
            "a": 10,
            "b": 20,
        },
    )

    assert result.success is True
    assert result.response == 30


async def _self_test_async() -> None:
    """
    Internal async self-test.
    """

    import asyncio

    manager = ToolManager()

    async def async_add(
        a: int,
        b: int,
    ) -> int:
        await asyncio.sleep(0)
        return a + b

    manager.register_function(
        "async_add",
        async_add,
        description="Async addition.",
        category="math",
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": [
                "a",
                "b",
            ],
        },
    )

    manager.whitelist(
        "async_add"
    )

    result = await manager.execute_async(
        "async_add",
        arguments={
            "a": 100,
            "b": 200,
        },
    )

    assert result.success is True
    assert result.response == 300


async def _self_test_batch() -> None:
    """
    Internal batch self-test.
    """

    manager = ToolManager()

    manager.register_function(
        "add",
        lambda a, b: a + b,
        description="Addition.",
        category="math",
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": [
                "a",
                "b",
            ],
        },
    )

    manager.whitelist(
        "add"
    )

    result = await manager.execute_batch(
        [
            {
                "tool": "add",
                "arguments": {
                    "a": 1,
                    "b": 2,
                },
            },
            {
                "tool": "add",
                "arguments": {
                    "a": 10,
                    "b": 20,
                },
            },
            {
                "tool": "add",
                "arguments": {
                    "a": 100,
                    "b": 200,
                },
            },
        ]
    )

    assert result.success is True
    assert result.total == 3
    assert result.completed == 3


def run_self_tests() -> None:
    """
    Jalankan seluruh self-test.
    """

    import asyncio

    _self_test_sync()
    _self_test_arguments()
    asyncio.run(
        _self_test_async()
    )
    asyncio.run(
        _self_test_batch()
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    "MANAGER_NAME",
    "MANAGER_VERSION",
    "STATUS_READY",
    "STATUS_HEALTHY",
    "STATUS_DEGRADED",
    "STATUS_FAILED",
    "ToolManagerConfig",
    "ToolManagerEvent",
    "ToolBatchItem",
    "ToolBatchResult",
    "ToolManager",
    "create_tool_manager",
    "get_default_tool_manager",
    "reset_default_tool_manager",
    "register_tool",
    "execute_tool",
    "execute_tool_async",
    "discover_tools",
    "tool_manager_health",
    "tool_manager_info",
    "run_self_tests",
]
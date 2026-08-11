from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    TypeAlias,
    runtime_checkable,
)
from uuid import uuid4


# ============================================================
# TYPE DEFINITIONS
# ============================================================

ToolArguments: TypeAlias = dict[str, Any]
ToolSchema: TypeAlias = dict[str, Any]
ToolPermission: TypeAlias = str


@runtime_checkable
class SyncToolCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class AsyncToolCallable(Protocol):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


ToolCallable: TypeAlias = (
    Callable[..., Any]
    | SyncToolCallable
    | AsyncToolCallable
)


# ============================================================
# ENUMS
# ============================================================

class ToolExecutionStatus(str, Enum):
    """
    Status eksekusi tool ZAI.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolPermissionStatus(str, Enum):
    """
    Status permission tool.
    """

    ALLOWED = "allowed"
    DENIED = "denied"
    MISSING = "missing"


# ============================================================
# TOOL DEFINITION
# ============================================================

@dataclass(slots=True)
class ToolDefinition:
    """
    Definisi sebuah tool yang dapat digunakan ZAI.
    """

    name: str
    function: ToolCallable

    description: str = ""

    permissions: list[str] = field(default_factory=list)

    schema: ToolSchema = field(default_factory=dict)

    enabled: bool = True

    whitelist_required: bool = False

    category: str = "general"

    version: str = "1.0.0"

    tags: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    timeout_seconds: float | None = None

    allow_positional_arguments: bool = False

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    execution_count: int = 0

    success_count: int = 0

    failure_count: int = 0

    last_execution_at: str | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()

        if not self.name:
            raise ValueError(
                "Nama tool tidak boleh kosong."
            )

        if not re.match(
            r"^[A-Za-z_][A-Za-z0-9_.-]*$",
            self.name,
        ):
            raise ValueError(
                f"Nama tool tidak valid: {self.name}"
            )

        if not callable(self.function):
            raise TypeError(
                f"Tool '{self.name}' harus callable."
            )

        self.description = str(
            self.description or ""
        ).strip()

        self.permissions = [
            str(permission).strip()
            for permission in self.permissions
            if str(permission).strip()
        ]

        self.tags = [
            str(tag).strip()
            for tag in self.tags
            if str(tag).strip()
        ]

        if not isinstance(
            self.schema,
            dict,
        ):
            raise TypeError(
                "schema harus berupa dictionary."
            )

        if self.timeout_seconds is not None:
            self.timeout_seconds = float(
                self.timeout_seconds
            )

            if self.timeout_seconds <= 0:
                raise ValueError(
                    "timeout_seconds harus > 0."
                )

    @property
    def is_async(self) -> bool:
        """
        Menentukan apakah callable merupakan async function.
        """

        return inspect.iscoroutinefunction(
            self.function
        )

    @property
    def success_rate(self) -> float:
        if self.execution_count <= 0:
            return 0.0

        return round(
            (
                self.success_count
                / self.execution_count
            )
            * 100.0,
            2,
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permissions": list(self.permissions),
            "schema": self.schema,
            "enabled": self.enabled,
            "whitelist_required": self.whitelist_required,
            "category": self.category,
            "version": self.version,
            "tags": list(self.tags),
            "is_async": self.is_async,
            "timeout_seconds": self.timeout_seconds,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
            "last_execution_at": self.last_execution_at,
            "metadata": dict(self.metadata),
        }


# ============================================================
# TOOL EXECUTION RECORD
# ============================================================

@dataclass(slots=True)
class ToolExecutionRecord:
    """
    Catatan setiap eksekusi tool.
    """

    execution_id: str

    tool: str

    status: ToolExecutionStatus

    arguments: ToolArguments = field(
        default_factory=dict
    )

    response: Any = None

    data: Any = None

    error: str | None = None

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    completed_at: str | None = None

    latency_ms: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    permission_status: ToolPermissionStatus = (
        ToolPermissionStatus.ALLOWED
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "tool": self.tool,
            "status": self.status.value,
            "arguments": dict(self.arguments),
            "response": self.response,
            "data": self.data,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
            "permission_status": (
                self.permission_status.value
            ),
        }


# ============================================================
# TOOL RESULT
# ============================================================

@dataclass(slots=True)
class ToolResult:
    """
    Hasil eksekusi tool ZAI.
    """

    success: bool

    tool: str

    status: ToolExecutionStatus

    response: Any = None

    data: Any = None

    error: str | None = None

    execution_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    completed_at: str | None = None

    latency_ms: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def complete(
        self,
        response: Any = None,
        *,
        data: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        self.success = True

        self.status = (
            ToolExecutionStatus.COMPLETED
        )

        self.response = response

        self.data = data

        if metadata:
            self.metadata.update(
                dict(metadata)
            )

        self.completed_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        return self

    def fail(
        self,
        error: str,
        *,
        status: ToolExecutionStatus = (
            ToolExecutionStatus.FAILED
        ),
    ) -> ToolResult:
        self.success = False

        self.status = status

        self.error = str(error)

        self.completed_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        return self

    def warn(
        self,
        message: str,
    ) -> ToolResult:
        self.warnings.append(
            str(message)
        )

        return self

    def add_warning(
        self,
        message: str,
    ) -> ToolResult:
        return self.warn(message)

    @property
    def has_error(self) -> bool:
        return bool(self.error)

    @property
    def is_completed(self) -> bool:
        return (
            self.status
            == ToolExecutionStatus.COMPLETED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tool": self.tool,
            "status": self.status.value,
            "response": self.response,
            "data": self.data,
            "error": self.error,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
        }


# ============================================================
# TOOL REGISTRY
# ============================================================

class ToolRegistry:
    """
    Central registry untuk seluruh tool ZAI.

    Fitur:

    - register_function
    - register
    - unregister
    - get
    - has
    - names
    - list_tools
    - whitelist
    - remove_whitelist
    - is_whitelisted
    - permission validation
    - schema validation
    - sync execution
    - async execution
    - execution history
    - statistics
    - health
    """

    VERSION = "2.0.0"

    DEFAULT_PERMISSION = "read"

    def __init__(
        self,
        *,
        strict_permissions: bool = False,
        strict_schema: bool = True,
        max_history: int = 1000,
    ) -> None:
        self._tools: dict[
            str,
            ToolDefinition,
        ] = {}

        self._whitelist: set[str] = set()

        self._history: list[
            ToolExecutionRecord
        ] = []

        self.strict_permissions = (
            strict_permissions
        )

        self.strict_schema = strict_schema

        self.max_history = max(
            1,
            int(max_history),
        )

        self.execution_count = 0

        self.success_count = 0

        self.failure_count = 0

        self.denied_count = 0

        self.blocked_count = 0

        self.timeout_count = 0

    # ========================================================
    # REGISTRATION
    # ========================================================

    def register(
        self,
        definition: ToolDefinition,
        *,
        overwrite: bool = False,
    ) -> ToolDefinition:
        if not isinstance(
            definition,
            ToolDefinition,
        ):
            raise TypeError(
                "definition harus ToolDefinition."
            )

        if (
            definition.name in self._tools
            and not overwrite
        ):
            raise ValueError(
                f"Tool '{definition.name}' "
                "sudah terdaftar."
            )

        self._tools[
            definition.name
        ] = definition

        return definition

    def register_function(
        self,
        name: str,
        function: ToolCallable,
        *,
        description: str = "",
        permissions: Iterable[str] | None = None,
        schema: ToolSchema | None = None,
        enabled: bool = True,
        whitelist_required: bool = False,
        category: str = "general",
        version: str = "1.0.0",
        tags: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
        allow_positional_arguments: bool = False,
        overwrite: bool = False,
    ) -> ToolDefinition:
        definition = ToolDefinition(
            name=name,
            function=function,
            description=description,
            permissions=list(
                permissions or []
            ),
            schema=dict(
                schema or {}
            ),
            enabled=enabled,
            whitelist_required=(
                whitelist_required
            ),
            category=category,
            version=version,
            tags=list(tags or []),
            metadata=dict(
                metadata or {}
            ),
            timeout_seconds=(
                timeout_seconds
            ),
            allow_positional_arguments=(
                allow_positional_arguments
            ),
        )

        return self.register(
            definition,
            overwrite=overwrite,
        )

    def unregister(
        self,
        name: str,
    ) -> bool:
        if name not in self._tools:
            return False

        del self._tools[name]

        self._whitelist.discard(name)

        return True

    # ========================================================
    # LOOKUP
    # ========================================================

    def get(
        self,
        name: str,
    ) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(
                f"Tool '{name}' tidak terdaftar."
            ) from exc

    def has(
        self,
        name: str,
    ) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(
            self._tools.keys()
        )

    def list_tools(self) -> list[
        dict[str, Any]
    ]:
        return [
            tool.info()
            for tool in self._tools.values()
        ]

    def active_tools(self) -> list[
        dict[str, Any]
    ]:
        return [
            tool.info()
            for tool in self._tools.values()
            if tool.enabled
        ]

    # ========================================================
    # WHITELIST
    # ========================================================

    def whitelist(
        self,
        name: str,
    ) -> bool:
        if not self.has(name):
            raise KeyError(
                f"Tool '{name}' tidak terdaftar."
            )

        self._whitelist.add(name)

        return True

    def remove_whitelist(
        self,
        name: str,
    ) -> bool:
        if name not in self._whitelist:
            return False

        self._whitelist.remove(name)

        return True

    def is_whitelisted(
        self,
        name: str,
    ) -> bool:
        return name in self._whitelist

    def whitelist_all(self) -> None:
        self._whitelist = set(
            self._tools.keys()
        )

    def clear_whitelist(self) -> None:
        self._whitelist.clear()

    def whitelisted_names(self) -> list[str]:
        return sorted(
            self._whitelist
        )

    # ========================================================
    # PERMISSION
    # ========================================================

    def check_permission(
        self,
        name: str,
        permissions: Iterable[str] | None = None,
    ) -> ToolPermissionStatus:
        tool = self.get(name)

        if not tool.permissions:
            return ToolPermissionStatus.ALLOWED

        provided = {
            str(permission)
            for permission in (
                permissions or []
            )
        }

        required = set(
            tool.permissions
        )

        if required.issubset(provided):
            return ToolPermissionStatus.ALLOWED

        return ToolPermissionStatus.DENIED

    # ========================================================
    # SCHEMA VALIDATION
    # ========================================================

    def validate_arguments(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
    ) -> tuple[bool, str | None]:
        tool = self.get(name)

        args = dict(
            arguments or {}
        )

        schema = tool.schema

        if not schema:
            return True, None

        schema_type = schema.get(
            "type"
        )

        if (
            schema_type
            and schema_type != "object"
        ):
            return (
                False,
                (
                    "Schema root tool harus "
                    "bertipe object."
                ),
            )

        properties = schema.get(
            "properties",
            {},
        )

        if not isinstance(
            properties,
            dict,
        ):
            return (
                False,
                "schema.properties harus object.",
            )

        required = schema.get(
            "required",
            [],
        )

        if not isinstance(
            required,
            list,
        ):
            return (
                False,
                "schema.required harus list.",
            )

        for key in required:
            if key not in args:
                return (
                    False,
                    (
                        f"Argument wajib "
                        f"'{key}' tidak ada."
                    ),
                )

        if schema.get(
            "additionalProperties",
            True,
        ) is False:
            unknown = [
                key
                for key in args
                if key not in properties
            ]

            if unknown:
                return (
                    False,
                    (
                        "Argument tidak dikenal: "
                        + ", ".join(
                            map(
                                str,
                                unknown,
                            )
                        )
                    ),
                )

        for key, value in args.items():
            if key not in properties:
                continue

            property_schema = properties[
                key
            ]

            if not isinstance(
                property_schema,
                dict,
            ):
                continue

            valid, error = (
                self._validate_value(
                    value,
                    property_schema,
                    path=key,
                )
            )

            if not valid:
                return False, error

        return True, None

    def _validate_value(
        self,
        value: Any,
        schema: Mapping[str, Any],
        *,
        path: str,
    ) -> tuple[bool, str | None]:
        expected = schema.get(
            "type"
        )

        if expected is None:
            return True, None

        valid = True

        if expected == "string":
            valid = isinstance(
                value,
                str,
            )

        elif expected == "integer":
            valid = (
                isinstance(
                    value,
                    int,
                )
                and not isinstance(
                    value,
                    bool,
                )
            )

        elif expected == "number":
            valid = (
                isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
                and not isinstance(
                    value,
                    bool,
                )
            )

        elif expected == "boolean":
            valid = isinstance(
                value,
                bool,
            )

        elif expected == "array":
            valid = isinstance(
                value,
                list,
            )

        elif expected == "object":
            valid = isinstance(
                value,
                dict,
            )

        elif expected == "null":
            valid = value is None

        elif expected == "any":
            valid = True

        if not valid:
            return (
                False,
                (
                    f"Argument '{path}' "
                    f"harus bertipe "
                    f"'{expected}'."
                ),
            )

        if (
            "enum" in schema
            and value not in schema["enum"]
        ):
            return (
                False,
                (
                    f"Argument '{path}' "
                    "memiliki nilai yang "
                    "tidak diperbolehkan."
                ),
            )

        if (
            isinstance(value, str)
            and "minLength" in schema
            and len(value)
            < int(schema["minLength"])
        ):
            return (
                False,
                (
                    f"Argument '{path}' "
                    "terlalu pendek."
                ),
            )

        if (
            isinstance(value, str)
            and "maxLength" in schema
            and len(value)
            > int(schema["maxLength"])
        ):
            return (
                False,
                (
                    f"Argument '{path}' "
                    "terlalu panjang."
                ),
            )

        if (
            isinstance(value, (int, float))
            and "minimum" in schema
            and value < schema["minimum"]
        ):
            return (
                False,
                (
                    f"Argument '{path}' "
                    "lebih kecil dari minimum."
                ),
            )

        if (
            isinstance(value, (int, float))
            and "maximum" in schema
            and value > schema["maximum"]
        ):
            return (
                False,
                (
                    f"Argument '{path}' "
                    "lebih besar dari maximum."
                ),
            )

        return True, None

    # ========================================================
    # INTERNAL RESULT
    # ========================================================

    def _new_result(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
    ) -> ToolResult:
        return ToolResult(
            success=False,
            tool=name,
            status=(
                ToolExecutionStatus.PENDING
            ),
            metadata={
                "registry_version": (
                    self.VERSION
                ),
            },
        )

    # ========================================================
    # HISTORY
    # ========================================================

    def _record(
        self,
        result: ToolResult,
        arguments: Mapping[str, Any] | None,
    ) -> None:
        record = ToolExecutionRecord(
            execution_id=(
                result.execution_id
            ),
            tool=result.tool,
            status=result.status,
            arguments=dict(
                arguments or {}
            ),
            response=result.response,
            data=result.data,
            error=result.error,
            created_at=result.created_at,
            completed_at=result.completed_at,
            latency_ms=result.latency_ms,
            metadata=dict(
                result.metadata
            ),
            warnings=list(
                result.warnings
            ),
        )

        self._history.append(record)

        if len(self._history) > self.max_history:
            del self._history[
                : len(self._history)
                - self.max_history
            ]

    def history(
        self,
        *,
        tool: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self._history

        if tool is not None:
            records = [
                record
                for record in records
                if record.tool == tool
            ]

        if limit is not None:
            limit = max(
                0,
                int(limit),
            )

            if limit == 0:
                records = []

            else:
                records = records[
                    -limit:
                ]

        return [
            record.to_dict()
            for record in records
        ]

    def clear_history(self) -> None:
        self._history.clear()

    # ========================================================
    # SYNC EXECUTION
    # ========================================================

    def execute(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        permissions: Iterable[str] | None = None,
        positional: Iterable[Any] | None = None,
    ) -> ToolResult:
        started = perf_counter()

        args = dict(
            arguments or {}
        )

        result = self._new_result(
            name,
            args,
        )

        self.execution_count += 1

        try:
            tool = self.get(name)

            result.status = (
                ToolExecutionStatus.RUNNING
            )

            result.metadata.update(
                {
                    "tool_version": tool.version,
                    "category": tool.category,
                    "async_tool": tool.is_async,
                }
            )

            if not tool.enabled:
                self.blocked_count += 1

                result.fail(
                    (
                        f"Tool '{name}' "
                        "sedang disabled."
                    ),
                    status=(
                        ToolExecutionStatus.BLOCKED
                    ),
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            if (
                tool.whitelist_required
                and not self.is_whitelisted(
                    name
                )
            ):
                self.denied_count += 1

                result.fail(
                    (
                        f"Tool '{name}' "
                        "belum masuk whitelist."
                    ),
                    status=(
                        ToolExecutionStatus.DENIED
                    ),
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            permission_status = (
                self.check_permission(
                    name,
                    permissions,
                )
            )

            if (
                self.strict_permissions
                and permission_status
                != ToolPermissionStatus.ALLOWED
            ):
                self.denied_count += 1

                result.fail(
                    (
                        f"Permission tool "
                        f"'{name}' ditolak."
                    ),
                    status=(
                        ToolExecutionStatus.DENIED
                    ),
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            if self.strict_schema:
                valid, error = (
                    self.validate_arguments(
                        name,
                        args,
                    )
                )

                if not valid:
                    result.fail(
                        error
                        or "Argument tidak valid."
                    )

                    return self._finish(
                        result,
                        args,
                        started,
                        success=False,
                    )

            positional_args = list(
                positional or []
            )

            if positional_args and not (
                tool.allow_positional_arguments
            ):
                result.fail(
                    (
                        f"Tool '{name}' "
                        "tidak mengizinkan "
                        "positional arguments."
                    )
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            # ------------------------------------------------
            # ASYNC FUNCTION FROM SYNC API
            # ------------------------------------------------

            if tool.is_async:
                try:
                    running_loop = (
                        asyncio.get_running_loop()
                    )

                except RuntimeError:
                    running_loop = None

                if running_loop is not None:
                    result.fail(
                        (
                            "Tool async tidak dapat "
                            "dijalankan melalui "
                            "execute() saat event loop "
                            "sedang aktif. Gunakan "
                            "execute_async()."
                        )
                    )

                    return self._finish(
                        result,
                        args,
                        started,
                        success=False,
                    )

                value = asyncio.run(
                    self._call_async(
                        tool,
                        positional_args,
                        args,
                    )
                )

            else:
                value = tool.function(
                    *positional_args,
                    **args,
                )

            result.complete(
                response=value,
                data=value,
            )

            return self._finish(
                result,
                args,
                started,
                success=True,
            )

        except Exception as exc:
            result.fail(
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

            return self._finish(
                result,
                args,
                started,
                success=False,
            )

    # ========================================================
    # ASYNC EXECUTION
    # ========================================================

    async def execute_async(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        permissions: Iterable[str] | None = None,
        positional: Iterable[Any] | None = None,
    ) -> ToolResult:
        started = perf_counter()

        args = dict(
            arguments or {}
        )

        result = self._new_result(
            name,
            args,
        )

        self.execution_count += 1

        try:
            tool = self.get(name)

            result.status = (
                ToolExecutionStatus.RUNNING
            )

            result.metadata.update(
                {
                    "tool_version": tool.version,
                    "category": tool.category,
                    "async_tool": tool.is_async,
                }
            )

            if not tool.enabled:
                self.blocked_count += 1

                result.fail(
                    (
                        f"Tool '{name}' "
                        "sedang disabled."
                    ),
                    status=(
                        ToolExecutionStatus.BLOCKED
                    ),
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            if (
                tool.whitelist_required
                and not self.is_whitelisted(
                    name
                )
            ):
                self.denied_count += 1

                result.fail(
                    (
                        f"Tool '{name}' "
                        "belum masuk whitelist."
                    ),
                    status=(
                        ToolExecutionStatus.DENIED
                    ),
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            permission_status = (
                self.check_permission(
                    name,
                    permissions,
                )
            )

            if (
                self.strict_permissions
                and permission_status
                != ToolPermissionStatus.ALLOWED
            ):
                self.denied_count += 1

                result.fail(
                    (
                        f"Permission tool "
                        f"'{name}' ditolak."
                    ),
                    status=(
                        ToolExecutionStatus.DENIED
                    ),
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            if self.strict_schema:
                valid, error = (
                    self.validate_arguments(
                        name,
                        args,
                    )
                )

                if not valid:
                    result.fail(
                        error
                        or "Argument tidak valid."
                    )

                    return self._finish(
                        result,
                        args,
                        started,
                        success=False,
                    )

            positional_args = list(
                positional or []
            )

            if positional_args and not (
                tool.allow_positional_arguments
            ):
                result.fail(
                    (
                        f"Tool '{name}' "
                        "tidak mengizinkan "
                        "positional arguments."
                    )
                )

                return self._finish(
                    result,
                    args,
                    started,
                    success=False,
                )

            if tool.is_async:
                value = await self._call_async(
                    tool,
                    positional_args,
                    args,
                )

            else:
                value = await asyncio.to_thread(
                    tool.function,
                    *positional_args,
                    **args,
                )

            result.complete(
                response=value,
                data=value,
            )

            return self._finish(
                result,
                args,
                started,
                success=True,
            )

        except asyncio.TimeoutError:
            self.timeout_count += 1

            result.fail(
                (
                    f"Tool '{name}' "
                    "timeout."
                ),
                status=(
                    ToolExecutionStatus.TIMEOUT
                ),
            )

            return self._finish(
                result,
                args,
                started,
                success=False,
            )

        except Exception as exc:
            result.fail(
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

            return self._finish(
                result,
                args,
                started,
                success=False,
            )

    async def _call_async(
        self,
        tool: ToolDefinition,
        positional: list[Any],
        arguments: dict[str, Any],
    ) -> Any:
        result = tool.function(
            *positional,
            **arguments,
        )

        if inspect.isawaitable(result):
            if tool.timeout_seconds:
                return await asyncio.wait_for(
                    result,
                    timeout=(
                        tool.timeout_seconds
                    ),
                )

            return await result

        return result

    # ========================================================
    # FINISH
    # ========================================================

    def _finish(
        self,
        result: ToolResult,
        arguments: Mapping[str, Any],
        started: float,
        *,
        success: bool,
    ) -> ToolResult:
        result.latency_ms = round(
            (
                perf_counter()
                - started
            )
            * 1000.0,
            4,
        )

        result.metadata.update(
            {
                "latency_ms": (
                    result.latency_ms
                ),
                "execution_count": (
                    self.execution_count
                ),
                "success_count": (
                    self.success_count
                    + (1 if success else 0)
                ),
                "failure_count": (
                    self.failure_count
                    + (0 if success else 1)
                ),
            }
        )

        if success:
            self.success_count += 1

        else:
            self.failure_count += 1

        tool = self._tools.get(
            result.tool
        )

        if tool is not None:
            tool.execution_count += 1

            tool.last_execution_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            if success:
                tool.success_count += 1

            else:
                tool.failure_count += 1

        self._record(
            result,
            arguments,
        )

        return result

    # ========================================================
    # BATCH EXECUTION
    # ========================================================

    def execute_many(
        self,
        calls: Iterable[
            Mapping[str, Any]
        ],
    ) -> list[ToolResult]:
        results: list[ToolResult] = []

        for call in calls:
            call_data = dict(call)

            name = call_data.pop(
                "name",
                call_data.pop(
                    "tool",
                    None,
                ),
            )

            if not name:
                result = ToolResult(
                    success=False,
                    tool="<unknown>",
                    status=(
                        ToolExecutionStatus.FAILED
                    ),
                )

                result.fail(
                    "Nama tool tidak diberikan."
                )

                results.append(result)

                continue

            arguments = call_data.pop(
                "arguments",
                {},
            )

            permissions = call_data.pop(
                "permissions",
                None,
            )

            positional = call_data.pop(
                "positional",
                None,
            )

            results.append(
                self.execute(
                    name,
                    arguments=arguments,
                    permissions=permissions,
                    positional=positional,
                )
            )

        return results

    async def execute_many_async(
        self,
        calls: Iterable[
            Mapping[str, Any]
        ],
    ) -> list[ToolResult]:
        async def execute_one(
            call: Mapping[str, Any],
        ) -> ToolResult:
            call_data = dict(call)

            name = call_data.pop(
                "name",
                call_data.pop(
                    "tool",
                    None,
                ),
            )

            if not name:
                result = ToolResult(
                    success=False,
                    tool="<unknown>",
                    status=(
                        ToolExecutionStatus.FAILED
                    ),
                )

                return result.fail(
                    "Nama tool tidak diberikan."
                )

            return await self.execute_async(
                name,
                arguments=call_data.pop(
                    "arguments",
                    {},
                ),
                permissions=call_data.pop(
                    "permissions",
                    None,
                ),
                positional=call_data.pop(
                    "positional",
                    None,
                ),
            )

        return await asyncio.gather(
            *[
                execute_one(call)
                for call in calls
            ]
        )

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        name: str,
    ) -> bool:
        tool = self.get(name)

        tool.enabled = True

        return True

    def disable(
        self,
        name: str,
    ) -> bool:
        tool = self.get(name)

        tool.enabled = False

        return True

    # ========================================================
    # STATISTICS
    # ========================================================

    @property
    def success_rate(self) -> float:
        if self.execution_count <= 0:
            return 0.0

        return round(
            (
                self.success_count
                / self.execution_count
            )
            * 100.0,
            2,
        )

    @property
    def failure_rate(self) -> float:
        if self.execution_count <= 0:
            return 0.0

        return round(
            (
                self.failure_count
                / self.execution_count
            )
            * 100.0,
            2,
        )

    def statistics(self) -> dict[str, Any]:
        return {
            "registry_version": (
                self.VERSION
            ),
            "total_tools": len(
                self._tools
            ),
            "active_tools": len(
                [
                    tool
                    for tool
                    in self._tools.values()
                    if tool.enabled
                ]
            ),
            "whitelisted_tools": len(
                self._whitelist
            ),
            "execution_count": (
                self.execution_count
            ),
            "success_count": (
                self.success_count
            ),
            "failure_count": (
                self.failure_count
            ),
            "denied_count": (
                self.denied_count
            ),
            "blocked_count": (
                self.blocked_count
            ),
            "timeout_count": (
                self.timeout_count
            ),
            "success_rate": (
                self.success_rate
            ),
            "failure_rate": (
                self.failure_rate
            ),
            "history_size": len(
                self._history
            ),
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> dict[str, Any]:
        return {
            "registry": "ToolRegistry",
            "version": self.VERSION,
            "status": "READY",
            "tools": self.list_tools(),
            "tool_names": self.names(),
            "whitelist": (
                self.whitelisted_names()
            ),
            "statistics": self.statistics(),
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health(self) -> dict[str, Any]:
        return {
            "registry": "ToolRegistry",
            "version": self.VERSION,
            "status": "HEALTHY",
            "total_tools": len(
                self._tools
            ),
            "active_tools": len(
                [
                    tool
                    for tool
                    in self._tools.values()
                    if tool.enabled
                ]
            ),
            "execution_count": (
                self.execution_count
            ),
            "success_count": (
                self.success_count
            ),
            "failure_count": (
                self.failure_count
            ),
            "success_rate": (
                self.success_rate
            ),
        }


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "AsyncToolCallable",
    "SyncToolCallable",
    "ToolCallable",
    "ToolArguments",
    "ToolSchema",
    "ToolPermission",
    "ToolExecutionStatus",
    "ToolPermissionStatus",
    "ToolDefinition",
    "ToolExecutionRecord",
    "ToolResult",
    "ToolRegistry",
]
from __future__ import annotations

"""
ZAI Tool Execution Loop
=======================

Modul ini menjadi jembatan antara ZAIBrain dan Tool Platform.

Arsitektur:

    ZAIBrain
       |
       v
    ToolExecutionLoop
       |
       +--> ToolPolicy
       |
       +--> ToolCallParser
       |
       +--> ToolExecutorAdapter
       |
       v
    ToolManager / ToolRegistry
       |
       v
    ToolResult
       |
       v
    Observation
       |
       v
    ZAIBrain

Tujuan utama:

1. Menormalisasi tool call.
2. Memvalidasi nama tool.
3. Memvalidasi argument.
4. Memeriksa permission.
5. Membatasi jumlah langkah.
6. Mendeteksi loop tool.
7. Menjalankan tool secara async jika tersedia.
8. Mendukung executor sync.
9. Menormalisasi berbagai bentuk ToolResult.
10. Menghasilkan observation yang dapat dikirim kembali ke Brain.
11. Menyimpan execution history.
12. Menyediakan statistik runtime.
13. Tidak melakukan arbitrary code execution.
14. Tidak mengakses infrastructure secara langsung.
15. Menggunakan dependency injection.

Modul ini sengaja dibuat sebagai adapter layer sehingga ZAIBrain
tidak harus mengetahui detail internal ToolManager atau ToolRegistry.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from inspect import isawaitable
from time import perf_counter
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)
from uuid import uuid4


# ============================================================================
# CONSTANTS
# ============================================================================

MODULE_NAME = "ToolExecutionLoop"
MODULE_VERSION = "1.0.0"

DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_TOOL_CALLS = 16
DEFAULT_HISTORY_LIMIT = 100
DEFAULT_MAX_ARGUMENTS = 64
DEFAULT_MAX_TOOL_NAME_LENGTH = 128
DEFAULT_MAX_ARGUMENT_STRING_LENGTH = 12000

STATUS_READY = "READY"
STATUS_HEALTHY = "HEALTHY"

EXECUTION_COMPLETED = "completed"
EXECUTION_FAILED = "failed"
EXECUTION_DENIED = "denied"
EXECUTION_BLOCKED = "blocked"
EXECUTION_TIMEOUT = "timeout"

DEFAULT_TOOL_CATEGORY = "general"


# ============================================================================
# HELPERS
# ============================================================================


def utc_now() -> str:
    """
    Menghasilkan timestamp UTC dalam ISO-8601.

    Returns:
        Timestamp UTC.
    """
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    """
    Normalisasi nilai menjadi string aman.

    Args:
        value: Nilai input.

    Returns:
        String yang sudah di-strip.
    """
    if value is None:
        return ""

    return str(value).strip()


def normalize_tool_name(value: Any) -> str:
    """
    Normalisasi nama tool.

    Args:
        value: Nama tool.

    Returns:
        Nama tool lowercase.

    Raises:
        ValueError: Jika nama tool kosong atau terlalu panjang.
    """
    name = normalize_text(value).lower()

    if not name:
        raise ValueError("Nama tool tidak boleh kosong.")

    if len(name) > DEFAULT_MAX_TOOL_NAME_LENGTH:
        raise ValueError(
            "Nama tool terlalu panjang."
        )

    return name


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Konversi nilai ke float secara defensif.

    Args:
        value: Nilai input.
        default: Nilai fallback.

    Returns:
        Float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Konversi nilai ke integer secara defensif.

    Args:
        value: Nilai input.
        default: Nilai fallback.

    Returns:
        Integer.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clone_dict(value: Any) -> dict[str, Any]:
    """
    Mengubah mapping menjadi dictionary baru.

    Args:
        value: Mapping atau nilai lain.

    Returns:
        Dictionary.
    """
    if isinstance(value, Mapping):
        return dict(value)

    return {}


# ============================================================================
# PROTOCOLS
# ============================================================================


@runtime_checkable
class AsyncToolExecutor(Protocol):
    """
    Protocol executor async.

    ToolManager atau adapter eksternal dapat memenuhi protocol ini
    tanpa harus mewarisi class tertentu.
    """

    async def execute_async(
        self,
        tool: str,
        arguments: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        ...


@runtime_checkable
class SyncToolExecutor(Protocol):
    """
    Protocol executor sync.
    """

    def execute(
        self,
        tool: str,
        arguments: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        ...


@runtime_checkable
class ToolAvailabilityProvider(Protocol):
    """
    Protocol untuk provider yang dapat memberitahu daftar tool.
    """

    def names(self) -> Sequence[str]:
        ...


# ============================================================================
# DATA TRANSFER OBJECTS
# ============================================================================


@dataclass(slots=True)
class ToolCall:
    """
    Representasi satu permintaan eksekusi tool.

    Attributes:
        name: Nama tool.
        arguments: Argument tool.
        call_id: ID unik.
        step: Nomor langkah.
        source: Sumber tool call.
        metadata: Metadata tambahan.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid4()))
    step: int = 1
    source: str = "brain"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = normalize_tool_name(self.name)

        if not isinstance(self.arguments, dict):
            self.arguments = dict(self.arguments or {})

        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata or {})

    def signature(self) -> str:
        """
        Menghasilkan signature untuk loop detection.

        Returns:
            Signature tool + arguments.
        """
        return (
            f"{self.name}:"
            f"{repr(sorted(self.arguments.items(), key=lambda item: item[0]))}"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize DTO.

        Returns:
            Dictionary.
        """
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
            "step": self.step,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ToolObservation:
    """
    Observation hasil eksekusi tool.

    Observation adalah format yang dikembalikan ke Brain setelah
    sebuah tool selesai dijalankan.
    """

    call_id: str
    tool: str
    success: bool
    status: str
    response: Any = None
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    step: int = 1
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize observation.

        Returns:
            Dictionary.
        """
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "success": self.success,
            "status": self.status,
            "response": self.response,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "step": self.step,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ToolExecutionRecord:
    """
    History record satu tool execution.
    """

    execution_id: str
    call: ToolCall
    observation: ToolObservation

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize execution record.

        Returns:
            Dictionary.
        """
        return {
            "execution_id": self.execution_id,
            "call": self.call.to_dict(),
            "observation": self.observation.to_dict(),
        }


@dataclass(slots=True)
class ToolLoopResult:
    """
    Hasil satu execution loop.

    Satu loop dapat menjalankan lebih dari satu tool call.
    """

    success: bool
    status: str
    observations: list[ToolObservation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    completed_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """
        Mengecek apakah loop gagal.

        Returns:
            True jika gagal.
        """
        return not self.success

    def add_observation(
        self,
        observation: ToolObservation,
    ) -> None:
        """
        Menambahkan observation.

        Args:
            observation: Observation tool.
        """
        self.observations.append(observation)

    def add_error(self, error: str) -> None:
        """
        Menambahkan error.

        Args:
            error: Pesan error.
        """
        message = normalize_text(error)

        if message:
            self.errors.append(message)

    def add_warning(self, warning: str) -> None:
        """
        Menambahkan warning.

        Args:
            warning: Pesan warning.
        """
        message = normalize_text(warning)

        if message:
            self.warnings.append(message)

    def finalize(
        self,
        *,
        success: bool,
        status: str,
        latency_ms: float,
    ) -> None:
        """
        Finalisasi loop result.

        Args:
            success: Status sukses.
            status: Status akhir.
            latency_ms: Latency total.
        """
        self.success = success
        self.status = status
        self.latency_ms = round(latency_ms, 4)
        self.completed_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize loop result.

        Returns:
            Dictionary.
        """
        return {
            "success": self.success,
            "status": self.status,
            "observations": [
                item.to_dict()
                for item in self.observations
            ],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "latency_ms": self.latency_ms,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# TOOL POLICY
# ============================================================================


@dataclass(slots=True)
class ToolPolicy:
    """
    Policy keamanan dan resource untuk ToolExecutionLoop.

    Policy tidak mengeksekusi tool.
    Policy hanya menentukan apakah execution diperbolehkan.
    """

    max_steps: int = DEFAULT_MAX_STEPS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_arguments: int = DEFAULT_MAX_ARGUMENTS
    max_argument_string_length: int = DEFAULT_MAX_ARGUMENT_STRING_LENGTH

    require_whitelist: bool = False
    allow_unknown_tools: bool = False
    allow_repeated_calls: bool = False

    blocked_tools: set[str] = field(default_factory=set)
    allowed_tools: Optional[set[str]] = None

    def __post_init__(self) -> None:
        self.max_steps = max(1, int(self.max_steps))
        self.max_tool_calls = max(1, int(self.max_tool_calls))
        self.max_arguments = max(1, int(self.max_arguments))

        self.blocked_tools = {
            normalize_tool_name(item)
            for item in self.blocked_tools
            if normalize_text(item)
        }

        if self.allowed_tools is not None:
            self.allowed_tools = {
                normalize_tool_name(item)
                for item in self.allowed_tools
                if normalize_text(item)
            }

    def check_tool_name(
        self,
        tool_name: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Memeriksa apakah tool diperbolehkan.

        Args:
            tool_name: Nama tool.

        Returns:
            Tuple allowed + error.
        """
        name = normalize_tool_name(tool_name)

        if name in self.blocked_tools:
            return (
                False,
                f"Tool '{name}' diblokir oleh policy.",
            )

        if (
            self.allowed_tools is not None
            and name not in self.allowed_tools
        ):
            return (
                False,
                f"Tool '{name}' tidak masuk allowed tools.",
            )

        return True, None

    def check_arguments(
        self,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Memeriksa ukuran argument.

        Args:
            arguments: Argument tool.

        Returns:
            Tuple valid + error.
        """
        if len(arguments) > self.max_arguments:
            return (
                False,
                (
                    "Jumlah argument tool melebihi batas "
                    f"{self.max_arguments}."
                ),
            )

        for key, value in arguments.items():
            if len(normalize_text(key)) > 256:
                return (
                    False,
                    "Nama argument terlalu panjang.",
                )

            if isinstance(value, str):
                if len(value) > self.max_argument_string_length:
                    return (
                        False,
                        (
                            f"Argument '{key}' terlalu panjang."
                        ),
                    )

        return True, None

    def check_step(
        self,
        step: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Memeriksa batas langkah.

        Args:
            step: Nomor langkah.

        Returns:
            Tuple allowed + error.
        """
        if step > self.max_steps:
            return (
                False,
                (
                    "Batas maksimum step tool tercapai: "
                    f"{self.max_steps}."
                ),
            )

        return True, None

    def check_call_count(
        self,
        count: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Memeriksa jumlah tool call.

        Args:
            count: Jumlah call.

        Returns:
            Tuple allowed + error.
        """
        if count >= self.max_tool_calls:
            return (
                False,
                (
                    "Batas maksimum tool call tercapai: "
                    f"{self.max_tool_calls}."
                ),
            )

        return True, None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize policy.

        Returns:
            Dictionary.
        """
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_arguments": self.max_arguments,
            "max_argument_string_length": (
                self.max_argument_string_length
            ),
            "require_whitelist": self.require_whitelist,
            "allow_unknown_tools": self.allow_unknown_tools,
            "allow_repeated_calls": self.allow_repeated_calls,
            "blocked_tools": sorted(self.blocked_tools),
            "allowed_tools": (
                sorted(self.allowed_tools)
                if self.allowed_tools is not None
                else None
            ),
        }


# ============================================================================
# TOOL CALL PARSER
# ============================================================================


class ToolCallParser:
    """
    Parser normalisasi tool call.

    Input yang didukung:

        ToolCall(...)
        {"name": "calculator", "arguments": {...}}
        {"tool": "calculator", "arguments": {...}}
        {"function": {"name": "...", "arguments": {...}}}
        {"function": {"name": "...", "arguments": "...JSON..."}}
        object dengan attribute name/tool/arguments

    Parser tidak mengeksekusi tool.
    """

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        step: int = 1,
        source: str = "brain",
    ) -> ToolCall:
        """
        Parse satu tool call.

        Args:
            value: Input tool call.
            step: Step execution.
            source: Sumber call.

        Returns:
            ToolCall.

        Raises:
            ValueError: Jika format tidak valid.
        """
        if isinstance(value, ToolCall):
            value.step = step
            value.source = source
            return value

        payload = cls._mapping_from_value(value)

        if "function" in payload:
            function = payload.get("function")

            if isinstance(function, Mapping):
                merged = dict(function)
                merged.update(
                    {
                        key: item
                        for key, item in payload.items()
                        if key != "function"
                    }
                )
                payload = merged

        name = (
            payload.get("name")
            or payload.get("tool")
            or payload.get("tool_name")
        )

        if not name:
            raise ValueError(
                "Tool call tidak memiliki nama tool."
            )

        arguments = (
            payload.get("arguments")
            or payload.get("args")
            or payload.get("parameters")
            or {}
        )

        arguments = cls._normalize_arguments(arguments)

        call_id = normalize_text(
            payload.get("call_id")
            or payload.get("id")
            or uuid4()
        )

        metadata = clone_dict(
            payload.get("metadata")
        )

        return ToolCall(
            name=normalize_tool_name(name),
            arguments=arguments,
            call_id=call_id,
            step=step,
            source=source,
            metadata=metadata,
        )

    @classmethod
    def parse_many(
        cls,
        values: Iterable[Any],
        *,
        step: int = 1,
        source: str = "brain",
    ) -> list[ToolCall]:
        """
        Parse banyak tool call.

        Args:
            values: Iterable input.
            step: Step.
            source: Sumber.

        Returns:
            List ToolCall.
        """
        calls: list[ToolCall] = []

        for value in values:
            calls.append(
                cls.parse(
                    value,
                    step=step,
                    source=source,
                )
            )

        return calls

    @staticmethod
    def _mapping_from_value(
        value: Any,
    ) -> dict[str, Any]:
        """
        Mengubah input object menjadi dictionary.

        Args:
            value: Input.

        Returns:
            Dictionary.

        Raises:
            ValueError: Jika input tidak valid.
        """
        if isinstance(value, Mapping):
            return dict(value)

        attributes = {}

        for key in (
            "name",
            "tool",
            "tool_name",
            "arguments",
            "args",
            "parameters",
            "call_id",
            "id",
            "metadata",
        ):
            if hasattr(value, key):
                attributes[key] = getattr(value, key)

        if not attributes:
            raise ValueError(
                "Format tool call tidak dikenali."
            )

        return attributes

    @staticmethod
    def _normalize_arguments(
        arguments: Any,
    ) -> dict[str, Any]:
        """
        Normalisasi argument.

        Args:
            arguments: Argument mentah.

        Returns:
            Dictionary argument.
        """
        if arguments is None:
            return {}

        if isinstance(arguments, Mapping):
            return dict(arguments)

        if isinstance(arguments, str):
            text = arguments.strip()

            if not text:
                return {}

            try:
                import json

                parsed = json.loads(text)

                if isinstance(parsed, Mapping):
                    return dict(parsed)

            except Exception:
                return {
                    "_raw_arguments": text,
                }

        try:
            return dict(arguments)
        except (TypeError, ValueError):
            return {
                "_raw_arguments": arguments,
            }


# ============================================================================
# TOOL EXECUTOR ADAPTER
# ============================================================================


class ToolExecutorAdapter:
    """
    Adapter universal untuk ToolManager / ToolRegistry.

    Prioritas method:

        1. execute_async(...)
        2. execute(...)
        3. callable executor

    Adapter tidak mengasumsikan implementasi internal tool platform.
    """

    def __init__(
        self,
        executor: Any,
    ) -> None:
        """
        Args:
            executor: ToolManager, ToolRegistry, callable, atau adapter.
        """
        if executor is None:
            raise ValueError(
                "Tool executor tidak boleh None."
            )

        self.executor = executor

    async def execute(
        self,
        call: ToolCall,
    ) -> Any:
        """
        Menjalankan satu ToolCall.

        Args:
            call: ToolCall.

        Returns:
            Raw tool result.
        """
        execute_async = getattr(
            self.executor,
            "execute_async",
            None,
        )

        if callable(execute_async):
            result = execute_async(
                call.name,
                arguments=dict(call.arguments),
            )

            if isawaitable(result):
                return await result

            return result

        execute = getattr(
            self.executor,
            "execute",
            None,
        )

        if callable(execute):
            result = execute(
                call.name,
                arguments=dict(call.arguments),
            )

            if isawaitable(result):
                return await result

            return result

        if callable(self.executor):
            result = self.executor(
                call.name,
                dict(call.arguments),
            )

            if isawaitable(result):
                return await result

            return result

        raise TypeError(
            "Executor tidak menyediakan execute_async, "
            "execute, atau callable interface."
        )

    def available_tools(self) -> list[str]:
        """
        Mencoba mengambil daftar tool.

        Returns:
            Nama tool yang tersedia.
        """
        candidates = (
            "names",
            "list_tools",
            "tools",
            "available_tools",
        )

        for method_name in candidates:
            method = getattr(
                self.executor,
                method_name,
                None,
            )

            if callable(method):
                try:
                    value = method()

                    if isinstance(value, Mapping):
                        return [
                            str(key)
                            for key in value.keys()
                        ]

                    if isinstance(value, Sequence):
                        return [
                            str(item)
                            for item in value
                        ]

                except Exception:
                    continue

        return []

    def has_tool(
        self,
        name: str,
    ) -> bool:
        """
        Mengecek availability tool jika executor mendukung.

        Args:
            name: Nama tool.

        Returns:
            True jika tersedia.
        """
        normalized = normalize_tool_name(name)

        method = getattr(
            self.executor,
            "has_tool",
            None,
        )

        if callable(method):
            try:
                return bool(
                    method(normalized)
                )
            except Exception:
                pass

        available = self.available_tools()

        if not available:
            return True

        normalized_available = {
            normalize_tool_name(item)
            for item in available
        }

        return normalized in normalized_available


# ============================================================================
# RESULT NORMALIZER
# ============================================================================


class ToolResultNormalizer:
    """
    Normalizer untuk berbagai format ToolResult.

    Tool platform ZAI dapat berkembang. Karena itu Brain tidak boleh
    bergantung pada satu bentuk object saja.
    """

    @classmethod
    def normalize(
        cls,
        raw: Any,
        call: ToolCall,
        latency_ms: float,
    ) -> ToolObservation:
        """
        Normalisasi raw result.

        Args:
            raw: Raw result.
            call: ToolCall.
            latency_ms: Latency.

        Returns:
            ToolObservation.
        """
        payload = cls._to_mapping(raw)

        success = cls._success(raw, payload)

        status = cls._status(
            raw,
            payload,
            success,
        )

        response = cls._value(
            raw,
            payload,
            "response",
        )

        data = cls._value(
            raw,
            payload,
            "data",
        )

        error = cls._value(
            raw,
            payload,
            "error",
        )

        if error is not None:
            error = normalize_text(error)

        metadata = clone_dict(
            cls._value(
                raw,
                payload,
                "metadata",
            )
        )

        return ToolObservation(
            call_id=call.call_id,
            tool=call.name,
            success=success,
            status=status,
            response=response,
            data=data,
            error=error,
            latency_ms=round(
                latency_ms,
                4,
            ),
            step=call.step,
            metadata=metadata,
        )

    @staticmethod
    def _to_mapping(
        raw: Any,
    ) -> dict[str, Any]:
        """
        Mengubah raw result ke mapping.

        Args:
            raw: Raw result.

        Returns:
            Dictionary.
        """
        if isinstance(raw, Mapping):
            return dict(raw)

        to_dict = getattr(
            raw,
            "to_dict",
            None,
        )

        if callable(to_dict):
            try:
                value = to_dict()

                if isinstance(value, Mapping):
                    return dict(value)

            except Exception:
                pass

        result: dict[str, Any] = {}

        for key in (
            "success",
            "status",
            "response",
            "data",
            "error",
            "metadata",
        ):
            if hasattr(raw, key):
                result[key] = getattr(
                    raw,
                    key,
                )

        return result

    @staticmethod
    def _value(
        raw: Any,
        payload: Mapping[str, Any],
        key: str,
    ) -> Any:
        """
        Mengambil field dari mapping atau object.
        """
        if key in payload:
            return payload[key]

        return getattr(
            raw,
            key,
            None,
        )

    @classmethod
    def _success(
        cls,
        raw: Any,
        payload: Mapping[str, Any],
    ) -> bool:
        """
        Menentukan success flag.
        """
        if "success" in payload:
            return bool(
                payload["success"]
            )

        status = normalize_text(
            cls._value(
                raw,
                payload,
                "status",
            )
        ).lower()

        if status in {
            EXECUTION_FAILED,
            EXECUTION_DENIED,
            EXECUTION_BLOCKED,
            EXECUTION_TIMEOUT,
        }:
            return False

        return True

    @classmethod
    def _status(
        cls,
        raw: Any,
        payload: Mapping[str, Any],
        success: bool,
    ) -> str:
        """
        Menentukan execution status.
        """
        status = normalize_text(
            cls._value(
                raw,
                payload,
                "status",
            )
        ).lower()

        if status:
            return status

        return (
            EXECUTION_COMPLETED
            if success
            else EXECUTION_FAILED
        )


# ============================================================================
# TOOL EXECUTION LOOP
# ============================================================================


class ToolExecutionLoop:
    """
    Execution loop untuk menghubungkan Brain dengan Tool Platform.

    Fitur:

        - single execution
        - batch execution
        - async execution
        - policy enforcement
        - loop detection
        - history
        - statistics
        - observation generation
        - error isolation
        - timeout-friendly adapter boundary
    """

    VERSION = MODULE_VERSION

    def __init__(
        self,
        executor: Any,
        *,
        policy: Optional[ToolPolicy] = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        whitelist: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Args:
            executor: ToolManager/ToolRegistry/executor.
            policy: Tool execution policy.
            history_limit: Batas history.
            whitelist: Daftar tool yang diizinkan.
        """
        self.adapter = ToolExecutorAdapter(
            executor
        )

        self.policy = (
            policy
            or ToolPolicy()
        )

        self.history_limit = max(
            1,
            int(history_limit),
        )

        self._history: list[
            ToolExecutionRecord
        ] = []

        self._active_signatures: set[str] = set()

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.denied_count = 0
        self.blocked_count = 0
        self.timeout_count = 0
        self.total_latency_ms = 0.0

        if whitelist is not None:
            self.policy.allowed_tools = {
                normalize_tool_name(item)
                for item in whitelist
                if normalize_text(item)
            }

    # ------------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """
        Informasi module.

        Returns:
            Dictionary info.
        """
        return {
            "loop": self.__class__.__name__,
            "version": self.VERSION,
            "status": STATUS_READY,
            "policy": self.policy.to_dict(),
            "history_size": len(self._history),
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "denied_count": self.denied_count,
            "blocked_count": self.blocked_count,
            "timeout_count": self.timeout_count,
            "success_rate": self.success_rate(),
            "available_tools": (
                self.adapter.available_tools()
            ),
        }

    # ------------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Health check.

        Returns:
            Health dictionary.
        """
        executor_ready = (
            self.adapter.executor is not None
        )

        return {
            "loop": self.__class__.__name__,
            "version": self.VERSION,
            "status": (
                STATUS_HEALTHY
                if executor_ready
                else "UNHEALTHY"
            ),
            "executor_ready": executor_ready,
            "history_size": len(self._history),
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
        }

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    def success_rate(self) -> float:
        """
        Success rate.

        Returns:
            Persentase sukses.
        """
        if self.execution_count <= 0:
            return 0.0

        return round(
            (
                self.success_count
                / self.execution_count
            )
            * 100.0,
            4,
        )

    def failure_rate(self) -> float:
        """
        Failure rate.

        Returns:
            Persentase gagal.
        """
        if self.execution_count <= 0:
            return 0.0

        return round(
            (
                self.failure_count
                / self.execution_count
            )
            * 100.0,
            4,
        )

    def average_latency_ms(self) -> float:
        """
        Average latency.

        Returns:
            Millisecond.
        """
        if self.execution_count <= 0:
            return 0.0

        return round(
            self.total_latency_ms
            / self.execution_count,
            4,
        )

    def stats(self) -> dict[str, Any]:
        """
        Statistik lengkap.

        Returns:
            Dictionary statistik.
        """
        return {
            "loop": self.__class__.__name__,
            "version": self.VERSION,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "denied_count": self.denied_count,
            "blocked_count": self.blocked_count,
            "timeout_count": self.timeout_count,
            "success_rate": self.success_rate(),
            "failure_rate": self.failure_rate(),
            "average_latency_ms": (
                self.average_latency_ms()
            ),
            "total_latency_ms": round(
                self.total_latency_ms,
                4,
            ),
            "history_size": len(self._history),
            "max_steps": self.policy.max_steps,
            "max_tool_calls": (
                self.policy.max_tool_calls
            ),
        }

    # ------------------------------------------------------------------------
    # SINGLE EXECUTION
    # ------------------------------------------------------------------------

    async def execute(
        self,
        tool_call: Any,
        *,
        step: int = 1,
        source: str = "brain",
    ) -> ToolObservation:
        """
        Eksekusi satu tool call.

        Args:
            tool_call: ToolCall atau raw payload.
            step: Step number.
            source: Sumber call.

        Returns:
            ToolObservation.
        """
        started = perf_counter()

        try:
            call = ToolCallParser.parse(
                tool_call,
                step=step,
                source=source,
            )
        except Exception as exc:
            latency = (
                perf_counter() - started
            ) * 1000.0

            self.execution_count += 1
            self.failure_count += 1
            self.total_latency_ms += latency

            return ToolObservation(
                call_id=str(uuid4()),
                tool="unknown",
                success=False,
                status=EXECUTION_FAILED,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                latency_ms=round(
                    latency,
                    4,
                ),
                step=step,
            )

        policy_error = self._validate_call(
            call
        )

        if policy_error is not None:
            latency = (
                perf_counter() - started
            ) * 1000.0

            observation = ToolObservation(
                call_id=call.call_id,
                tool=call.name,
                success=False,
                status=policy_error[0],
                error=policy_error[1],
                latency_ms=round(
                    latency,
                    4,
                ),
                step=step,
            )

            self._record_denial(
                observation
            )

            return observation

        try:
            raw_result = await self.adapter.execute(
                call
            )

            latency = (
                perf_counter() - started
            ) * 1000.0

            observation = (
                ToolResultNormalizer.normalize(
                    raw_result,
                    call,
                    latency,
                )
            )

        except TimeoutError as exc:
            latency = (
                perf_counter() - started
            ) * 1000.0

            observation = ToolObservation(
                call_id=call.call_id,
                tool=call.name,
                success=False,
                status=EXECUTION_TIMEOUT,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                latency_ms=round(
                    latency,
                    4,
                ),
                step=step,
            )

        except Exception as exc:
            latency = (
                perf_counter() - started
            ) * 1000.0

            observation = ToolObservation(
                call_id=call.call_id,
                tool=call.name,
                success=False,
                status=EXECUTION_FAILED,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                latency_ms=round(
                    latency,
                    4,
                ),
                step=step,
            )

        self._record_observation(
            call,
            observation,
        )

        return observation

    # ------------------------------------------------------------------------
    # BATCH EXECUTION
    # ------------------------------------------------------------------------

    async def execute_many(
        self,
        tool_calls: Iterable[Any],
        *,
        step: int = 1,
        source: str = "brain",
        parallel: bool = False,
    ) -> list[ToolObservation]:
        """
        Menjalankan banyak tool call.

        Args:
            tool_calls: Iterable tool call.
            step: Step.
            source: Source.
            parallel: Parallel execution.

        Returns:
            List observations.
        """
        calls = list(tool_calls)

        if not calls:
            return []

        if parallel:
            import asyncio

            tasks = [
                self.execute(
                    item,
                    step=step,
                    source=source,
                )
                for item in calls
            ]

            return list(
                await asyncio.gather(
                    *tasks
                )
            )

        results: list[ToolObservation] = []

        for item in calls:
            if len(results) >= self.policy.max_tool_calls:
                break

            results.append(
                await self.execute(
                    item,
                    step=step,
                    source=source,
                )
            )

        return results

    # ------------------------------------------------------------------------
    # LOOP EXECUTION
    # ------------------------------------------------------------------------

    async def run(
        self,
        tool_calls: Sequence[Any],
        *,
        source: str = "brain",
    ) -> ToolLoopResult:
        """
        Menjalankan execution loop.

        Tool call yang masuk dianggap sebagai action plan dari Brain.

        Args:
            tool_calls: Sequence tool call.
            source: Source.

        Returns:
            ToolLoopResult.
        """
        started = perf_counter()

        result = ToolLoopResult(
            success=True,
            status=EXECUTION_COMPLETED,
        )

        if not tool_calls:
            result.metadata.update(
                {
                    "message": (
                        "Tidak ada tool call."
                    )
                }
            )

            result.finalize(
                success=True,
                status=EXECUTION_COMPLETED,
                latency_ms=(
                    perf_counter() - started
                )
                * 1000.0,
            )

            return result

        for index, raw_call in enumerate(
            tool_calls,
            start=1,
        ):
            if index > self.policy.max_tool_calls:
                result.add_warning(
                    (
                        "Tool call dihentikan karena "
                        "batas maksimum tercapai."
                    )
                )
                break

            step = index

            allowed, error = (
                self.policy.check_step(step)
            )

            if not allowed:
                result.add_error(
                    error or "Step ditolak."
                )
                result.status = EXECUTION_BLOCKED
                result.success = False
                break

            observation = await self.execute(
                raw_call,
                step=step,
                source=source,
            )

            result.add_observation(
                observation
            )

            result.steps = step
            result.tool_calls += 1

            if not observation.success:
                result.success = False

                if observation.status in {
                    EXECUTION_DENIED,
                    EXECUTION_BLOCKED,
                    EXECUTION_TIMEOUT,
                }:
                    result.status = (
                        observation.status
                    )
                else:
                    result.status = (
                        EXECUTION_FAILED
                    )

        if result.success:
            result.status = EXECUTION_COMPLETED

        if result.errors:
            result.success = False

        result.latency_ms = round(
            (
                perf_counter() - started
            )
            * 1000.0,
            4,
        )

        result.completed_at = utc_now()

        return result

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    def _validate_call(
        self,
        call: ToolCall,
    ) -> Optional[
        tuple[str, str]
    ]:
        """
        Validasi satu tool call.

        Returns:
            None jika valid, tuple status/error jika invalid.
        """
        allowed, error = (
            self.policy.check_tool_name(
                call.name
            )
        )

        if not allowed:
            return (
                EXECUTION_BLOCKED,
                error or (
                    "Tool tidak diperbolehkan."
                ),
            )

        valid, argument_error = (
            self.policy.check_arguments(
                call.arguments
            )
        )

        if not valid:
            return (
                EXECUTION_BLOCKED,
                argument_error or (
                    "Argument tool tidak valid."
                ),
            )

        valid_step, step_error = (
            self.policy.check_step(
                call.step
            )
        )

        if not valid_step:
            return (
                EXECUTION_BLOCKED,
                step_error or (
                    "Step tool tidak valid."
                ),
            )

        if not self.policy.allow_unknown_tools:
            if not self.adapter.has_tool(
                call.name
            ):
                return (
                    EXECUTION_FAILED,
                    (
                        f"Tool '{call.name}' "
                        "tidak tersedia."
                    ),
                )

        signature = call.signature()

        if (
            not self.policy.allow_repeated_calls
            and signature in self._active_signatures
        ):
            return (
                EXECUTION_BLOCKED,
                (
                    "Tool call berulang terdeteksi: "
                    f"{call.name}."
                ),
            )

        return None

    # ------------------------------------------------------------------------
    # RECORDING
    # ------------------------------------------------------------------------

    def _record_observation(
        self,
        call: ToolCall,
        observation: ToolObservation,
    ) -> None:
        """
        Menyimpan execution result.
        """
        self.execution_count += 1

        self.total_latency_ms += (
            observation.latency_ms
        )

        if observation.success:
            self.success_count += 1
        else:
            self.failure_count += 1

        if observation.status == EXECUTION_TIMEOUT:
            self.timeout_count += 1

        signature = call.signature()

        self._active_signatures.add(
            signature
        )

        record = ToolExecutionRecord(
            execution_id=str(uuid4()),
            call=call,
            observation=observation,
        )

        self._history.append(
            record
        )

        self._trim_history()

    def _record_denial(
        self,
        observation: ToolObservation,
    ) -> None:
        """
        Mencatat execution yang ditolak.
        """
        self.execution_count += 1
        self.failure_count += 1
        self.denied_count += 1

        self.total_latency_ms += (
            observation.latency_ms
        )

        if observation.status == EXECUTION_BLOCKED:
            self.blocked_count += 1

        call = ToolCall(
            name=observation.tool,
            arguments={},
            call_id=observation.call_id,
            step=observation.step,
            source="policy",
        )

        record = ToolExecutionRecord(
            execution_id=str(uuid4()),
            call=call,
            observation=observation,
        )

        self._history.append(
            record
        )

        self._trim_history()

    def _trim_history(self) -> None:
        """
        Menjaga history sesuai limit.
        """
        overflow = (
            len(self._history)
            - self.history_limit
        )

        if overflow > 0:
            del self._history[
                :overflow
            ]

    # ------------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------------

    def history(
        self,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Mengambil execution history.

        Args:
            limit: Jumlah maksimum record.

        Returns:
            List history.
        """
        records = self._history

        if limit is not None:
            limit = max(
                0,
                int(limit),
            )

            if limit == 0:
                return []

            records = records[-limit:]

        return [
            item.to_dict()
            for item in records
        ]

    def clear_history(self) -> None:
        """
        Menghapus history dan signature aktif.
        """
        self._history.clear()
        self._active_signatures.clear()

    # ------------------------------------------------------------------------
    # TOOL DISCOVERY
    # ------------------------------------------------------------------------

    def available_tools(self) -> list[str]:
        """
        Mengambil daftar tool yang tersedia.

        Returns:
            List nama tool.
        """
        return self.adapter.available_tools()

    def has_tool(
        self,
        name: str,
    ) -> bool:
        """
        Mengecek apakah tool tersedia.

        Args:
            name: Nama tool.

        Returns:
            True jika tersedia.
        """
        return self.adapter.has_tool(
            name
        )

    # ------------------------------------------------------------------------
    # POLICY CONTROL
    # ------------------------------------------------------------------------

    def allow_tool(
        self,
        name: str,
    ) -> None:
        """
        Menambahkan tool ke allow-list.

        Args:
            name: Nama tool.
        """
        normalized = normalize_tool_name(
            name
        )

        if self.policy.allowed_tools is None:
            self.policy.allowed_tools = set()

        self.policy.allowed_tools.add(
            normalized
        )

    def block_tool(
        self,
        name: str,
    ) -> None:
        """
        Memblokir tool.

        Args:
            name: Nama tool.
        """
        self.policy.blocked_tools.add(
            normalize_tool_name(name)
        )

    def unblock_tool(
        self,
        name: str,
    ) -> None:
        """
        Membuka block tool.

        Args:
            name: Nama tool.
        """
        self.policy.blocked_tools.discard(
            normalize_tool_name(name)
        )

    def set_max_steps(
        self,
        value: int,
    ) -> None:
        """
        Mengubah maksimum steps.

        Args:
            value: Maximum step.
        """
        self.policy.max_steps = max(
            1,
            int(value),
        )

    def set_max_tool_calls(
        self,
        value: int,
    ) -> None:
        """
        Mengubah maksimum tool calls.

        Args:
            value: Maximum calls.
        """
        self.policy.max_tool_calls = max(
            1,
            int(value),
        )

    # ------------------------------------------------------------------------
    # OBSERVATION FORMAT
    # ------------------------------------------------------------------------

    @staticmethod
    def observation_for_brain(
        observation: ToolObservation,
    ) -> dict[str, Any]:
        """
        Membuat observation minimal untuk Brain.

        Format ini sengaja stabil agar ZAIBrain tidak bergantung pada
        detail internal ToolResult.

        Args:
            observation: Observation.

        Returns:
            Brain observation.
        """
        return {
            "type": "tool_observation",
            "call_id": observation.call_id,
            "tool": observation.tool,
            "success": observation.success,
            "status": observation.status,
            "response": observation.response,
            "data": observation.data,
            "error": observation.error,
            "latency_ms": observation.latency_ms,
            "step": observation.step,
            "metadata": dict(
                observation.metadata
            ),
        }

    @classmethod
    def observations_for_brain(
        cls,
        observations: Iterable[
            ToolObservation
        ],
    ) -> list[dict[str, Any]]:
        """
        Mengubah banyak observation.

        Args:
            observations: Iterable observation.

        Returns:
            List observation.
        """
        return [
            cls.observation_for_brain(
                item
            )
            for item in observations
        ]


# ============================================================================
# FACTORY
# ============================================================================


def create_tool_execution_loop(
    executor: Any,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    require_whitelist: bool = False,
    allowed_tools: Optional[
        Iterable[str]
    ] = None,
    blocked_tools: Optional[
        Iterable[str]
    ] = None,
) -> ToolExecutionLoop:
    """
    Factory standar ToolExecutionLoop.

    Args:
        executor: ToolManager/ToolRegistry.
        max_steps: Maximum step.
        max_tool_calls: Maximum tool call.
        require_whitelist: Require allow-list.
        allowed_tools: Allowed tools.
        blocked_tools: Blocked tools.

    Returns:
        ToolExecutionLoop.
    """
    policy = ToolPolicy(
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        require_whitelist=require_whitelist,
        allowed_tools=(
            set(allowed_tools)
            if allowed_tools is not None
            else None
        ),
        blocked_tools=(
            set(blocked_tools)
            if blocked_tools is not None
            else set()
        ),
    )

    return ToolExecutionLoop(
        executor,
        policy=policy,
    )


# ============================================================================
# DEFAULT SINGLETON SUPPORT
# ============================================================================


_default_loop: Optional[
    ToolExecutionLoop
] = None


def configure_default_tool_loop(
    executor: Any,
    *,
    policy: Optional[ToolPolicy] = None,
) -> ToolExecutionLoop:
    """
    Mengkonfigurasi singleton execution loop.

    Args:
        executor: Tool executor.
        policy: Optional policy.

    Returns:
        ToolExecutionLoop.
    """
    global _default_loop

    _default_loop = ToolExecutionLoop(
        executor,
        policy=policy,
    )

    return _default_loop


def get_default_tool_loop() -> ToolExecutionLoop:
    """
    Mengambil singleton execution loop.

    Returns:
        ToolExecutionLoop.

    Raises:
        RuntimeError: Jika belum dikonfigurasi.
    """
    if _default_loop is None:
        raise RuntimeError(
            "Default ToolExecutionLoop belum "
            "dikonfigurasi."
        )

    return _default_loop


# ============================================================================
# TESTABLE MOCK EXECUTOR
# ============================================================================


class InMemoryToolExecutor:
    """
    Executor kecil untuk unit/integration test.

    Class ini bukan production tool manager.
    Fungsinya hanya menyediakan executor deterministic untuk testing
    ToolExecutionLoop tanpa membutuhkan network atau infrastructure.
    """

    def __init__(self) -> None:
        self._tools: dict[
            str,
            Callable[..., Any],
        ] = {}

    def register(
        self,
        name: str,
        function: Callable[..., Any],
    ) -> None:
        """
        Register function.

        Args:
            name: Nama tool.
            function: Callable.
        """
        normalized = normalize_tool_name(
            name
        )

        if not callable(function):
            raise TypeError(
                "Tool function harus callable."
            )

        self._tools[
            normalized
        ] = function

    def names(self) -> list[str]:
        """
        Mengambil daftar nama tool.

        Returns:
            List tool.
        """
        return list(
            self._tools.keys()
        )

    def has_tool(
        self,
        name: str,
    ) -> bool:
        """
        Mengecek tool.

        Args:
            name: Nama tool.

        Returns:
            True jika tersedia.
        """
        return (
            normalize_tool_name(name)
            in self._tools
        )

    def execute(
        self,
        tool: str,
        arguments: Optional[
            dict[str, Any]
        ] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute sync function.

        Args:
            tool: Nama tool.
            arguments: Arguments.
            kwargs: Compatibility kwargs.

        Returns:
            Normalized dictionary.
        """
        name = normalize_tool_name(
            tool
        )

        function = self._tools.get(
            name
        )

        if function is None:
            return {
                "success": False,
                "tool": name,
                "status": EXECUTION_FAILED,
                "response": None,
                "data": None,
                "error": (
                    f"Tool '{name}' tidak tersedia."
                ),
            }

        payload = dict(
            arguments or {}
        )

        try:
            response = function(
                **payload
            )

            return {
                "success": True,
                "tool": name,
                "status": EXECUTION_COMPLETED,
                "response": response,
                "data": response,
                "error": None,
            }

        except Exception as exc:
            return {
                "success": False,
                "tool": name,
                "status": EXECUTION_FAILED,
                "response": None,
                "data": None,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    "MODULE_NAME",
    "MODULE_VERSION",
    "ToolCall",
    "ToolObservation",
    "ToolExecutionRecord",
    "ToolLoopResult",
    "ToolPolicy",
    "ToolCallParser",
    "ToolExecutorAdapter",
    "ToolResultNormalizer",
    "ToolExecutionLoop",
    "create_tool_execution_loop",
    "configure_default_tool_loop",
    "get_default_tool_loop",
    "InMemoryToolExecutor",
]
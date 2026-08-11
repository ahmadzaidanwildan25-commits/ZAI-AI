from __future__ import annotations

"""
ZAI - Agent Result
==================

Core result object untuk seluruh sistem agent ZAI.

File:
    ai/agents/agent_result.py

Tanggung jawab:
    - Menyimpan hasil eksekusi agent.
    - Menyimpan status task.
    - Menyimpan response agent.
    - Menyimpan metadata.
    - Menyimpan observation/event.
    - Menyimpan warning.
    - Menyimpan error.
    - Menyimpan execution ID.
    - Menyimpan timestamp.
    - Mendukung method chaining.
    - Menyediakan serialisasi ke dictionary.
    - Menyediakan helper untuk runtime dan monitoring.

Design goals:
    - Aman.
    - Mudah dikembangkan.
    - Backward-compatible dengan BaseAgent.
    - Backward-compatible dengan AgentRuntime.
    - Tidak bergantung pada framework eksternal.
    - Cocok untuk FastAPI.
    - Cocok untuk logging.
    - Cocok untuk testing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4


class AgentResultError(Exception):
    """
    Base exception untuk AgentResult.
    """

    pass


class InvalidAgentResultError(AgentResultError):
    """
    Dipakai ketika data AgentResult tidak valid.
    """

    pass


@dataclass(slots=True)
class AgentObservation:
    """
    Representasi satu observation/event selama agent berjalan.
    """

    event: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: AgentResult.utcnow_iso()
    )
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        Mengubah observation menjadi dictionary.
        """

        return {
            "event": self.event,
            "data": dict(self.data),
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


@dataclass(slots=True)
class AgentWarning:
    """
    Representasi warning yang dihasilkan agent.
    """

    message: str
    code: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: AgentResult.utcnow_iso()
    )
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        Mengubah warning menjadi dictionary.
        """

        payload: dict[str, Any] = {
            "message": self.message,
            "code": self.code,
            "data": dict(self.data),
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }

        return payload


@dataclass(slots=True)
class AgentError:
    """
    Representasi error yang dihasilkan agent.
    """

    message: str
    code: str | None = None
    error_type: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: AgentResult.utcnow_iso()
    )
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        Mengubah error menjadi dictionary.
        """

        return {
            "message": self.message,
            "code": self.code,
            "error_type": self.error_type,
            "data": dict(self.data),
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


class AgentResult:
    """
    Result object utama ZAI Agent System.

    Contoh:

        result = AgentResult(
            success=True,
            agent="general_agent",
            response="Halo dari ZAI",
            task="Halo ZAI",
            status="running",
        )

        result.add_observation(
            "task_received",
            task="Halo ZAI",
        )

        result.complete(
            "Halo! Saya ZAI.",
            latency_ms=10.5,
        )

        print(result.to_dict())

    Semua mutation method mengembalikan self sehingga mendukung:

        result \
            .add_observation("start") \
            .add_warning("warning") \
            .set_metadata("source", "runtime") \
            .complete("OK")
    """

    VERSION = "2.3.0"

    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"
    STATUS_TIMEOUT = "timeout"
    STATUS_PARTIAL = "partial"

    VALID_STATUSES = frozenset(
        {
            STATUS_RUNNING,
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_CANCELLED,
            STATUS_TIMEOUT,
            STATUS_PARTIAL,
        }
    )

    SUCCESS_STATUSES = frozenset(
        {
            STATUS_COMPLETED,
        }
    )

    FAILURE_STATUSES = frozenset(
        {
            STATUS_FAILED,
            STATUS_CANCELLED,
            STATUS_TIMEOUT,
        }
    )

    TERMINAL_STATUSES = frozenset(
        {
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_CANCELLED,
            STATUS_TIMEOUT,
            STATUS_PARTIAL,
        }
    )

    def __init__(
        self,
        success: bool,
        agent: str,
        response: str,
        task: str,
        status: str = STATUS_RUNNING,
        *,
        execution_id: str | UUID | None = None,
        created_at: str | datetime | None = None,
        completed_at: str | datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        observations: Iterable[
            AgentObservation | Mapping[str, Any]
        ]
        | None = None,
        warnings: Iterable[
            AgentWarning | Mapping[str, Any] | str
        ]
        | None = None,
        errors: Iterable[
            AgentError | Mapping[str, Any] | str
        ]
        | None = None,
    ) -> None:
        """
        Membuat AgentResult baru.
        """

        self.success = bool(success)

        self.agent = self._normalize_text(
            agent,
            field_name="agent",
        )

        self.response = (
            response
            if response is not None
            else ""
        )

        self.task = (
            task
            if task is not None
            else ""
        )

        self.status = self._normalize_status(
            status
        )

        self.execution_id = self._normalize_execution_id(
            execution_id
        )

        self.created_at = self._normalize_timestamp(
            created_at
        )

        self.completed_at = self._normalize_optional_timestamp(
            completed_at
        )

        self.metadata: dict[str, Any] = {}

        if metadata:
            self.metadata.update(
                dict(metadata)
            )

        self.observations: list[AgentObservation] = []
        self.warnings: list[AgentWarning] = []
        self.errors: list[AgentError] = []

        self._sequence = 0

        if observations:
            for observation in observations:
                self._load_observation(
                    observation
                )

        if warnings:
            for warning in warnings:
                self._load_warning(
                    warning
                )

        if errors:
            for error in errors:
                self._load_error(
                    error
                )

        self._synchronize_success_state()

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    @staticmethod
    def utc_now() -> datetime:
        """
        Mengembalikan waktu UTC timezone-aware.
        """

        return datetime.now(
            timezone.utc
        )

    @staticmethod
    def utcnow_iso() -> str:
        """
        Mengembalikan timestamp UTC ISO-8601.
        """

        return AgentResult.utc_now().isoformat()

    @staticmethod
    def _normalize_text(
        value: Any,
        *,
        field_name: str,
    ) -> str:
        """
        Normalisasi nilai text.
        """

        if value is None:
            raise InvalidAgentResultError(
                f"{field_name} tidak boleh None."
            )

        text = str(value)

        if not text.strip():
            raise InvalidAgentResultError(
                f"{field_name} tidak boleh kosong."
            )

        return text

    @classmethod
    def _normalize_status(
        cls,
        status: str | None,
    ) -> str:
        """
        Validasi status.
        """

        normalized = (
            str(status).strip().lower()
            if status is not None
            else cls.STATUS_RUNNING
        )

        if normalized not in cls.VALID_STATUSES:
            raise InvalidAgentResultError(
                "Status AgentResult tidak valid: "
                f"{status!r}. "
                f"Valid status: "
                f"{sorted(cls.VALID_STATUSES)}"
            )

        return normalized

    @staticmethod
    def _normalize_execution_id(
        execution_id: str | UUID | None,
    ) -> str:
        """
        Normalisasi execution ID.
        """

        if execution_id is None:
            return str(uuid4())

        if isinstance(execution_id, UUID):
            return str(execution_id)

        value = str(execution_id).strip()

        if not value:
            return str(uuid4())

        return value

    @classmethod
    def _normalize_timestamp(
        cls,
        value: str | datetime | None,
    ) -> str:
        """
        Normalisasi timestamp wajib.
        """

        if value is None:
            return cls.utcnow_iso()

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(
                    tzinfo=timezone.utc
                )

            return value.isoformat()

        text = str(value).strip()

        if not text:
            return cls.utcnow_iso()

        return text

    @classmethod
    def _normalize_optional_timestamp(
        cls,
        value: str | datetime | None,
    ) -> str | None:
        """
        Normalisasi timestamp optional.
        """

        if value is None:
            return None

        return cls._normalize_timestamp(
            value
        )

    # ============================================================
    # INTERNAL SEQUENCE
    # ============================================================

    def _next_sequence(self) -> int:
        """
        Menghasilkan nomor sequence berikutnya.
        """

        self._sequence += 1

        return self._sequence

    def _synchronize_success_state(self) -> None:
        """
        Menjaga konsistensi success dan status.
        """

        if self.status == self.STATUS_COMPLETED:
            self.success = True

        elif self.status in self.FAILURE_STATUSES:
            self.success = False

        elif self.status == self.STATUS_PARTIAL:
            self.success = False

    # ============================================================
    # OBSERVATION
    # ============================================================

    def add_observation(
        self,
        event: str,
        **data: Any,
    ) -> AgentResult:
        """
        Menambahkan observation/event.

        Contoh:

            result.add_observation(
                "task_received",
                task="Halo ZAI",
            )
        """

        event_name = self._normalize_text(
            event,
            field_name="event",
        )

        observation = AgentObservation(
            event=event_name,
            data=dict(data),
            timestamp=self.utcnow_iso(),
            sequence=self._next_sequence(),
        )

        self.observations.append(
            observation
        )

        return self

    def observation(
        self,
        event: str,
        **data: Any,
    ) -> AgentResult:
        """
        Alias add_observation().
        """

        return self.add_observation(
            event,
            **data,
        )

    def record_event(
        self,
        event: str,
        **data: Any,
    ) -> AgentResult:
        """
        Alias untuk event logging.
        """

        return self.add_observation(
            event,
            **data,
        )

    # ============================================================
    # WARNING
    # ============================================================

    def add_warning(
        self,
        message: str,
        *,
        code: str | None = None,
        **data: Any,
    ) -> AgentResult:
        """
        Menambahkan warning.

        Method ini diperlukan oleh testing:

            result.add_warning("test warning")
        """

        warning_message = self._normalize_text(
            message,
            field_name="warning",
        )

        warning = AgentWarning(
            message=warning_message,
            code=code,
            data=dict(data),
            timestamp=self.utcnow_iso(),
            sequence=self._next_sequence(),
        )

        self.warnings.append(
            warning
        )

        return self

    def warning(
        self,
        message: str,
        *,
        code: str | None = None,
        **data: Any,
    ) -> AgentResult:
        """
        Alias add_warning().
        """

        return self.add_warning(
            message,
            code=code,
            **data,
        )

    # ============================================================
    # ERROR
    # ============================================================

    def add_error(
        self,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
        **data: Any,
    ) -> AgentResult:
        """
        Menambahkan error.

        Jika result sedang running/completed,
        error akan membuat result failed.

        Contoh:

            result.add_error(
                "Simulated error"
            )
        """

        error_message = self._normalize_text(
            message,
            field_name="error",
        )

        resolved_error_type = error_type

        if resolved_error_type is None:
            resolved_error_type = (
                "AgentError"
            )

        error = AgentError(
            message=error_message,
            code=code,
            error_type=resolved_error_type,
            data=dict(data),
            timestamp=self.utcnow_iso(),
            sequence=self._next_sequence(),
        )

        self.errors.append(
            error
        )

        self.success = False

        if self.status not in {
            self.STATUS_CANCELLED,
            self.STATUS_TIMEOUT,
        }:
            self.status = self.STATUS_FAILED

        self.completed_at = (
            self.completed_at
            or self.utcnow_iso()
        )

        return self

    def error(
        self,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
        **data: Any,
    ) -> AgentResult:
        """
        Alias add_error().
        """

        return self.add_error(
            message,
            code=code,
            error_type=error_type,
            **data,
        )

    def add_exception(
        self,
        exc: BaseException,
        *,
        code: str | None = None,
        **data: Any,
    ) -> AgentResult:
        """
        Menambahkan exception sebagai error.
        """

        if not isinstance(
            exc,
            BaseException,
        ):
            raise TypeError(
                "exc harus merupakan "
                "instance BaseException."
            )

        return self.add_error(
            f"{type(exc).__name__}: {exc}",
            code=code,
            error_type=type(exc).__name__,
            **data,
        )

    # ============================================================
    # METADATA
    # ============================================================

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> AgentResult:
        """
        Mengatur satu metadata.
        """

        metadata_key = self._normalize_text(
            key,
            field_name="metadata key",
        )

        self.metadata[
            metadata_key
        ] = value

        return self

    def update_metadata(
        self,
        values: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Update banyak metadata sekaligus.
        """

        if values:
            self.metadata.update(
                dict(values)
            )

        if kwargs:
            self.metadata.update(
                kwargs
            )

        return self

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Mengambil metadata.
        """

        return self.metadata.get(
            key,
            default,
        )

    def has_metadata(
        self,
        key: str,
    ) -> bool:
        """
        Mengecek apakah metadata tersedia.
        """

        return key in self.metadata

    def remove_metadata(
        self,
        key: str,
    ) -> AgentResult:
        """
        Menghapus metadata.
        """

        self.metadata.pop(
            key,
            None,
        )

        return self

    def clear_metadata(self) -> AgentResult:
        """
        Menghapus seluruh metadata.
        """

        self.metadata.clear()

        return self

    # ============================================================
    # RESPONSE
    # ============================================================

    def set_response(
        self,
        response: Any,
    ) -> AgentResult:
        """
        Mengatur response.
        """

        if response is None:
            self.response = ""
        elif isinstance(
            response,
            str,
        ):
            self.response = response
        else:
            self.response = str(
                response
            )

        return self

    def append_response(
        self,
        text: Any,
    ) -> AgentResult:
        """
        Menambahkan text ke response.
        """

        if text is None:
            return self

        self.response += str(text)

        return self

    def clear_response(self) -> AgentResult:
        """
        Mengosongkan response.
        """

        self.response = ""

        return self

    # ============================================================
    # TASK
    # ============================================================

    def set_task(
        self,
        task: Any,
    ) -> AgentResult:
        """
        Mengubah task.
        """

        self.task = (
            ""
            if task is None
            else str(task)
        )

        return self

    # ============================================================
    # STATUS
    # ============================================================

    def set_status(
        self,
        status: str,
    ) -> AgentResult:
        """
        Mengubah status result.
        """

        self.status = self._normalize_status(
            status
        )

        self._synchronize_success_state()

        if self.status in self.TERMINAL_STATUSES:
            self.completed_at = (
                self.completed_at
                or self.utcnow_iso()
            )

        return self

    def mark_running(self) -> AgentResult:
        """
        Menandai result sebagai running.
        """

        self.status = self.STATUS_RUNNING
        self.success = False
        self.completed_at = None

        return self

    def mark_completed(
        self,
        response: Any | None = None,
        **metadata: Any,
    ) -> AgentResult:
        """
        Alias semantic untuk complete().
        """

        return self.complete(
            response=response,
            **metadata,
        )

    def mark_failed(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        **data: Any,
    ) -> AgentResult:
        """
        Menandai result failed.
        """

        self.success = False
        self.status = self.STATUS_FAILED

        if message:
            self.response = str(
                message
            )

        if message:
            self.add_error(
                message,
                code=code,
                **data,
            )

        self.completed_at = (
            self.completed_at
            or self.utcnow_iso()
        )

        return self

    def mark_cancelled(
        self,
        reason: str | None = None,
    ) -> AgentResult:
        """
        Menandai result cancelled.
        """

        self.success = False
        self.status = self.STATUS_CANCELLED

        if reason:
            self.add_warning(
                reason,
                code="TASK_CANCELLED",
            )

        self.completed_at = (
            self.completed_at
            or self.utcnow_iso()
        )

        return self

    def mark_timeout(
        self,
        message: str | None = None,
    ) -> AgentResult:
        """
        Menandai result timeout.
        """

        self.success = False
        self.status = self.STATUS_TIMEOUT

        if message:
            self.add_error(
                message,
                code="TASK_TIMEOUT",
                error_type="TimeoutError",
            )

        self.completed_at = (
            self.completed_at
            or self.utcnow_iso()
        )

        return self

    def mark_partial(
        self,
        response: Any | None = None,
    ) -> AgentResult:
        """
        Menandai result partial.
        """

        self.success = False
        self.status = self.STATUS_PARTIAL

        if response is not None:
            self.response = str(
                response
            )

        self.completed_at = (
            self.completed_at
            or self.utcnow_iso()
        )

        return self

    # ============================================================
    # COMPLETE
    # ============================================================

    def complete(
        self,
        response: Any | None = None,
        **metadata: Any,
    ) -> AgentResult:
        """
        Menyelesaikan result sebagai sukses.

        Contoh:

            result.complete(
                "ZAI test berhasil",
                latency_ms=1.0,
            )

        Method mengembalikan self untuk chaining.
        """

        if response is not None:
            self.set_response(
                response
            )

        if metadata:
            self.update_metadata(
                metadata
            )

        self.success = True
        self.status = self.STATUS_COMPLETED

        self.completed_at = (
            self.completed_at
            or self.utcnow_iso()
        )

        return self

    # ============================================================
    # FAIL
    # ============================================================

    def fail(
        self,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
        **data: Any,
    ) -> AgentResult:
        """
        Shortcut untuk failure.
        """

        return self.add_error(
            message,
            code=code,
            error_type=error_type,
            **data,
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> AgentResult:
        """
        Mereset result menjadi running.

        Execution ID dibuat ulang.
        """

        self.success = False
        self.response = ""
        self.status = self.STATUS_RUNNING

        self.execution_id = str(
            uuid4()
        )

        self.created_at = (
            self.utcnow_iso()
        )

        self.completed_at = None

        self.metadata.clear()
        self.observations.clear()
        self.warnings.clear()
        self.errors.clear()

        self._sequence = 0

        return self

    # ============================================================
    # COUNTS
    # ============================================================

    @property
    def observation_count(self) -> int:
        """
        Jumlah observation.
        """

        return len(
            self.observations
        )

    @property
    def warning_count(self) -> int:
        """
        Jumlah warning.
        """

        return len(
            self.warnings
        )

    @property
    def error_count(self) -> int:
        """
        Jumlah error.
        """

        return len(
            self.errors
        )

    # ============================================================
    # BOOLEAN HELPERS
    # ============================================================

    @property
    def has_errors(self) -> bool:
        """
        True jika terdapat error.
        """

        return bool(
            self.errors
        )

    @property
    def has_warnings(self) -> bool:
        """
        True jika terdapat warning.
        """

        return bool(
            self.warnings
        )

    @property
    def has_observations(self) -> bool:
        """
        True jika terdapat observation.
        """

        return bool(
            self.observations
        )

    @property
    def is_running(self) -> bool:
        """
        True jika sedang running.
        """

        return (
            self.status
            == self.STATUS_RUNNING
        )

    @property
    def is_completed(self) -> bool:
        """
        True jika completed.
        """

        return (
            self.status
            == self.STATUS_COMPLETED
        )

    @property
    def is_failed(self) -> bool:
        """
        True jika failed.
        """

        return (
            self.status
            == self.STATUS_FAILED
        )

    @property
    def is_cancelled(self) -> bool:
        """
        True jika cancelled.
        """

        return (
            self.status
            == self.STATUS_CANCELLED
        )

    @property
    def is_timeout(self) -> bool:
        """
        True jika timeout.
        """

        return (
            self.status
            == self.STATUS_TIMEOUT
        )

    @property
    def is_partial(self) -> bool:
        """
        True jika partial.
        """

        return (
            self.status
            == self.STATUS_PARTIAL
        )

    @property
    def is_terminal(self) -> bool:
        """
        True jika task sudah selesai.
        """

        return (
            self.status
            in self.TERMINAL_STATUSES
        )

    @property
    def ok(self) -> bool:
        """
        Alias success.
        """

        return self.success

    @property
    def failed(self) -> bool:
        """
        Alias is_failed.
        """

        return self.is_failed

    # ============================================================
    # LATENCY
    # ============================================================

    @property
    def latency_ms(self) -> float | None:
        """
        Mengambil latency dari metadata jika tersedia.
        """

        value = self.metadata.get(
            "latency_ms"
        )

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @latency_ms.setter
    def latency_ms(
        self,
        value: float | int | None,
    ) -> None:
        """
        Menyimpan latency.
        """

        if value is None:
            self.metadata.pop(
                "latency_ms",
                None,
            )
            return

        self.metadata[
            "latency_ms"
        ] = float(value)

    # ============================================================
    # TIMESTAMP HELPERS
    # ============================================================

    @property
    def duration_ms(self) -> float | None:
        """
        Menghitung durasi dari created_at
        ke completed_at.

        Jika timestamp tidak bisa diparse,
        return None.
        """

        if not self.completed_at:
            return None

        try:
            start = datetime.fromisoformat(
                self.created_at
            )

            end = datetime.fromisoformat(
                self.completed_at
            )

            if start.tzinfo is None:
                start = start.replace(
                    tzinfo=timezone.utc
                )

            if end.tzinfo is None:
                end = end.replace(
                    tzinfo=timezone.utc
                )

            value = (
                end - start
            ).total_seconds() * 1000

            return round(
                max(value, 0.0),
                2,
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(self) -> dict[str, Any]:
        """
        Ringkasan ringan AgentResult.
        """

        return {
            "success": self.success,
            "agent": self.agent,
            "status": self.status,
            "execution_id": self.execution_id,
            "response": self.response,
            "task": self.task,
            "observation_count": self.observation_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "is_terminal": self.is_terminal,
            "latency_ms": self.latency_ms,
            "duration_ms": self.duration_ms,
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def to_dict(
        self,
        *,
        include_observations: bool = True,
        include_warnings: bool = True,
        include_errors: bool = True,
    ) -> dict[str, Any]:
        """
        Serialisasi lengkap menjadi dictionary.

        Bentuk output kompatibel dengan output
        yang sudah terlihat pada testing ZAI.
        """

        payload: dict[str, Any] = {
            "success": self.success,
            "agent": self.agent,
            "response": self.response,
            "task": self.task,
            "status": self.status,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": dict(self.metadata),
            "observation_count": self.observation_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
        }

        if include_observations:
            payload[
                "observations"
            ] = [
                item.to_dict()
                for item in self.observations
            ]

        if include_warnings:
            payload[
                "warnings"
            ] = [
                item.to_dict()
                for item in self.warnings
            ]

        if include_errors:
            payload[
                "errors"
            ] = [
                item.to_dict()
                for item in self.errors
            ]

        return payload

    def dict(
        self,
    ) -> dict[str, Any]:
        """
        Alias kompatibilitas gaya Pydantic.
        """

        return self.to_dict()

    def model_dump(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Alias kompatibilitas Pydantic v2.
        """

        return self.to_dict(
            **kwargs
        )

    # ============================================================
    # JSON-SAFE SERIALIZATION
    # ============================================================

    @classmethod
    def _json_safe(
        cls,
        value: Any,
    ) -> Any:
        """
        Mengubah object menjadi struktur
        yang aman untuk JSON.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(
            value,
            UUID,
        ):
            return str(value)

        if isinstance(
            value,
            datetime,
        ):
            return cls._normalize_timestamp(
                value
            )

        if isinstance(
            value,
            Mapping,
        ):
            return {
                str(key): cls._json_safe(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
                frozenset,
            ),
        ):
            return [
                cls._json_safe(item)
                for item in value
            ]

        if hasattr(
            value,
            "to_dict",
        ):
            try:
                return cls._json_safe(
                    value.to_dict()
                )
            except Exception:
                pass

        return str(value)

    def to_json_dict(
        self,
    ) -> dict[str, Any]:
        """
        Menghasilkan dictionary JSON-safe.
        """

        return self._json_safe(
            self.to_dict()
        )

    # ============================================================
    # COPY / CLONE
    # ============================================================

    def clone(
        self,
        *,
        new_execution_id: bool = False,
    ) -> AgentResult:
        """
        Membuat salinan AgentResult.
        """

        cloned = AgentResult(
            success=self.success,
            agent=self.agent,
            response=self.response,
            task=self.task,
            status=self.status,
            execution_id=(
                None
                if new_execution_id
                else self.execution_id
            ),
            created_at=self.created_at,
            completed_at=self.completed_at,
            metadata=dict(
                self.metadata
            ),
        )

        for observation in self.observations:
            cloned.observations.append(
                AgentObservation(
                    event=observation.event,
                    data=dict(
                        observation.data
                    ),
                    timestamp=observation.timestamp,
                    sequence=observation.sequence,
                )
            )

        for warning in self.warnings:
            cloned.warnings.append(
                AgentWarning(
                    message=warning.message,
                    code=warning.code,
                    data=dict(
                        warning.data
                    ),
                    timestamp=warning.timestamp,
                    sequence=warning.sequence,
                )
            )

        for error in self.errors:
            cloned.errors.append(
                AgentError(
                    message=error.message,
                    code=error.code,
                    error_type=error.error_type,
                    data=dict(
                        error.data
                    ),
                    timestamp=error.timestamp,
                    sequence=error.sequence,
                )
            )

        cloned._sequence = self._sequence

        return cloned

    # ============================================================
    # LOAD HELPERS
    # ============================================================

    def _load_observation(
        self,
        value: AgentObservation | Mapping[str, Any],
    ) -> None:
        """
        Memuat observation dari object/dict.
        """

        if isinstance(
            value,
            AgentObservation,
        ):
            observation = value

        elif isinstance(
            value,
            Mapping,
        ):
            observation = AgentObservation(
                event=str(
                    value.get(
                        "event",
                        "",
                    )
                ),
                data=dict(
                    value.get(
                        "data",
                        {},
                    )
                ),
                timestamp=str(
                    value.get(
                        "timestamp",
                        self.utcnow_iso(),
                    )
                ),
                sequence=int(
                    value.get(
                        "sequence",
                        self._next_sequence(),
                    )
                ),
            )

        else:
            raise TypeError(
                "Observation harus "
                "AgentObservation atau Mapping."
            )

        self.observations.append(
            observation
        )

        self._sequence = max(
            self._sequence,
            observation.sequence,
        )

    def _load_warning(
        self,
        value: AgentWarning | Mapping[str, Any] | str,
    ) -> None:
        """
        Memuat warning.
        """

        if isinstance(
            value,
            AgentWarning,
        ):
            warning = value

        elif isinstance(
            value,
            Mapping,
        ):
            warning = AgentWarning(
                message=str(
                    value.get(
                        "message",
                        "",
                    )
                ),
                code=value.get(
                    "code"
                ),
                data=dict(
                    value.get(
                        "data",
                        {},
                    )
                ),
                timestamp=str(
                    value.get(
                        "timestamp",
                        self.utcnow_iso(),
                    )
                ),
                sequence=int(
                    value.get(
                        "sequence",
                        self._next_sequence(),
                    )
                ),
            )

        else:
            warning = AgentWarning(
                message=str(value),
                timestamp=self.utcnow_iso(),
                sequence=self._next_sequence(),
            )

        self.warnings.append(
            warning
        )

        self._sequence = max(
            self._sequence,
            warning.sequence,
        )

    def _load_error(
        self,
        value: AgentError | Mapping[str, Any] | str,
    ) -> None:
        """
        Memuat error.
        """

        if isinstance(
            value,
            AgentError,
        ):
            error = value

        elif isinstance(
            value,
            Mapping,
        ):
            error = AgentError(
                message=str(
                    value.get(
                        "message",
                        "",
                    )
                ),
                code=value.get(
                    "code"
                ),
                error_type=value.get(
                    "error_type"
                ),
                data=dict(
                    value.get(
                        "data",
                        {},
                    )
                ),
                timestamp=str(
                    value.get(
                        "timestamp",
                        self.utcnow_iso(),
                    )
                ),
                sequence=int(
                    value.get(
                        "sequence",
                        self._next_sequence(),
                    )
                ),
            )

        else:
            error = AgentError(
                message=str(value),
                timestamp=self.utcnow_iso(),
                sequence=self._next_sequence(),
            )

        self.errors.append(
            error
        )

        self._sequence = max(
            self._sequence,
            error.sequence,
        )

    # ============================================================
    # FACTORY METHODS
    # ============================================================

    @classmethod
    def running(
        cls,
        agent: str,
        task: str,
        *,
        response: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        """
        Factory result running.
        """

        return cls(
            success=False,
            agent=agent,
            response=response,
            task=task,
            status=cls.STATUS_RUNNING,
            metadata=metadata,
        )

    @classmethod
    def success_result(
        cls,
        agent: str,
        task: str,
        response: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        """
        Factory result sukses.
        """

        result = cls(
            success=True,
            agent=agent,
            response=response,
            task=task,
            status=cls.STATUS_COMPLETED,
            metadata=metadata,
        )

        result.completed_at = (
            result.utcnow_iso()
        )

        return result

    @classmethod
    def failure_result(
        cls,
        agent: str,
        task: str,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        code: str | None = None,
        error_type: str | None = None,
    ) -> AgentResult:
        """
        Factory result failure.
        """

        result = cls(
            success=False,
            agent=agent,
            response="",
            task=task,
            status=cls.STATUS_RUNNING,
            metadata=metadata,
        )

        result.add_error(
            message,
            code=code,
            error_type=error_type,
        )

        return result

    # ============================================================
    # MERGE
    # ============================================================

    def merge(
        self,
        other: AgentResult,
        *,
        include_response: bool = True,
        include_metadata: bool = True,
        include_observations: bool = True,
        include_warnings: bool = True,
        include_errors: bool = True,
    ) -> AgentResult:
        """
        Menggabungkan result lain ke result ini.
        """

        if not isinstance(
            other,
            AgentResult,
        ):
            raise TypeError(
                "other harus merupakan "
                "AgentResult."
            )

        if include_response and other.response:
            self.response = other.response

        if include_metadata:
            self.metadata.update(
                other.metadata
            )

        if include_observations:
            for item in other.observations:
                self.add_observation(
                    item.event,
                    **dict(item.data),
                )

        if include_warnings:
            for item in other.warnings:
                self.add_warning(
                    item.message,
                    code=item.code,
                    **dict(item.data),
                )

        if include_errors:
            for item in other.errors:
                self.add_error(
                    item.message,
                    code=item.code,
                    error_type=item.error_type,
                    **dict(item.data),
                )

        self.success = other.success
        self.status = other.status

        if other.completed_at:
            self.completed_at = (
                other.completed_at
            )

        return self

    # ============================================================
    # OBSERVATION SEARCH
    # ============================================================

    def find_observations(
        self,
        event: str,
    ) -> list[AgentObservation]:
        """
        Mengambil semua observation berdasarkan event.
        """

        return [
            item
            for item in self.observations
            if item.event == event
        ]

    def has_observation(
        self,
        event: str,
    ) -> bool:
        """
        Mengecek keberadaan event.
        """

        return any(
            item.event == event
            for item in self.observations
        )

    # ============================================================
    # WARNING SEARCH
    # ============================================================

    def find_warnings(
        self,
        code: str,
    ) -> list[AgentWarning]:
        """
        Mengambil warning berdasarkan code.
        """

        return [
            item
            for item in self.warnings
            if item.code == code
        ]

    # ============================================================
    # ERROR SEARCH
    # ============================================================

    def find_errors(
        self,
        code: str,
    ) -> list[AgentError]:
        """
        Mengambil error berdasarkan code.
        """

        return [
            item
            for item in self.errors
            if item.code == code
        ]

    # ============================================================
    # DISPLAY
    # ============================================================

    def __repr__(self) -> str:
        """
        Representation developer-friendly.
        """

        return (
            "AgentResult("
            f"success={self.success!r}, "
            f"agent={self.agent!r}, "
            f"status={self.status!r}, "
            f"execution_id={self.execution_id!r}, "
            f"observation_count="
            f"{self.observation_count}, "
            f"warning_count="
            f"{self.warning_count}, "
            f"error_count="
            f"{self.error_count}"
            ")"
        )

    def __str__(self) -> str:
        """
        String representation.
        """

        return self.response

    # ============================================================
    # ITERATION / CONTAINER HELPERS
    # ============================================================

    def __len__(self) -> int:
        """
        Jumlah total event/error/warning.
        """

        return (
            self.observation_count
            + self.warning_count
            + self.error_count
        )

    def __bool__(self) -> bool:
        """
        AgentResult dianggap truthy jika success.
        """

        return self.success

    # ============================================================
    # EXPORT
    # ============================================================

    def export(
        self,
        *,
        compact: bool = False,
    ) -> dict[str, Any]:
        """
        Export result.

        compact=True:
            hanya summary.

        compact=False:
            full result.
        """

        if compact:
            return self.summary()

        return self.to_json_dict()

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate(self) -> AgentResult:
        """
        Memvalidasi internal state.

        Method mengembalikan self jika valid.
        """

        if not self.agent.strip():
            raise InvalidAgentResultError(
                "Agent kosong."
            )

        if self.status not in self.VALID_STATUSES:
            raise InvalidAgentResultError(
                f"Status invalid: {self.status}"
            )

        if not self.execution_id:
            raise InvalidAgentResultError(
                "Execution ID kosong."
            )

        if not self.created_at:
            raise InvalidAgentResultError(
                "created_at kosong."
            )

        if (
            self.status
            == self.STATUS_COMPLETED
            and not self.success
        ):
            raise InvalidAgentResultError(
                "Result completed harus success=True."
            )

        if (
            self.status
            in self.FAILURE_STATUSES
            and self.success
        ):
            raise InvalidAgentResultError(
                "Result failure tidak boleh success=True."
            )

        return self


# =================================================================
# PUBLIC EXPORTS
# =================================================================

__all__ = [
    "AgentResult",
    "AgentResultError",
    "InvalidAgentResultError",
    "AgentObservation",
    "AgentWarning",
    "AgentError",
]
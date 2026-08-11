from __future__ import annotations

"""
ZAI Agent Result
================

Central result object untuk seluruh sistem agent ZAI.

File:
    ai/agents/agent_result.py

Tujuan utama:
    - Menyimpan hasil eksekusi agent.
    - Menyimpan response agent.
    - Menyimpan status task.
    - Menyimpan observation/event.
    - Menyimpan warning.
    - Menyimpan error.
    - Menyimpan metadata.
    - Mendukung chaining API.
    - Mendukung serialisasi ke dictionary.
    - Mendukung serialisasi JSON.
    - Mendukung cloning.
    - Mendukung merge result.
    - Mendukung lifecycle:
        running
        completed
        failed
        cancelled
        skipped
    - Menjadi fondasi result system ZAI ke depannya.

Versi:
    2.2.0

Catatan:
    File ini tidak bergantung pada module ZAI lain sehingga aman
    untuk di-import dari BaseAgent, Runtime, Registry, maupun agent.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional
from uuid import UUID, uuid4
import copy
import json


# ============================================================================
# CONSTANTS
# ============================================================================

RESULT_VERSION = "2.2.0"

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_SKIPPED = "skipped"

VALID_STATUSES = {
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_SKIPPED,
}

TERMINAL_STATUSES = {
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_SKIPPED,
}

SUCCESS_STATUSES = {
    STATUS_COMPLETED,
}

FAILURE_STATUSES = {
    STATUS_FAILED,
}

DEFAULT_AGENT = "unknown_agent"

DEFAULT_RESPONSE = ""

DEFAULT_TASK = ""

DEFAULT_STATUS = STATUS_RUNNING


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _utc_now() -> datetime:
    """
    Menghasilkan timestamp UTC timezone-aware.
    """
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    """
    Menghasilkan timestamp ISO 8601.
    """
    return _utc_now().isoformat()


def _ensure_string(value: Any, default: str = "") -> str:
    """
    Memastikan nilai menjadi string.
    """
    if value is None:
        return default

    if isinstance(value, str):
        return value

    return str(value)


def _ensure_dict(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Mengubah mapping menjadi dictionary baru.
    """
    if value is None:
        return {}

    return dict(value)


def _safe_copy(value: Any) -> Any:
    """
    Deep copy dengan fallback apabila object tidak bisa dicopy.
    """
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _json_default(value: Any) -> Any:
    """
    JSON fallback untuk object yang tidak langsung serializable.
    """

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, tuple):
        return list(value)

    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass

    return str(value)


def _normalize_status(status: Any) -> str:
    """
    Normalisasi status result.
    """

    normalized = _ensure_string(
        status,
        DEFAULT_STATUS,
    ).strip().lower()

    if normalized not in VALID_STATUSES:
        raise ValueError(
            f"Status result tidak valid: '{normalized}'. "
            f"Gunakan salah satu: {sorted(VALID_STATUSES)}"
        )

    return normalized


# ============================================================================
# OBSERVATION
# ============================================================================


@dataclass
class AgentObservation:
    """
    Event/observation yang dihasilkan selama lifecycle agent.
    """

    event: str

    data: dict[str, Any] = field(
        default_factory=dict,
    )

    timestamp: str = field(
        default_factory=_timestamp,
    )

    sequence: int = 0

    def __post_init__(self) -> None:
        self.event = _ensure_string(
            self.event,
            "unknown_event",
        )

        self.data = _ensure_dict(
            self.data,
        )

        self.sequence = int(
            self.sequence,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Mengubah observation menjadi dictionary.
        """

        return {
            "event": self.event,
            "data": _safe_copy(self.data),
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }

    def clone(self) -> AgentObservation:
        """
        Membuat salinan observation.
        """

        return AgentObservation(
            event=self.event,
            data=_safe_copy(self.data),
            timestamp=self.timestamp,
            sequence=self.sequence,
        )


# ============================================================================
# AGENT ERROR
# ============================================================================


@dataclass
class AgentError:
    """
    Struktur error yang dihasilkan agent.
    """

    message: str

    error_type: str = "AgentError"

    timestamp: str = field(
        default_factory=_timestamp,
    )

    data: dict[str, Any] = field(
        default_factory=dict,
    )

    sequence: int = 0

    def __post_init__(self) -> None:
        self.message = _ensure_string(
            self.message,
        )

        self.error_type = _ensure_string(
            self.error_type,
            "AgentError",
        )

        self.data = _ensure_dict(
            self.data,
        )

        self.sequence = int(
            self.sequence,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisasi error.
        """

        return {
            "message": self.message,
            "type": self.error_type,
            "timestamp": self.timestamp,
            "data": _safe_copy(self.data),
            "sequence": self.sequence,
        }

    def clone(self) -> AgentError:
        """
        Clone error.
        """

        return AgentError(
            message=self.message,
            error_type=self.error_type,
            timestamp=self.timestamp,
            data=_safe_copy(self.data),
            sequence=self.sequence,
        )


# ============================================================================
# AGENT WARNING
# ============================================================================


@dataclass
class AgentWarning:
    """
    Struktur warning yang dihasilkan agent.
    """

    message: str

    warning_type: str = "AgentWarning"

    timestamp: str = field(
        default_factory=_timestamp,
    )

    data: dict[str, Any] = field(
        default_factory=dict,
    )

    sequence: int = 0

    def __post_init__(self) -> None:
        self.message = _ensure_string(
            self.message,
        )

        self.warning_type = _ensure_string(
            self.warning_type,
            "AgentWarning",
        )

        self.data = _ensure_dict(
            self.data,
        )

        self.sequence = int(
            self.sequence,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisasi warning.
        """

        return {
            "message": self.message,
            "type": self.warning_type,
            "timestamp": self.timestamp,
            "data": _safe_copy(self.data),
            "sequence": self.sequence,
        }

    def clone(self) -> AgentWarning:
        """
        Clone warning.
        """

        return AgentWarning(
            message=self.message,
            warning_type=self.warning_type,
            timestamp=self.timestamp,
            data=_safe_copy(self.data),
            sequence=self.sequence,
        )


# ============================================================================
# AGENT RESULT
# ============================================================================


@dataclass
class AgentResult:
    """
    Result utama dari eksekusi ZAI Agent.

    Contoh:

        result = AgentResult(
            success=True,
            agent="general_agent",
            response="Halo!",
            task="Halo ZAI",
            status="running",
        )

        result.add_observation(
            "task_received",
            task="Halo ZAI",
        )

        result.complete(
            "Halo! Saya ZAI.",
        )

    API ini sengaja dibuat fluent/chaining:

        result \
            .add_observation("started") \
            .set_metadata("source", "runtime") \
            .complete("OK")
    """

    success: bool = True

    agent: str = DEFAULT_AGENT

    response: str = DEFAULT_RESPONSE

    task: str = DEFAULT_TASK

    status: str = DEFAULT_STATUS

    execution_id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    created_at: str = field(
        default_factory=_timestamp,
    )

    completed_at: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    observations: list[AgentObservation] = field(
        default_factory=list,
    )

    errors: list[AgentError] = field(
        default_factory=list,
    )

    warnings: list[AgentWarning] = field(
        default_factory=list,
    )

    result_version: str = RESULT_VERSION

    def __post_init__(self) -> None:
        """
        Normalisasi object setelah dataclass dibuat.
        """

        self.success = bool(
            self.success,
        )

        self.agent = _ensure_string(
            self.agent,
            DEFAULT_AGENT,
        )

        self.response = _ensure_string(
            self.response,
        )

        self.task = _ensure_string(
            self.task,
        )

        self.status = _normalize_status(
            self.status,
        )

        self.execution_id = _ensure_string(
            self.execution_id,
            str(uuid4()),
        )

        self.created_at = _ensure_string(
            self.created_at,
            _timestamp(),
        )

        if self.completed_at is not None:
            self.completed_at = _ensure_string(
                self.completed_at,
            )

        self.metadata = _ensure_dict(
            self.metadata,
        )

        self.result_version = _ensure_string(
            self.result_version,
            RESULT_VERSION,
        )

        self._normalize_observations()

        self._normalize_errors()

        self._normalize_warnings()

        self._synchronize_success_with_status()

    # ------------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------------

    def _normalize_observations(self) -> None:
        """
        Memastikan seluruh observation berbentuk AgentObservation.
        """

        normalized: list[AgentObservation] = []

        for index, item in enumerate(
            self.observations,
            start=1,
        ):
            if isinstance(
                item,
                AgentObservation,
            ):
                observation = item
            elif isinstance(
                item,
                Mapping,
            ):
                observation = AgentObservation(
                    event=_ensure_string(
                        item.get(
                            "event",
                            "unknown_event",
                        ),
                    ),
                    data=_ensure_dict(
                        item.get(
                            "data",
                            {},
                        ),
                    ),
                    timestamp=_ensure_string(
                        item.get(
                            "timestamp",
                            _timestamp(),
                        ),
                    ),
                    sequence=int(
                        item.get(
                            "sequence",
                            index,
                        ),
                    ),
                )
            else:
                observation = AgentObservation(
                    event=_ensure_string(
                        item,
                        "unknown_event",
                    ),
                    sequence=index,
                )

            if observation.sequence <= 0:
                observation.sequence = index

            normalized.append(
                observation,
            )

        self.observations = normalized

    def _normalize_errors(self) -> None:
        """
        Memastikan error tersimpan sebagai AgentError.
        """

        normalized: list[AgentError] = []

        for index, item in enumerate(
            self.errors,
            start=1,
        ):
            if isinstance(
                item,
                AgentError,
            ):
                error = item
            elif isinstance(
                item,
                Mapping,
            ):
                error = AgentError(
                    message=_ensure_string(
                        item.get(
                            "message",
                            "",
                        ),
                    ),
                    error_type=_ensure_string(
                        item.get(
                            "type",
                            "AgentError",
                        ),
                    ),
                    timestamp=_ensure_string(
                        item.get(
                            "timestamp",
                            _timestamp(),
                        ),
                    ),
                    data=_ensure_dict(
                        item.get(
                            "data",
                            {},
                        ),
                    ),
                    sequence=int(
                        item.get(
                            "sequence",
                            index,
                        ),
                    ),
                )
            else:
                error = AgentError(
                    message=_ensure_string(
                        item,
                    ),
                    sequence=index,
                )

            if error.sequence <= 0:
                error.sequence = index

            normalized.append(
                error,
            )

        self.errors = normalized

    def _normalize_warnings(self) -> None:
        """
        Memastikan warning tersimpan sebagai AgentWarning.
        """

        normalized: list[AgentWarning] = []

        for index, item in enumerate(
            self.warnings,
            start=1,
        ):
            if isinstance(
                item,
                AgentWarning,
            ):
                warning = item
            elif isinstance(
                item,
                Mapping,
            ):
                warning = AgentWarning(
                    message=_ensure_string(
                        item.get(
                            "message",
                            "",
                        ),
                    ),
                    warning_type=_ensure_string(
                        item.get(
                            "type",
                            "AgentWarning",
                        ),
                    ),
                    timestamp=_ensure_string(
                        item.get(
                            "timestamp",
                            _timestamp(),
                        ),
                    ),
                    data=_ensure_dict(
                        item.get(
                            "data",
                            {},
                        ),
                    ),
                    sequence=int(
                        item.get(
                            "sequence",
                            index,
                        ),
                    ),
                )
            else:
                warning = AgentWarning(
                    message=_ensure_string(
                        item,
                    ),
                    sequence=index,
                )

            if warning.sequence <= 0:
                warning.sequence = index

            normalized.append(
                warning,
            )

        self.warnings = normalized

    def _synchronize_success_with_status(self) -> None:
        """
        Menjaga success dan status agar konsisten.

        completed => success True
        failed    => success False

        Status running tidak memaksa nilai success.
        """

        if self.status == STATUS_COMPLETED:
            self.success = True

        elif self.status == STATUS_FAILED:
            self.success = False

    # ------------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------------

    @property
    def observation_count(self) -> int:
        """
        Jumlah observation.
        """

        return len(
            self.observations,
        )

    @property
    def error_count(self) -> int:
        """
        Jumlah error.
        """

        return len(
            self.errors,
        )

    @property
    def warning_count(self) -> int:
        """
        Jumlah warning.
        """

        return len(
            self.warnings,
        )

    @property
    def has_errors(self) -> bool:
        """
        True jika result memiliki error.
        """

        return bool(
            self.errors,
        )

    @property
    def has_warnings(self) -> bool:
        """
        True jika result memiliki warning.
        """

        return bool(
            self.warnings,
        )

    @property
    def is_running(self) -> bool:
        """
        True jika result sedang berjalan.
        """

        return self.status == STATUS_RUNNING

    @property
    def is_completed(self) -> bool:
        """
        True jika task selesai sukses.
        """

        return self.status == STATUS_COMPLETED

    @property
    def is_failed(self) -> bool:
        """
        True jika task gagal.
        """

        return self.status == STATUS_FAILED

    @property
    def is_cancelled(self) -> bool:
        """
        True jika task dibatalkan.
        """

        return self.status == STATUS_CANCELLED

    @property
    def is_skipped(self) -> bool:
        """
        True jika task dilewati.
        """

        return self.status == STATUS_SKIPPED

    @property
    def is_terminal(self) -> bool:
        """
        True jika lifecycle sudah mencapai state akhir.
        """

        return self.status in TERMINAL_STATUSES

    @property
    def duration_available(self) -> bool:
        """
        True jika created_at dan completed_at tersedia.
        """

        return (
            self.created_at is not None
            and self.completed_at is not None
        )

    @property
    def latest_observation(
        self,
    ) -> Optional[AgentObservation]:
        """
        Observation terakhir.
        """

        if not self.observations:
            return None

        return self.observations[-1]

    @property
    def latest_error(
        self,
    ) -> Optional[AgentError]:
        """
        Error terakhir.
        """

        if not self.errors:
            return None

        return self.errors[-1]

    @property
    def latest_warning(
        self,
    ) -> Optional[AgentWarning]:
        """
        Warning terakhir.
        """

        if not self.warnings:
            return None

        return self.warnings[-1]

    # ------------------------------------------------------------------------
    # OBSERVATION API
    # ------------------------------------------------------------------------

    def add_observation(
        self,
        event: str,
        **data: Any,
    ) -> AgentResult:
        """
        Menambahkan observation.

        Contoh:

            result.add_observation(
                "task_received",
                task=task,
            )
        """

        observation = AgentObservation(
            event=_ensure_string(
                event,
                "unknown_event",
            ),
            data=data,
            sequence=self.observation_count + 1,
        )

        self.observations.append(
            observation,
        )

        return self

    def add_observation_data(
        self,
        event: str,
        data: Mapping[str, Any],
    ) -> AgentResult:
        """
        Alternatif add_observation dengan mapping.
        """

        observation = AgentObservation(
            event=_ensure_string(
                event,
                "unknown_event",
            ),
            data=dict(data),
            sequence=self.observation_count + 1,
        )

        self.observations.append(
            observation,
        )

        return self

    def clear_observations(
        self,
    ) -> AgentResult:
        """
        Menghapus seluruh observation.
        """

        self.observations.clear()

        return self

    # ------------------------------------------------------------------------
    # WARNING API
    # ------------------------------------------------------------------------

    def add_warning(
        self,
        message: str,
        warning_type: str = "AgentWarning",
        **data: Any,
    ) -> AgentResult:
        """
        Menambahkan warning.

        Contoh:

            result.add_warning(
                "Model belum tersedia",
            )
        """

        warning = AgentWarning(
            message=_ensure_string(
                message,
            ),
            warning_type=_ensure_string(
                warning_type,
                "AgentWarning",
            ),
            data=data,
            sequence=self.warning_count + 1,
        )

        self.warnings.append(
            warning,
        )

        return self

    def clear_warnings(
        self,
    ) -> AgentResult:
        """
        Menghapus seluruh warning.
        """

        self.warnings.clear()

        return self

    # ------------------------------------------------------------------------
    # ERROR API
    # ------------------------------------------------------------------------

    def add_error(
        self,
        message: Any,
        error_type: Optional[str] = None,
        **data: Any,
    ) -> AgentResult:
        """
        Menambahkan error.

        PENTING:
            Menambahkan error otomatis membuat result failed.

        Contoh:

            result.add_error(
                "Connection failed",
            )
        """

        if isinstance(
            message,
            BaseException,
        ):
            if error_type is None:
                error_type = type(
                    message,
                ).__name__

            message_text = str(
                message,
            )
        else:
            message_text = _ensure_string(
                message,
            )

            if error_type is None:
                error_type = "AgentError"

        error = AgentError(
            message=message_text,
            error_type=_ensure_string(
                error_type,
                "AgentError",
            ),
            data=data,
            sequence=self.error_count + 1,
        )

        self.errors.append(
            error,
        )

        self.success = False

        self.status = STATUS_FAILED

        self.completed_at = _timestamp()

        return self

    def add_exception(
        self,
        exc: BaseException,
        **data: Any,
    ) -> AgentResult:
        """
        Shortcut untuk exception.
        """

        return self.add_error(
            exc,
            type(exc).__name__,
            **data,
        )

    def clear_errors(
        self,
    ) -> AgentResult:
        """
        Menghapus seluruh error.

        Tidak otomatis mengubah status karena status lifecycle
        harus dikontrol secara eksplisit.
        """

        self.errors.clear()

        return self

    # ------------------------------------------------------------------------
    # METADATA API
    # ------------------------------------------------------------------------

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> AgentResult:
        """
        Menetapkan satu metadata.

        Mendukung chaining:

            result.set_metadata(
                "source",
                "runtime",
            ).set_metadata(
                "priority",
                "high",
            )
        """

        self.metadata[
            _ensure_string(key)
        ] = value

        return self

    def set_metadata_many(
        self,
        values: Mapping[str, Any],
    ) -> AgentResult:
        """
        Menetapkan banyak metadata sekaligus.
        """

        for key, value in values.items():
            self.metadata[
                _ensure_string(key)
            ] = value

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
        Mengecek keberadaan metadata.
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

    def clear_metadata(
        self,
    ) -> AgentResult:
        """
        Menghapus seluruh metadata.
        """

        self.metadata.clear()

        return self

    # ------------------------------------------------------------------------
    # RESPONSE API
    # ------------------------------------------------------------------------

    def set_response(
        self,
        response: Any,
    ) -> AgentResult:
        """
        Mengubah response.
        """

        self.response = _ensure_string(
            response,
        )

        return self

    def append_response(
        self,
        response: Any,
        separator: str = "",
    ) -> AgentResult:
        """
        Menambahkan teks ke response.
        """

        addition = _ensure_string(
            response,
        )

        if not self.response:
            self.response = addition

        else:
            self.response = (
                self.response
                + separator
                + addition
            )

        return self

    def prepend_response(
        self,
        response: Any,
        separator: str = "",
    ) -> AgentResult:
        """
        Menambahkan teks di depan response.
        """

        prefix = _ensure_string(
            response,
        )

        if not self.response:
            self.response = prefix

        else:
            self.response = (
                prefix
                + separator
                + self.response
            )

        return self

    # ------------------------------------------------------------------------
    # TASK API
    # ------------------------------------------------------------------------

    def set_task(
        self,
        task: Any,
    ) -> AgentResult:
        """
        Mengubah task.
        """

        self.task = _ensure_string(
            task,
        )

        return self

    # ------------------------------------------------------------------------
    # STATUS API
    # ------------------------------------------------------------------------

    def set_status(
        self,
        status: str,
    ) -> AgentResult:
        """
        Mengubah status dengan validasi.
        """

        normalized = _normalize_status(
            status,
        )

        self.status = normalized

        if normalized == STATUS_COMPLETED:
            self.success = True

        elif normalized == STATUS_FAILED:
            self.success = False

        if normalized in TERMINAL_STATUSES:
            if self.completed_at is None:
                self.completed_at = _timestamp()

        return self

    def start(
        self,
    ) -> AgentResult:
        """
        Memulai lifecycle result.
        """

        self.status = STATUS_RUNNING

        self.success = True

        self.completed_at = None

        self.add_observation(
            "result_started",
            agent=self.agent,
            execution_id=self.execution_id,
        )

        return self

    # ------------------------------------------------------------------------
    # COMPLETE API
    # ------------------------------------------------------------------------

    def complete(
        self,
        response: Optional[Any] = None,
        **metadata: Any,
    ) -> AgentResult:
        """
        MENANDAI RESULT BERHASIL SELESAI.

        Ini adalah method yang sebelumnya hilang dan menyebabkan:

            AttributeError:
                'AgentResult' object has no attribute 'complete'

        Contoh:

            result.complete(
                "ZAI test berhasil",
                latency_ms=1.0,
            )

        Behavior:

            success = True
            status = completed
            completed_at = timestamp
            metadata diperbarui
        """

        if response is not None:
            self.response = _ensure_string(
                response,
            )

        if metadata:
            self.metadata.update(
                metadata,
            )

        self.success = True

        self.status = STATUS_COMPLETED

        self.completed_at = _timestamp()

        self.add_observation(
            "result_completed",
            response_length=len(
                self.response,
            ),
        )

        return self

    def succeed(
        self,
        response: Optional[Any] = None,
        **metadata: Any,
    ) -> AgentResult:
        """
        Alias untuk complete().
        """

        return self.complete(
            response,
            **metadata,
        )

    # ------------------------------------------------------------------------
    # FAIL API
    # ------------------------------------------------------------------------

    def fail(
        self,
        message: Optional[Any] = None,
        error_type: Optional[str] = None,
        **data: Any,
    ) -> AgentResult:
        """
        Menandai result sebagai gagal.

        Contoh:

            result.fail(
                "Ollama tidak tersedia",
            )
        """

        if message is not None:
            self.add_error(
                message,
                error_type,
                **data,
            )

        else:
            self.success = False

            self.status = STATUS_FAILED

            self.completed_at = _timestamp()

        return self

    def failure(
        self,
        message: Optional[Any] = None,
        error_type: Optional[str] = None,
        **data: Any,
    ) -> AgentResult:
        """
        Alias fail().
        """

        return self.fail(
            message,
            error_type,
            **data,
        )

    # ------------------------------------------------------------------------
    # CANCEL API
    # ------------------------------------------------------------------------

    def cancel(
        self,
        reason: Optional[Any] = None,
    ) -> AgentResult:
        """
        Membatalkan task.
        """

        self.status = STATUS_CANCELLED

        self.success = False

        self.completed_at = _timestamp()

        if reason is not None:
            self.add_observation(
                "task_cancelled",
                reason=_ensure_string(
                    reason,
                ),
            )

        return self

    # ------------------------------------------------------------------------
    # SKIP API
    # ------------------------------------------------------------------------

    def skip(
        self,
        reason: Optional[Any] = None,
    ) -> AgentResult:
        """
        Menandai task sebagai skipped.
        """

        self.status = STATUS_SKIPPED

        self.success = False

        self.completed_at = _timestamp()

        if reason is not None:
            self.add_observation(
                "task_skipped",
                reason=_ensure_string(
                    reason,
                ),
            )

        return self

    # ------------------------------------------------------------------------
    # RESET API
    # ------------------------------------------------------------------------

    def reset(
        self,
        *,
        preserve_execution_id: bool = False,
        preserve_task: bool = True,
        preserve_agent: bool = True,
    ) -> AgentResult:
        """
        Mengembalikan result ke state awal.

        Cocok jika object ingin digunakan ulang.
        """

        if not preserve_execution_id:
            self.execution_id = str(
                uuid4(),
            )

        if not preserve_task:
            self.task = ""

        if not preserve_agent:
            self.agent = DEFAULT_AGENT

        self.success = True

        self.response = ""

        self.status = STATUS_RUNNING

        self.created_at = _timestamp()

        self.completed_at = None

        self.metadata.clear()

        self.observations.clear()

        self.errors.clear()

        self.warnings.clear()

        return self

    # ------------------------------------------------------------------------
    # VALIDATION API
    # ------------------------------------------------------------------------

    def validate(
        self,
    ) -> list[str]:
        """
        Memvalidasi result.

        Mengembalikan list error validasi.
        List kosong berarti valid.
        """

        problems: list[str] = []

        if not self.agent:
            problems.append(
                "agent tidak boleh kosong.",
            )

        if not self.execution_id:
            problems.append(
                "execution_id tidak boleh kosong.",
            )

        if self.status not in VALID_STATUSES:
            problems.append(
                f"status tidak valid: {self.status}",
            )

        if self.status == STATUS_COMPLETED:
            if not self.success:
                problems.append(
                    "status completed harus success=True.",
                )

        if self.status == STATUS_FAILED:
            if self.success:
                problems.append(
                    "status failed harus success=False.",
                )

        if self.status in TERMINAL_STATUSES:
            if not self.completed_at:
                problems.append(
                    "terminal result harus memiliki completed_at.",
                )

        return problems

    @property
    def is_valid(self) -> bool:
        """
        True jika result valid.
        """

        return not self.validate()

    # ------------------------------------------------------------------------
    # SUMMARY API
    # ------------------------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Ringkasan result.
        """

        return {
            "success": self.success,
            "agent": self.agent,
            "status": self.status,
            "execution_id": self.execution_id,
            "task": self.task,
            "response_length": len(
                self.response,
            ),
            "observation_count": self.observation_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "is_terminal": self.is_terminal,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    def stats(
        self,
    ) -> dict[str, Any]:
        """
        Statistik result.
        """

        return {
            "observation_count": self.observation_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "response_length": len(
                self.response,
            ),
            "metadata_count": len(
                self.metadata,
            ),
            "success": self.success,
            "status": self.status,
        }

    # ------------------------------------------------------------------------
    # SERIALIZATION API
    # ------------------------------------------------------------------------

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Serialisasi penuh result.

        Format kompatibel dengan test yang sedang kita gunakan.
        """

        return {
            "success": self.success,
            "agent": self.agent,
            "response": self.response,
            "task": self.task,
            "status": self.status,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": _safe_copy(
                self.metadata,
            ),
            "observation_count": self.observation_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "observations": [
                observation.to_dict()
                for observation in self.observations
            ],
            "warnings": [
                warning.to_dict()
                for warning in self.warnings
            ],
            "errors": [
                error.to_dict()
                for error in self.errors
            ],
            "result_version": self.result_version,
        }

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """
        Alias to_dict().
        """

        return self.to_dict()

    def to_json(
        self,
        *,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
    ) -> str:
        """
        Serialisasi JSON.
        """

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=_json_default,
        )

    def as_json(
        self,
        *,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
    ) -> str:
        """
        Alias to_json().
        """

        return self.to_json(
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    # ------------------------------------------------------------------------
    # CLONE API
    # ------------------------------------------------------------------------

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
                str(uuid4())
                if new_execution_id
                else self.execution_id
            ),
            created_at=self.created_at,
            completed_at=self.completed_at,
            metadata=_safe_copy(
                self.metadata,
            ),
            observations=[
                item.clone()
                for item in self.observations
            ],
            errors=[
                item.clone()
                for item in self.errors
            ],
            warnings=[
                item.clone()
                for item in self.warnings
            ],
            result_version=self.result_version,
        )

        return cloned

    # ------------------------------------------------------------------------
    # MERGE API
    # ------------------------------------------------------------------------

    def merge(
        self,
        other: AgentResult,
        *,
        include_response: bool = False,
        include_metadata: bool = True,
        include_observations: bool = True,
        include_warnings: bool = True,
        include_errors: bool = True,
    ) -> AgentResult:
        """
        Menggabungkan result lain ke result ini.

        Berguna untuk multi-agent pipeline.
        """

        if not isinstance(
            other,
            AgentResult,
        ):
            raise TypeError(
                "other harus merupakan AgentResult.",
            )

        if include_response and other.response:
            self.response = other.response

        if include_metadata:
            self.metadata.update(
                _safe_copy(
                    other.metadata,
                ),
            )

        if include_observations:
            for observation in other.observations:
                cloned = observation.clone()

                cloned.sequence = (
                    self.observation_count + 1
                )

                self.observations.append(
                    cloned,
                )

        if include_warnings:
            for warning in other.warnings:
                cloned = warning.clone()

                cloned.sequence = (
                    self.warning_count + 1
                )

                self.warnings.append(
                    cloned,
                )

        if include_errors:
            for error in other.errors:
                cloned = error.clone()

                cloned.sequence = (
                    self.error_count + 1
                )

                self.errors.append(
                    cloned,
                )

        if other.status == STATUS_FAILED:
            self.success = False
            self.status = STATUS_FAILED

        elif (
            other.status == STATUS_COMPLETED
            and self.status == STATUS_RUNNING
        ):
            self.success = True
            self.status = STATUS_COMPLETED
            self.completed_at = (
                other.completed_at
            )

        return self

    # ------------------------------------------------------------------------
    # OBSERVATION SEARCH
    # ------------------------------------------------------------------------

    def find_observations(
        self,
        event: str,
    ) -> list[AgentObservation]:
        """
        Mencari observation berdasarkan nama event.
        """

        return [
            observation
            for observation in self.observations
            if observation.event == event
        ]

    def has_observation(
        self,
        event: str,
    ) -> bool:
        """
        Mengecek apakah event pernah terjadi.
        """

        return any(
            observation.event == event
            for observation in self.observations
        )

    # ------------------------------------------------------------------------
    # ERROR SEARCH
    # ------------------------------------------------------------------------

    def find_errors(
        self,
        error_type: str,
    ) -> list[AgentError]:
        """
        Mencari error berdasarkan tipe.
        """

        return [
            error
            for error in self.errors
            if error.error_type == error_type
        ]

    def has_error_type(
        self,
        error_type: str,
    ) -> bool:
        """
        Mengecek tipe error.
        """

        return any(
            error.error_type == error_type
            for error in self.errors
        )

    # ------------------------------------------------------------------------
    # WARNING SEARCH
    # ------------------------------------------------------------------------

    def find_warnings(
        self,
        warning_type: str,
    ) -> list[AgentWarning]:
        """
        Mencari warning berdasarkan tipe.
        """

        return [
            warning
            for warning in self.warnings
            if warning.warning_type == warning_type
        ]

    def has_warning_type(
        self,
        warning_type: str,
    ) -> bool:
        """
        Mengecek tipe warning.
        """

        return any(
            warning.warning_type == warning_type
            for warning in self.warnings
        )

    # ------------------------------------------------------------------------
    # RESPONSE HELPERS
    # ------------------------------------------------------------------------

    def response_empty(
        self,
    ) -> bool:
        """
        True jika response kosong.
        """

        return not bool(
            self.response.strip(),
        )

    def response_available(
        self,
    ) -> bool:
        """
        True jika response tersedia.
        """

        return bool(
            self.response.strip(),
        )

    # ------------------------------------------------------------------------
    # EXECUTION INFORMATION
    # ------------------------------------------------------------------------

    def execution_info(
        self,
    ) -> dict[str, Any]:
        """
        Informasi execution.
        """

        return {
            "execution_id": self.execution_id,
            "agent": self.agent,
            "task": self.task,
            "status": self.status,
            "success": self.success,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    # ------------------------------------------------------------------------
    # DEBUG API
    # ------------------------------------------------------------------------

    def debug(
        self,
    ) -> dict[str, Any]:
        """
        Informasi debug lengkap.
        """

        return {
            "result": self.to_dict(),
            "summary": self.summary(),
            "stats": self.stats(),
            "validation": {
                "valid": self.is_valid,
                "problems": self.validate(),
            },
        }

    # ------------------------------------------------------------------------
    # STRING REPRESENTATION
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Representation untuk debugging.
        """

        return (
            "AgentResult("
            f"agent={self.agent!r}, "
            f"status={self.status!r}, "
            f"success={self.success!r}, "
            f"execution_id={self.execution_id!r}"
            ")"
        )

    def __str__(self) -> str:
        """
        String representation singkat.
        """

        return (
            f"[{self.status.upper()}] "
            f"{self.agent}: "
            f"{self.response}"
        )

    # ------------------------------------------------------------------------
    # ITERATION
    # ------------------------------------------------------------------------

    def __iter__(self):
        """
        Iterasi dictionary result.
        """

        return iter(
            self.to_dict().items(),
        )

    def keys(self):
        """
        Dictionary-like keys.
        """

        return self.to_dict().keys()

    def values(self):
        """
        Dictionary-like values.
        """

        return self.to_dict().values()

    def items(self):
        """
        Dictionary-like items.
        """

        return self.to_dict().items()

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Dictionary-like get.
        """

        return self.to_dict().get(
            key,
            default,
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_result(
    agent: str,
    task: str,
    *,
    response: str = "",
    status: str = STATUS_RUNNING,
    success: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AgentResult:
    """
    Factory membuat AgentResult.
    """

    return AgentResult(
        success=success,
        agent=agent,
        response=response,
        task=task,
        status=status,
        metadata=_ensure_dict(
            metadata,
        ),
    )


def success_result(
    agent: str,
    task: str,
    response: Any,
    **metadata: Any,
) -> AgentResult:
    """
    Factory result sukses.
    """

    result = AgentResult(
        success=True,
        agent=agent,
        response="",
        task=task,
        status=STATUS_RUNNING,
    )

    return result.complete(
        response,
        **metadata,
    )


def failure_result(
    agent: str,
    task: str,
    error: Any,
    error_type: Optional[str] = None,
    **metadata: Any,
) -> AgentResult:
    """
    Factory result gagal.
    """

    result = AgentResult(
        success=False,
        agent=agent,
        response="",
        task=task,
        status=STATUS_RUNNING,
        metadata=metadata,
    )

    result.add_error(
        error,
        error_type,
    )

    return result


def cancelled_result(
    agent: str,
    task: str,
    reason: Optional[str] = None,
) -> AgentResult:
    """
    Factory result cancelled.
    """

    result = AgentResult(
        success=False,
        agent=agent,
        response="",
        task=task,
        status=STATUS_RUNNING,
    )

    return result.cancel(
        reason,
    )


def skipped_result(
    agent: str,
    task: str,
    reason: Optional[str] = None,
) -> AgentResult:
    """
    Factory result skipped.
    """

    result = AgentResult(
        success=False,
        agent=agent,
        response="",
        task=task,
        status=STATUS_RUNNING,
    )

    return result.skip(
        reason,
    )


# ============================================================================
# RESULT COLLECTION
# ============================================================================


def results_to_dict(
    results: Iterable[AgentResult],
) -> list[dict[str, Any]]:
    """
    Mengubah collection result menjadi list dictionary.
    """

    return [
        result.to_dict()
        for result in results
    ]


def results_to_json(
    results: Iterable[AgentResult],
    *,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """
    Mengubah collection result menjadi JSON.
    """

    return json.dumps(
        results_to_dict(results),
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=_json_default,
    )


def count_successes(
    results: Iterable[AgentResult],
) -> int:
    """
    Menghitung result sukses.
    """

    return sum(
        1
        for result in results
        if result.success
    )


def count_failures(
    results: Iterable[AgentResult],
) -> int:
    """
    Menghitung result gagal.
    """

    return sum(
        1
        for result in results
        if not result.success
    )


def calculate_success_rate(
    results: Iterable[AgentResult],
) -> float:
    """
    Menghitung success rate.
    """

    result_list = list(
        results,
    )

    if not result_list:
        return 0.0

    successes = count_successes(
        result_list,
    )

    return round(
        (
            successes
            / len(result_list)
        )
        * 100,
        2,
    )


# ============================================================================
# RESULT FACTORY ALIASES
# ============================================================================


new_result = create_result

ok_result = success_result

error_result = failure_result


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================


__all__ = [
    "RESULT_VERSION",
    "STATUS_RUNNING",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_CANCELLED",
    "STATUS_SKIPPED",
    "VALID_STATUSES",
    "TERMINAL_STATUSES",
    "SUCCESS_STATUSES",
    "FAILURE_STATUSES",
    "AgentObservation",
    "AgentError",
    "AgentWarning",
    "AgentResult",
    "create_result",
    "success_result",
    "failure_result",
    "cancelled_result",
    "skipped_result",
    "results_to_dict",
    "results_to_json",
    "count_successes",
    "count_failures",
    "calculate_success_rate",
    "new_result",
    "ok_result",
    "error_result",
]
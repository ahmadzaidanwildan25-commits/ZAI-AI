from __future__ import annotations

"""
ZAI Memory Manager
==================

Memory subsystem untuk Super ZAI.

Tujuan utama:
- menyimpan memory secara terstruktur
- mengambil memory berdasarkan ID
- mencari memory berdasarkan query
- mendukung namespace
- mendukung kategori memory
- mendukung importance
- mendukung confidence
- mendukung tags
- mendukung metadata
- mendukung expiration / TTL
- mendukung pinning
- mendukung archive
- mendukung soft delete
- mendukung memory statistics
- mendukung ranking
- mendukung recent memories
- mendukung important memories
- mendukung contextual memories
- mendukung export/import JSON
- mendukung persistent storage lokal
- tidak membutuhkan dependency eksternal
- aman digunakan oleh ZAIBrain
- dapat diperluas ke vector database pada fase berikutnya

Memory Manager sengaja dibuat dependency-light agar fondasi ZAI
tetap stabil sebelum masuk ke semantic/vector memory.
"""

import hashlib
import json
import math
import re
import threading
import uuid

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence


# ============================================================================
# CONSTANTS
# ============================================================================


MEMORY_MANAGER_VERSION = "1.0.0"

DEFAULT_NAMESPACE = "default"

DEFAULT_CATEGORY = "general"

DEFAULT_IMPORTANCE = 0.5

DEFAULT_CONFIDENCE = 1.0

DEFAULT_MAX_RESULTS = 20

DEFAULT_STORAGE_FILENAME = "memory_store.json"

MAX_TEXT_LENGTH = 100_000

MAX_TAG_LENGTH = 100

MAX_NAMESPACE_LENGTH = 100

MAX_CATEGORY_LENGTH = 100

MAX_METADATA_DEPTH = 5


# ============================================================================
# ENUM-LIKE CONSTANTS
# ============================================================================


class MemoryStatus:
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    EXPIRED = "expired"


class MemoryCategory:
    GENERAL = "general"
    FACT = "fact"
    PREFERENCE = "preference"
    CONTEXT = "context"
    CONVERSATION = "conversation"
    TASK = "task"
    DECISION = "decision"
    GOAL = "goal"
    KNOWLEDGE = "knowledge"
    SYSTEM = "system"
    ERROR = "error"
    OBSERVATION = "observation"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def utc_now() -> datetime:
    """
    Return current UTC datetime.
    """
    return datetime.now().astimezone().astimezone(
        tz=None
    ).replace(tzinfo=None)


def iso_now() -> str:
    """
    Return current timestamp in ISO format.

    Timestamp sengaja disimpan sebagai string agar mudah
    diserialisasi ke JSON.
    """
    return datetime.now().astimezone().isoformat()


def parse_datetime(value: Any) -> datetime | None:
    """
    Parse datetime dari berbagai bentuk input.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if not isinstance(value, str):
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Batasi angka ke range tertentu.
    """
    return max(minimum, min(maximum, value))


def normalize_text(
    value: Any,
) -> str:
    """
    Normalisasi text untuk pencarian dan penyimpanan.
    """
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_key(
    value: Any,
    default: str,
) -> str:
    """
    Normalisasi namespace/category.
    """
    text = normalize_text(value)

    if not text:
        return default

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9_.:-]+",
        "_",
        text,
    )

    return text[:MAX_NAMESPACE_LENGTH]


def normalize_tags(
    tags: Iterable[Any] | None,
) -> list[str]:
    """
    Normalisasi tag dan menghapus duplicate.
    """
    if not tags:
        return []

    result: list[str] = []

    seen: set[str] = set()

    for raw_tag in tags:
        tag = normalize_text(raw_tag).lower()

        if not tag:
            continue

        tag = tag[:MAX_TAG_LENGTH]

        if tag in seen:
            continue

        seen.add(tag)

        result.append(tag)

    return result


def sanitize_metadata(
    metadata: Mapping[str, Any] | None,
    depth: int = 0,
) -> dict[str, Any]:
    """
    Sanitasi metadata agar aman disimpan ke JSON.
    """

    if metadata is None:
        return {}

    if depth > MAX_METADATA_DEPTH:
        return {
            "_truncated": True,
        }

    result: dict[str, Any] = {}

    for key, value in metadata.items():

        safe_key = normalize_text(key)

        if not safe_key:
            continue

        if isinstance(
            value,
            Mapping,
        ):
            result[safe_key] = sanitize_metadata(
                value,
                depth + 1,
            )
            continue

        if isinstance(
            value,
            (list, tuple),
        ):
            safe_list: list[Any] = []

            for item in value:

                if isinstance(
                    item,
                    Mapping,
                ):
                    safe_list.append(
                        sanitize_metadata(
                            item,
                            depth + 1,
                        )
                    )

                elif isinstance(
                    item,
                    (str, int, float, bool),
                ):
                    safe_list.append(item)

                elif item is None:
                    safe_list.append(None)

                else:
                    safe_list.append(str(item))

            result[safe_key] = safe_list

            continue

        if value is None:
            result[safe_key] = None
            continue

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    result[safe_key] = None
                else:
                    result[safe_key] = value
            else:
                result[safe_key] = value

            continue

        result[safe_key] = str(value)

    return result


def tokenize(
    text: str,
) -> list[str]:
    """
    Tokenisasi sederhana untuk lexical memory search.
    """
    text = normalize_text(text).lower()

    if not text:
        return []

    return re.findall(
        r"[a-zA-Z0-9_]+",
        text,
    )


def unique_tokens(
    tokens: Sequence[str],
) -> set[str]:
    """
    Set token unik.
    """
    return {
        token
        for token in tokens
        if token
    }


def token_overlap_score(
    query_tokens: set[str],
    document_tokens: set[str],
) -> float:
    """
    Hitung lexical overlap sederhana.
    """
    if not query_tokens:
        return 0.0

    if not document_tokens:
        return 0.0

    overlap = query_tokens.intersection(
        document_tokens
    )

    return len(overlap) / len(query_tokens)


def substring_score(
    query: str,
    text: str,
) -> float:
    """
    Hitung bonus jika query muncul sebagai substring.
    """
    normalized_query = normalize_text(query).lower()

    normalized_text = normalize_text(text).lower()

    if not normalized_query:
        return 0.0

    if not normalized_text:
        return 0.0

    if normalized_query in normalized_text:
        return 1.0

    return 0.0


def stable_hash(
    text: str,
) -> str:
    """
    SHA256 hash untuk deduplication.
    """
    return hashlib.sha256(
        normalize_text(text).encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================================
# MEMORY DATA MODEL
# ============================================================================


@dataclass
class MemoryRecord:
    """
    Representasi satu memory ZAI.
    """

    memory_id: str

    content: str

    namespace: str = DEFAULT_NAMESPACE

    category: str = DEFAULT_CATEGORY

    importance: float = DEFAULT_IMPORTANCE

    confidence: float = DEFAULT_CONFIDENCE

    status: str = MemoryStatus.ACTIVE

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=iso_now
    )

    updated_at: str = field(
        default_factory=iso_now
    )

    accessed_at: str | None = None

    expires_at: str | None = None

    archived_at: str | None = None

    deleted_at: str | None = None

    access_count: int = 0

    version: int = 1

    content_hash: str = ""

    source: str = "zai"

    pinned: bool = False

    def __post_init__(self) -> None:

        self.content = normalize_text(
            self.content
        )[:MAX_TEXT_LENGTH]

        self.namespace = normalize_key(
            self.namespace,
            DEFAULT_NAMESPACE,
        )

        self.category = normalize_key(
            self.category,
            DEFAULT_CATEGORY,
        )

        self.importance = clamp(
            float(self.importance),
            0.0,
            1.0,
        )

        self.confidence = clamp(
            float(self.confidence),
            0.0,
            1.0,
        )

        self.tags = normalize_tags(
            self.tags
        )

        self.metadata = sanitize_metadata(
            self.metadata
        )

        if not self.content_hash:
            self.content_hash = stable_hash(
                self.content
            )

    @property
    def is_active(self) -> bool:
        return self.status == MemoryStatus.ACTIVE

    @property
    def is_archived(self) -> bool:
        return self.status == MemoryStatus.ARCHIVED

    @property
    def is_deleted(self) -> bool:
        return self.status == MemoryStatus.DELETED

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False

        expires = parse_datetime(
            self.expires_at
        )

        if expires is None:
            return False

        current = datetime.now().astimezone()

        if expires.tzinfo is None:
            current = current.replace(
                tzinfo=None
            )

        return current >= expires

    @property
    def searchable_text(self) -> str:
        return " ".join(
            [
                self.content,
                self.namespace,
                self.category,
                " ".join(self.tags),
            ]
        )

    def touch(self) -> None:
        """
        Update access information.
        """
        self.access_count += 1

        self.accessed_at = iso_now()

        self.updated_at = iso_now()

    def archive(self) -> None:
        """
        Archive memory.
        """
        self.status = MemoryStatus.ARCHIVED

        self.archived_at = iso_now()

        self.updated_at = iso_now()

        self.version += 1

    def restore(self) -> None:
        """
        Restore archived/expired memory.
        """
        self.status = MemoryStatus.ACTIVE

        self.archived_at = None

        self.deleted_at = None

        self.updated_at = iso_now()

        self.version += 1

    def delete(self) -> None:
        """
        Soft delete.
        """
        self.status = MemoryStatus.DELETED

        self.deleted_at = iso_now()

        self.updated_at = iso_now()

        self.version += 1

    def pin(self) -> None:
        """
        Pin memory.
        """
        self.pinned = True

        self.updated_at = iso_now()

        self.version += 1

    def unpin(self) -> None:
        """
        Unpin memory.
        """
        self.pinned = False

        self.updated_at = iso_now()

        self.version += 1

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize memory.
        """
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "MemoryRecord":
        """
        Deserialize memory.
        """
        allowed = {
            field_name
            for field_name in cls.__dataclass_fields__
        }

        payload = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(
            **payload
        )


@dataclass
class MemorySearchResult:
    """
    Hasil pencarian memory.
    """

    memory: MemoryRecord

    score: float

    lexical_score: float

    substring_score: float

    importance_score: float

    confidence_score: float

    recency_score: float

    pin_bonus: float

    reasons: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": round(
                self.score,
                6,
            ),
            "lexical_score": round(
                self.lexical_score,
                6,
            ),
            "substring_score": round(
                self.substring_score,
                6,
            ),
            "importance_score": round(
                self.importance_score,
                6,
            ),
            "confidence_score": round(
                self.confidence_score,
                6,
            ),
            "recency_score": round(
                self.recency_score,
                6,
            ),
            "pin_bonus": round(
                self.pin_bonus,
                6,
            ),
            "reasons": list(self.reasons),
        }


@dataclass
class MemoryExecutionResult:
    """
    Result object untuk operasi Memory Manager.
    """

    success: bool

    operation: str

    status: str

    memory_id: str | None = None

    response: Any = None

    error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=iso_now
    )

    execution_id: str = field(
        default_factory=lambda: str(
            uuid.uuid4()
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "status": self.status,
            "memory_id": self.memory_id,
            "response": self.response,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "execution_id": self.execution_id,
        }


# ============================================================================
# MEMORY MANAGER
# ============================================================================


class MemoryManager:
    """
    Core memory engine untuk ZAI.

    Design goals:

    1. deterministic
    2. dependency-light
    3. persistent
    4. searchable
    5. thread-safe
    6. extensible
    7. suitable for ZAIBrain integration
    """

    VERSION = MEMORY_MANAGER_VERSION

    def __init__(
        self,
        storage_path: str | Path | None = None,
        auto_load: bool = True,
        autosave: bool = True,
    ) -> None:

        self.storage_path = Path(
            storage_path
            if storage_path is not None
            else Path("data")
            / DEFAULT_STORAGE_FILENAME
        )

        self.autosave = bool(
            autosave
        )

        self._memories: dict[
            str,
            MemoryRecord,
        ] = {}

        self._history: list[
            dict[str, Any]
        ] = []

        self._lock = threading.RLock()

        self.execution_count = 0

        self.success_count = 0

        self.failure_count = 0

        self.search_count = 0

        self.write_count = 0

        self.delete_count = 0

        self.archive_count = 0

        self.restore_count = 0

        self._started_at = iso_now()

        if auto_load:
            self.load()

    # ------------------------------------------------------------------
    # BASIC INFO
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """
        Informasi Memory Manager.
        """
        with self._lock:
            return {
                "memory": "MemoryManager",
                "version": self.VERSION,
                "status": "READY",
                "total_memories": len(
                    self._memories
                ),
                "active_memories": self.count(
                    status=MemoryStatus.ACTIVE
                ),
                "archived_memories": self.count(
                    status=MemoryStatus.ARCHIVED
                ),
                "deleted_memories": self.count(
                    status=MemoryStatus.DELETED
                ),
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "search_count": self.search_count,
                "write_count": self.write_count,
                "delete_count": self.delete_count,
                "archive_count": self.archive_count,
                "restore_count": self.restore_count,
                "storage_path": str(
                    self.storage_path
                ),
                "autosave": self.autosave,
                "started_at": self._started_at,
            }

    def health(self) -> dict[str, Any]:
        """
        Health check.
        """
        with self._lock:

            storage_ready = True

            storage_error = None

            try:
                self.storage_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            except Exception as exc:
                storage_ready = False
                storage_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            status = (
                "HEALTHY"
                if storage_ready
                else "DEGRADED"
            )

            return {
                "memory": "MemoryManager",
                "version": self.VERSION,
                "status": status,
                "storage_ready": storage_ready,
                "storage_error": storage_error,
                "total_memories": len(
                    self._memories
                ),
                "active_memories": self.count(
                    status=MemoryStatus.ACTIVE
                ),
                "search_count": self.search_count,
                "write_count": self.write_count,
            }

    # ------------------------------------------------------------------
    # INTERNAL EXECUTION
    # ------------------------------------------------------------------

    def _record_success(self) -> None:
        self.execution_count += 1
        self.success_count += 1

    def _record_failure(self) -> None:
        self.execution_count += 1
        self.failure_count += 1

    def _history_event(
        self,
        operation: str,
        **data: Any,
    ) -> None:
        self._history.append(
            {
                "operation": operation,
                "timestamp": iso_now(),
                "data": sanitize_metadata(
                    data
                ),
            }
        )

        if len(self._history) > 1000:
            del self._history[
                :-1000
            ]

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        category: str = DEFAULT_CATEGORY,
        importance: float = DEFAULT_IMPORTANCE,
        confidence: float = DEFAULT_CONFIDENCE,
        tags: Iterable[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        source: str = "zai",
        expires_in_seconds: float | None = None,
        pinned: bool = False,
        deduplicate: bool = True,
    ) -> MemoryRecord:

        with self._lock:

            normalized_content = normalize_text(
                content
            )

            if not normalized_content:
                raise ValueError(
                    "Memory content tidak boleh kosong."
                )

            if len(normalized_content) > MAX_TEXT_LENGTH:
                raise ValueError(
                    "Memory content terlalu panjang."
                )

            normalized_namespace = normalize_key(
                namespace,
                DEFAULT_NAMESPACE,
            )

            normalized_category = normalize_key(
                category,
                DEFAULT_CATEGORY,
            )

            content_hash = stable_hash(
                normalized_content
            )

            if deduplicate:

                existing = self._find_duplicate(
                    content_hash,
                    normalized_namespace,
                )

                if existing is not None:
                    existing.touch()

                    if importance > existing.importance:
                        existing.importance = clamp(
                            float(importance),
                            0.0,
                            1.0,
                        )

                    if confidence > existing.confidence:
                        existing.confidence = clamp(
                            float(confidence),
                            0.0,
                            1.0,
                        )

                    self._record_success()

                    self._history_event(
                        "memory_deduplicated",
                        memory_id=existing.memory_id,
                    )

                    return existing

            expires_at = None

            if expires_in_seconds is not None:

                seconds = float(
                    expires_in_seconds
                )

                if seconds <= 0:
                    raise ValueError(
                        "expires_in_seconds harus lebih besar dari 0."
                    )

                expires_at = (
                    datetime.now().astimezone()
                    + timedelta(
                        seconds=seconds
                    )
                ).isoformat()

            memory = MemoryRecord(
                memory_id=str(
                    uuid.uuid4()
                ),
                content=normalized_content,
                namespace=normalized_namespace,
                category=normalized_category,
                importance=importance,
                confidence=confidence,
                tags=normalize_tags(tags),
                metadata=sanitize_metadata(
                    metadata
                ),
                source=normalize_text(
                    source
                ) or "zai",
                expires_at=expires_at,
                pinned=bool(pinned),
                content_hash=content_hash,
            )

            self._memories[
                memory.memory_id
            ] = memory

            self.write_count += 1

            self._record_success()

            self._history_event(
                "memory_created",
                memory_id=memory.memory_id,
                namespace=memory.namespace,
                category=memory.category,
            )

            if self.autosave:
                self.save()

            return memory

    # ------------------------------------------------------------------
    # DUPLICATE DETECTION
    # ------------------------------------------------------------------

    def _find_duplicate(
        self,
        content_hash: str,
        namespace: str,
    ) -> MemoryRecord | None:

        for memory in self._memories.values():

            if memory.status == MemoryStatus.DELETED:
                continue

            if memory.namespace != namespace:
                continue

            if memory.content_hash == content_hash:
                return memory

        return None

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def get(
        self,
        memory_id: str,
        *,
        touch: bool = True,
        include_deleted: bool = False,
    ) -> MemoryRecord | None:

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return None

            if (
                memory.is_deleted
                and not include_deleted
            ):
                return None

            if memory.is_expired:

                if memory.status == MemoryStatus.ACTIVE:
                    memory.status = MemoryStatus.EXPIRED
                    memory.updated_at = iso_now()

                if not include_deleted:
                    return None

            if touch:
                memory.touch()

            return memory

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        namespace: str | None = None,
        category: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        tags: Iterable[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        source: str | None = None,
        expires_in_seconds: float | None = None,
    ) -> MemoryRecord:

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                raise KeyError(
                    f"Memory '{memory_id}' tidak ditemukan."
                )

            if memory.is_deleted:
                raise ValueError(
                    "Memory yang sudah dihapus tidak dapat diupdate."
                )

            if content is not None:

                normalized = normalize_text(
                    content
                )

                if not normalized:
                    raise ValueError(
                        "Content memory tidak boleh kosong."
                    )

                memory.content = normalized[
                    :MAX_TEXT_LENGTH
                ]

                memory.content_hash = stable_hash(
                    memory.content
                )

            if namespace is not None:
                memory.namespace = normalize_key(
                    namespace,
                    DEFAULT_NAMESPACE,
                )

            if category is not None:
                memory.category = normalize_key(
                    category,
                    DEFAULT_CATEGORY,
                )

            if importance is not None:
                memory.importance = clamp(
                    float(importance),
                    0.0,
                    1.0,
                )

            if confidence is not None:
                memory.confidence = clamp(
                    float(confidence),
                    0.0,
                    1.0,
                )

            if tags is not None:
                memory.tags = normalize_tags(
                    tags
                )

            if metadata is not None:
                memory.metadata = sanitize_metadata(
                    metadata
                )

            if source is not None:
                memory.source = (
                    normalize_text(source)
                    or "zai"
                )

            if expires_in_seconds is not None:

                seconds = float(
                    expires_in_seconds
                )

                if seconds <= 0:
                    raise ValueError(
                        "expires_in_seconds harus lebih besar dari 0."
                    )

                memory.expires_at = (
                    datetime.now().astimezone()
                    + timedelta(
                        seconds=seconds
                    )
                ).isoformat()

            memory.updated_at = iso_now()

            memory.version += 1

            self.write_count += 1

            self._record_success()

            self._history_event(
                "memory_updated",
                memory_id=memory_id,
            )

            if self.autosave:
                self.save()

            return memory

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def delete(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                self._record_failure()
                return False

            memory.delete()

            self.delete_count += 1

            self._record_success()

            self._history_event(
                "memory_deleted",
                memory_id=memory_id,
            )

            if self.autosave:
                self.save()

            return True

    # ------------------------------------------------------------------
    # ARCHIVE
    # ------------------------------------------------------------------

    def archive(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                self._record_failure()
                return False

            memory.archive()

            self.archive_count += 1

            self._record_success()

            self._history_event(
                "memory_archived",
                memory_id=memory_id,
            )

            if self.autosave:
                self.save()

            return True

    # ------------------------------------------------------------------
    # RESTORE
    # ------------------------------------------------------------------

    def restore(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                self._record_failure()
                return False

            memory.restore()

            self.restore_count += 1

            self._record_success()

            self._history_event(
                "memory_restored",
                memory_id=memory_id,
            )

            if self.autosave:
                self.save()

            return True

    # ------------------------------------------------------------------
    # PIN
    # ------------------------------------------------------------------

    def pin(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return False

            memory.pin()

            if self.autosave:
                self.save()

            return True

    def unpin(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return False

            memory.unpin()

            if self.autosave:
                self.save()

            return True

    # ------------------------------------------------------------------
    # COUNT
    # ------------------------------------------------------------------

    def count(
        self,
        *,
        namespace: str | None = None,
        category: str | None = None,
        status: str | None = None,
    ) -> int:

        namespace_normalized = (
            normalize_key(
                namespace,
                DEFAULT_NAMESPACE,
            )
            if namespace is not None
            else None
        )

        category_normalized = (
            normalize_key(
                category,
                DEFAULT_CATEGORY,
            )
            if category is not None
            else None
        )

        with self._lock:

            total = 0

            for memory in self._memories.values():

                if (
                    namespace_normalized is not None
                    and memory.namespace
                    != namespace_normalized
                ):
                    continue

                if (
                    category_normalized is not None
                    and memory.category
                    != category_normalized
                ):
                    continue

                if (
                    status is not None
                    and memory.status != status
                ):
                    continue

                total += 1

            return total

    # ------------------------------------------------------------------
    # LIST
    # ------------------------------------------------------------------

    def list_memories(
        self,
        *,
        namespace: str | None = None,
        category: str | None = None,
        status: str = MemoryStatus.ACTIVE,
        limit: int = DEFAULT_MAX_RESULTS,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:

        namespace_normalized = (
            normalize_key(
                namespace,
                DEFAULT_NAMESPACE,
            )
            if namespace is not None
            else None
        )

        category_normalized = (
            normalize_key(
                category,
                DEFAULT_CATEGORY,
            )
            if category is not None
            else None
        )

        limit = max(
            1,
            min(
                int(limit),
                1000,
            ),
        )

        with self._lock:

            result: list[
                MemoryRecord
            ] = []

            for memory in self._memories.values():

                if (
                    status is not None
                    and memory.status != status
                ):
                    continue

                if (
                    namespace_normalized is not None
                    and memory.namespace
                    != namespace_normalized
                ):
                    continue

                if (
                    category_normalized is not None
                    and memory.category
                    != category_normalized
                ):
                    continue

                if (
                    memory.is_expired
                    and not include_expired
                ):
                    continue

                result.append(memory)

            result.sort(
                key=lambda item: (
                    item.pinned,
                    item.importance,
                    item.created_at,
                ),
                reverse=True,
            )

            return result[:limit]

    # ------------------------------------------------------------------
    # RECENT
    # ------------------------------------------------------------------

    def recent(
        self,
        limit: int = DEFAULT_MAX_RESULTS,
        *,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:

        memories = self.list_memories(
            namespace=namespace,
            status=MemoryStatus.ACTIVE,
            limit=1000,
        )

        memories.sort(
            key=lambda memory: memory.created_at,
            reverse=True,
        )

        return memories[:limit]

    # ------------------------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------------------------

    def important(
        self,
        limit: int = DEFAULT_MAX_RESULTS,
        *,
        namespace: str | None = None,
        minimum_importance: float = 0.7,
    ) -> list[MemoryRecord]:

        memories = self.list_memories(
            namespace=namespace,
            status=MemoryStatus.ACTIVE,
            limit=1000,
        )

        memories = [
            memory
            for memory in memories
            if memory.importance
            >= minimum_importance
        ]

        memories.sort(
            key=lambda memory: (
                memory.pinned,
                memory.importance,
                memory.confidence,
            ),
            reverse=True,
        )

        return memories[:limit]

    # ------------------------------------------------------------------
    # TAG SEARCH
    # ------------------------------------------------------------------

    def by_tag(
        self,
        tag: str,
        *,
        limit: int = DEFAULT_MAX_RESULTS,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:

        normalized_tag = normalize_text(
            tag
        ).lower()

        if not normalized_tag:
            return []

        memories = self.list_memories(
            namespace=namespace,
            status=MemoryStatus.ACTIVE,
            limit=1000,
        )

        result = [
            memory
            for memory in memories
            if normalized_tag in memory.tags
        ]

        return result[:limit]

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        namespace: str | None = None,
        category: str | None = None,
        limit: int = DEFAULT_MAX_RESULTS,
        minimum_score: float = 0.05,
        include_archived: bool = False,
        touch_results: bool = False,
    ) -> list[MemorySearchResult]:

        started = datetime.now().astimezone()

        normalized_query = normalize_text(
            query
        )

        if not normalized_query:
            return []

        query_tokens = unique_tokens(
            tokenize(normalized_query)
        )

        namespace_normalized = (
            normalize_key(
                namespace,
                DEFAULT_NAMESPACE,
            )
            if namespace is not None
            else None
        )

        category_normalized = (
            normalize_key(
                category,
                DEFAULT_CATEGORY,
            )
            if category is not None
            else None
        )

        with self._lock:

            self.search_count += 1

            results: list[
                MemorySearchResult
            ] = []

            for memory in self._memories.values():

                if memory.is_deleted:
                    continue

                if (
                    memory.is_archived
                    and not include_archived
                ):
                    continue

                if (
                    namespace_normalized is not None
                    and memory.namespace
                    != namespace_normalized
                ):
                    continue

                if (
                    category_normalized is not None
                    and memory.category
                    != category_normalized
                ):
                    continue

                if memory.is_expired:
                    continue

                document_tokens = unique_tokens(
                    tokenize(
                        memory.searchable_text
                    )
                )

                lexical = token_overlap_score(
                    query_tokens,
                    document_tokens,
                )

                substring = substring_score(
                    normalized_query,
                    memory.searchable_text,
                )

                importance_score = memory.importance

                confidence_score = memory.confidence

                recency_score = self._recency_score(
                    memory
                )

                pin_bonus = (
                    0.10
                    if memory.pinned
                    else 0.0
                )

                score = (
                    lexical * 0.45
                    + substring * 0.25
                    + importance_score * 0.10
                    + confidence_score * 0.10
                    + recency_score * 0.10
                    + pin_bonus
                )

                if score < minimum_score:
                    continue

                reasons: list[str] = []

                if lexical > 0:
                    reasons.append(
                        "token_overlap"
                    )

                if substring > 0:
                    reasons.append(
                        "substring_match"
                    )

                if importance_score >= 0.7:
                    reasons.append(
                        "high_importance"
                    )

                if confidence_score >= 0.8:
                    reasons.append(
                        "high_confidence"
                    )

                if recency_score >= 0.7:
                    reasons.append(
                        "recent_memory"
                    )

                if memory.pinned:
                    reasons.append(
                        "pinned_memory"
                    )

                results.append(
                    MemorySearchResult(
                        memory=memory,
                        score=score,
                        lexical_score=lexical,
                        substring_score=substring,
                        importance_score=importance_score,
                        confidence_score=confidence_score,
                        recency_score=recency_score,
                        pin_bonus=pin_bonus,
                        reasons=reasons,
                    )
                )

            results.sort(
                key=lambda item: (
                    item.score,
                    item.memory.importance,
                    item.memory.confidence,
                ),
                reverse=True,
            )

            results = results[:limit]

            if touch_results:

                for result in results:
                    result.memory.touch()

            latency = (
                datetime.now().astimezone()
                - started
            ).total_seconds() * 1000

            self._history_event(
                "memory_search",
                query=normalized_query,
                result_count=len(results),
                latency_ms=round(
                    latency,
                    4,
                ),
            )

            return results

    # ------------------------------------------------------------------
    # RECENCY SCORE
    # ------------------------------------------------------------------

    def _recency_score(
        self,
        memory: MemoryRecord,
    ) -> float:

        created = parse_datetime(
            memory.created_at
        )

        if created is None:
            return 0.0

        current = datetime.now().astimezone()

        if created.tzinfo is None:
            current = current.replace(
                tzinfo=None
            )

        age_seconds = max(
            0.0,
            (
                current - created
            ).total_seconds(),
        )

        half_life = 60 * 60 * 24 * 30

        score = math.exp(
            -age_seconds / half_life
        )

        return clamp(
            score,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # CONTEXT
    # ------------------------------------------------------------------

    def context(
        self,
        query: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        limit: int = 8,
    ) -> list[MemoryRecord]:

        results = self.search(
            query,
            namespace=namespace,
            limit=limit,
            minimum_score=0.05,
            touch_results=False,
        )

        return [
            result.memory
            for result in results
        ]

    # ------------------------------------------------------------------
    # FORGET
    # ------------------------------------------------------------------

    def forget(
        self,
        query: str,
        *,
        namespace: str | None = None,
        limit: int = 10,
    ) -> int:

        results = self.search(
            query,
            namespace=namespace,
            limit=limit,
            minimum_score=0.30,
        )

        count = 0

        for result in results:

            if self.delete(
                result.memory.memory_id
            ):
                count += 1

        return count

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def cleanup_expired(
        self,
    ) -> int:

        with self._lock:

            count = 0

            for memory in self._memories.values():

                if memory.status != MemoryStatus.ACTIVE:
                    continue

                if not memory.is_expired:
                    continue

                memory.status = (
                    MemoryStatus.EXPIRED
                )

                memory.updated_at = iso_now()

                memory.version += 1

                count += 1

            if count > 0:

                self._history_event(
                    "expired_memories_cleaned",
                    count=count,
                )

                if self.autosave:
                    self.save()

            return count

    # ------------------------------------------------------------------
    # HARD PURGE
    # ------------------------------------------------------------------

    def purge_deleted(
        self,
        *,
        older_than_days: int | None = None,
    ) -> int:

        with self._lock:

            now = datetime.now().astimezone()

            targets: list[str] = []

            for memory_id, memory in self._memories.items():

                if memory.status != MemoryStatus.DELETED:
                    continue

                if older_than_days is not None:

                    deleted_at = parse_datetime(
                        memory.deleted_at
                    )

                    if deleted_at is None:
                        continue

                    if deleted_at.tzinfo is None:
                        comparison_now = now.replace(
                            tzinfo=None
                        )
                    else:
                        comparison_now = now

                    age = (
                        comparison_now
                        - deleted_at
                    ).total_seconds()

                    if age < (
                        older_than_days
                        * 86400
                    ):
                        continue

                targets.append(
                    memory_id
                )

            for memory_id in targets:
                del self._memories[
                    memory_id
                ]

            if targets:
                self._history_event(
                    "deleted_memories_purged",
                    count=len(targets),
                )

                if self.autosave:
                    self.save()

            return len(targets)

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def export_data(
        self,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:

        with self._lock:

            memories = []

            for memory in self._memories.values():

                if (
                    memory.is_deleted
                    and not include_deleted
                ):
                    continue

                memories.append(
                    memory.to_dict()
                )

            return {
                "schema_version": "1.0.0",
                "manager_version": self.VERSION,
                "exported_at": iso_now(),
                "memories": memories,
                "statistics": self.statistics(),
            }

    # ------------------------------------------------------------------
    # IMPORT
    # ------------------------------------------------------------------

    def import_data(
        self,
        data: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> int:

        if not isinstance(
            data,
            Mapping,
        ):
            raise TypeError(
                "Memory import harus berupa mapping."
            )

        raw_memories = data.get(
            "memories",
            [],
        )

        if not isinstance(
            raw_memories,
            list,
        ):
            raise ValueError(
                "Field 'memories' harus berupa list."
            )

        with self._lock:

            if replace:
                self._memories.clear()

            imported = 0

            for raw_memory in raw_memories:

                if not isinstance(
                    raw_memory,
                    Mapping,
                ):
                    continue

                try:
                    memory = MemoryRecord.from_dict(
                        raw_memory
                    )
                except Exception:
                    continue

                self._memories[
                    memory.memory_id
                ] = memory

                imported += 1

            self._history_event(
                "memory_data_imported",
                count=imported,
                replace=replace,
            )

            if self.autosave:
                self.save()

            return imported

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(self) -> MemoryExecutionResult:

        with self._lock:

            try:

                self.storage_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                payload = self.export_data(
                    include_deleted=True
                )

                temporary_path = self.storage_path.with_suffix(
                    self.storage_path.suffix
                    + ".tmp"
                )

                temporary_path.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                temporary_path.replace(
                    self.storage_path
                )

                self._record_success()

                return MemoryExecutionResult(
                    success=True,
                    operation="save",
                    status="completed",
                    response=True,
                    metadata={
                        "storage_path": str(
                            self.storage_path
                        ),
                        "memory_count": len(
                            self._memories
                        ),
                    },
                )

            except Exception as exc:

                self._record_failure()

                return MemoryExecutionResult(
                    success=False,
                    operation="save",
                    status="failed",
                    response=False,
                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load(self) -> MemoryExecutionResult:

        with self._lock:

            if not self.storage_path.exists():

                return MemoryExecutionResult(
                    success=True,
                    operation="load",
                    status="empty",
                    response=0,
                    metadata={
                        "storage_exists": False,
                        "memory_count": 0,
                    },
                )

            try:

                raw = json.loads(
                    self.storage_path.read_text(
                        encoding="utf-8"
                    )
                )

                imported = self.import_data(
                    raw,
                    replace=True,
                )

                return MemoryExecutionResult(
                    success=True,
                    operation="load",
                    status="completed",
                    response=imported,
                    metadata={
                        "storage_exists": True,
                        "memory_count": imported,
                    },
                )

            except Exception as exc:

                self._record_failure()

                return MemoryExecutionResult(
                    success=False,
                    operation="load",
                    status="failed",
                    response=0,
                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

    # ------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------

    def history(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        limit = max(
            1,
            min(
                int(limit),
                1000,
            ),
        )

        with self._lock:
            return list(
                self._history[-limit:]
            )

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:

        with self._lock:

            total = len(
                self._memories
            )

            active = self.count(
                status=MemoryStatus.ACTIVE
            )

            archived = self.count(
                status=MemoryStatus.ARCHIVED
            )

            deleted = self.count(
                status=MemoryStatus.DELETED
            )

            expired = self.count(
                status=MemoryStatus.EXPIRED
            )

            average_importance = 0.0

            average_confidence = 0.0

            if total > 0:

                average_importance = (
                    sum(
                        memory.importance
                        for memory in self._memories.values()
                    )
                    / total
                )

                average_confidence = (
                    sum(
                        memory.confidence
                        for memory in self._memories.values()
                    )
                    / total
                )

            namespaces: dict[str, int] = {}

            categories: dict[str, int] = {}

            tags: dict[str, int] = {}

            for memory in self._memories.values():

                namespaces[
                    memory.namespace
                ] = (
                    namespaces.get(
                        memory.namespace,
                        0,
                    )
                    + 1
                )

                categories[
                    memory.category
                ] = (
                    categories.get(
                        memory.category,
                        0,
                    )
                    + 1
                )

                for tag in memory.tags:
                    tags[tag] = (
                        tags.get(
                            tag,
                            0,
                        )
                        + 1
                    )

            success_rate = 0.0

            failure_rate = 0.0

            if self.execution_count > 0:

                success_rate = (
                    self.success_count
                    / self.execution_count
                    * 100
                )

                failure_rate = (
                    self.failure_count
                    / self.execution_count
                    * 100
                )

            return {
                "manager": "MemoryManager",
                "version": self.VERSION,
                "total_memories": total,
                "active_memories": active,
                "archived_memories": archived,
                "deleted_memories": deleted,
                "expired_memories": expired,
                "pinned_memories": sum(
                    1
                    for memory
                    in self._memories.values()
                    if memory.pinned
                ),
                "average_importance": round(
                    average_importance,
                    4,
                ),
                "average_confidence": round(
                    average_confidence,
                    4,
                ),
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": round(
                    success_rate,
                    4,
                ),
                "failure_rate": round(
                    failure_rate,
                    4,
                ),
                "search_count": self.search_count,
                "write_count": self.write_count,
                "delete_count": self.delete_count,
                "archive_count": self.archive_count,
                "restore_count": self.restore_count,
                "history_size": len(
                    self._history
                ),
                "namespace_distribution": namespaces,
                "category_distribution": categories,
                "tag_distribution": tags,
            }

    # ------------------------------------------------------------------
    # SNAPSHOT
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:

        with self._lock:

            return {
                "info": self.info(),
                "health": self.health(),
                "statistics": self.statistics(),
                "recent": [
                    memory.to_dict()
                    for memory in self.recent(
                        limit=10
                    )
                ],
                "important": [
                    memory.to_dict()
                    for memory in self.important(
                        limit=10
                    )
                ],
            }

    # ------------------------------------------------------------------
    # CLEAR
    # ------------------------------------------------------------------

    def clear(
        self,
        *,
        namespace: str | None = None,
        category: str | None = None,
        include_pinned: bool = False,
    ) -> int:

        with self._lock:

            namespace_normalized = (
                normalize_key(
                    namespace,
                    DEFAULT_NAMESPACE,
                )
                if namespace is not None
                else None
            )

            category_normalized = (
                normalize_key(
                    category,
                    DEFAULT_CATEGORY,
                )
                if category is not None
                else None
            )

            targets: list[str] = []

            for memory_id, memory in self._memories.items():

                if memory.is_deleted:
                    continue

                if (
                    not include_pinned
                    and memory.pinned
                ):
                    continue

                if (
                    namespace_normalized is not None
                    and memory.namespace
                    != namespace_normalized
                ):
                    continue

                if (
                    category_normalized is not None
                    and memory.category
                    != category_normalized
                ):
                    continue

                targets.append(
                    memory_id
                )

            for memory_id in targets:

                memory = self._memories[
                    memory_id
                ]

                memory.delete()

            if targets:

                self.delete_count += len(
                    targets
                )

                self._history_event(
                    "memory_namespace_cleared",
                    count=len(targets),
                    namespace=namespace_normalized,
                    category=category_normalized,
                )

                if self.autosave:
                    self.save()

            return len(targets)

    # ------------------------------------------------------------------
    # MEMORY IDs
    # ------------------------------------------------------------------

    def ids(
        self,
        *,
        status: str | None = MemoryStatus.ACTIVE,
    ) -> list[str]:

        with self._lock:

            if status is None:
                return list(
                    self._memories.keys()
                )

            return [
                memory_id
                for memory_id, memory
                in self._memories.items()
                if memory.status == status
            ]

    # ------------------------------------------------------------------
    # EXISTS
    # ------------------------------------------------------------------

    def exists(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:
            return memory_id in self._memories

    # ------------------------------------------------------------------
    # DUPLICATE CHECK
    # ------------------------------------------------------------------

    def has_content(
        self,
        content: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> bool:

        normalized_content = normalize_text(
            content
        )

        content_hash = stable_hash(
            normalized_content
        )

        namespace_normalized = normalize_key(
            namespace,
            DEFAULT_NAMESPACE,
        )

        with self._lock:

            return (
                self._find_duplicate(
                    content_hash,
                    namespace_normalized,
                )
                is not None
            )

    # ------------------------------------------------------------------
    # MEMORY BY CATEGORY
    # ------------------------------------------------------------------

    def by_category(
        self,
        category: str,
        *,
        limit: int = DEFAULT_MAX_RESULTS,
    ) -> list[MemoryRecord]:

        return self.list_memories(
            category=category,
            status=MemoryStatus.ACTIVE,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # MEMORY BY NAMESPACE
    # ------------------------------------------------------------------

    def by_namespace(
        self,
        namespace: str,
        *,
        limit: int = DEFAULT_MAX_RESULTS,
    ) -> list[MemoryRecord]:

        return self.list_memories(
            namespace=namespace,
            status=MemoryStatus.ACTIVE,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # MEMORY SUMMARY
    # ------------------------------------------------------------------

    def summarize(
        self,
        *,
        namespace: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:

        memories = self.list_memories(
            namespace=namespace,
            status=MemoryStatus.ACTIVE,
            limit=limit,
        )

        return {
            "namespace": namespace
            or DEFAULT_NAMESPACE,
            "memory_count": len(
                memories
            ),
            "memories": [
                {
                    "memory_id": memory.memory_id,
                    "category": memory.category,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "content": memory.content,
                    "tags": memory.tags,
                }
                for memory in memories
            ],
        }


# ============================================================================
# MEMORY SERVICE
# ============================================================================


class MemoryService:
    """
    Facade sederhana untuk integrasi ZAIBrain.

    MemoryService menjaga agar Brain tidak harus mengetahui detail
    internal MemoryManager.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        manager: MemoryManager | None = None,
    ) -> None:

        self.manager = (
            manager
            if manager is not None
            else MemoryManager()
        )

    def info(self) -> dict[str, Any]:

        return {
            "service": "MemoryService",
            "version": self.VERSION,
            "status": "READY",
            "manager": self.manager.info(),
        }

    def health(self) -> dict[str, Any]:

        health = self.manager.health()

        return {
            "service": "MemoryService",
            "version": self.VERSION,
            "status": health["status"],
            "manager": health,
        }

    def remember(
        self,
        content: str,
        **kwargs: Any,
    ) -> MemoryRecord:

        return self.manager.remember(
            content,
            **kwargs,
        )

    def recall(
        self,
        query: str,
        *,
        limit: int = 8,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> list[MemoryRecord]:

        return self.manager.context(
            query,
            namespace=namespace,
            limit=limit,
        )

    def forget(
        self,
        query: str,
        *,
        namespace: str | None = None,
    ) -> int:

        return self.manager.forget(
            query,
            namespace=namespace,
        )

    def statistics(self) -> dict[str, Any]:

        return self.manager.statistics()


# ============================================================================
# SELF TEST
# ============================================================================


def run_self_test() -> dict[str, Any]:
    """
    Test internal Memory Engine tanpa membutuhkan external package.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as directory:

        storage = (
            Path(directory)
            / "memory_test.json"
        )

        manager = MemoryManager(
            storage_path=storage,
            auto_load=False,
            autosave=True,
        )

        memory = manager.remember(
            "ZAI sedang membangun sistem AI multi agent.",
            namespace="zai",
            category=MemoryCategory.KNOWLEDGE,
            importance=0.9,
            confidence=0.95,
            tags=[
                "zai",
                "ai",
                "multi-agent",
            ],
            metadata={
                "source": "self_test",
                "phase": 1,
            },
            pinned=True,
        )

        assert memory.memory_id

        assert manager.exists(
            memory.memory_id
        )

        fetched = manager.get(
            memory.memory_id
        )

        assert fetched is not None

        assert fetched.content.startswith(
            "ZAI sedang"
        )

        results = manager.search(
            "sistem AI multi agent",
            namespace="zai",
        )

        assert len(results) >= 1

        assert (
            results[0].memory.memory_id
            == memory.memory_id
        )

        assert manager.has_content(
            "ZAI sedang membangun sistem AI multi agent.",
            namespace="zai",
        )

        pinned = manager.important(
            namespace="zai"
        )

        assert len(pinned) >= 1

        by_tag = manager.by_tag(
            "multi-agent",
            namespace="zai",
        )

        assert len(by_tag) >= 1

        manager.update(
            memory.memory_id,
            importance=1.0,
        )

        assert (
            manager.get(
                memory.memory_id
            ).importance
            == 1.0
        )

        manager.archive(
            memory.memory_id
        )

        archived = manager.get(
            memory.memory_id,
            include_deleted=True,
        )

        assert archived is not None

        manager.restore(
            memory.memory_id
        )

        restored = manager.get(
            memory.memory_id
        )

        assert restored is not None

        manager.pin(
            memory.memory_id
        )

        assert manager.get(
            memory.memory_id
        ).pinned is True

        manager.unpin(
            memory.memory_id
        )

        assert manager.get(
            memory.memory_id
        ).pinned is False

        save_result = manager.save()

        assert save_result.success is True

        manager_two = MemoryManager(
            storage_path=storage,
            auto_load=True,
            autosave=False,
        )

        loaded = manager_two.get(
            memory.memory_id
        )

        assert loaded is not None

        stats = manager_two.statistics()

        assert stats[
            "total_memories"
        ] >= 1

        health = manager_two.health()

        assert health[
            "status"
        ] in {
            "HEALTHY",
            "DEGRADED",
        }

        snapshot = manager_two.snapshot()

        assert "info" in snapshot

        assert "statistics" in snapshot

        return {
            "success": True,
            "status": "PASS",
            "memory_id": memory.memory_id,
            "search_results": len(
                results
            ),
            "statistics": stats,
            "health": health,
        }


if __name__ == "__main__":

    result = run_self_test()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "MEMORY_MANAGER_SELF_TEST_OK"
    )
from __future__ import annotations

import json
import re
import threading
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class MemoryErrorBase(Exception):
    """Base exception untuk sistem memory ZAI."""


class MemoryValidationError(MemoryErrorBase):
    """Raised ketika data memory tidak valid."""


class MemoryNotFoundError(MemoryErrorBase):
    """Raised ketika memory tidak ditemukan."""


class MemoryStorageError(MemoryErrorBase):
    """Raised ketika terjadi masalah storage."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9_:\-.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text).lower()

    if not normalized:
        return []

    return re.findall(
        r"[a-zA-Z0-9_À-ÿ]+",
        normalized,
        flags=re.UNICODE,
    )


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class MemoryItem:
    memory_id: str
    namespace: str
    key: str
    value: Any

    memory_type: str = "fact"
    source: str = "zai"
    importance: float = 0.5
    confidence: float = 0.5

    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: str = field(default_factory=utc_iso)
    updated_at: str = field(default_factory=utc_iso)
    accessed_at: str = field(default_factory=utc_iso)

    access_count: int = 0
    version: int = 1
    active: bool = True

    expires_at: Optional[str] = None

    def __post_init__(self) -> None:
        self.memory_id = normalize_text(self.memory_id)
        self.namespace = normalize_key(self.namespace) or "default"
        self.key = normalize_key(self.key)

        if not self.key:
            raise MemoryValidationError(
                "Memory key tidak boleh kosong."
            )

        self.memory_type = (
            normalize_key(self.memory_type)
            or "fact"
        )

        self.source = (
            normalize_text(self.source)
            or "zai"
        )

        self.importance = max(
            0.0,
            min(1.0, safe_float(self.importance, 0.5)),
        )

        self.confidence = max(
            0.0,
            min(1.0, safe_float(self.confidence, 0.5)),
        )

        self.tags = sorted(
            {
                normalize_key(tag)
                for tag in self.tags
                if normalize_key(tag)
            }
        )

        if not isinstance(self.metadata, dict):
            self.metadata = {}

        self.access_count = max(
            0,
            safe_int(self.access_count),
        )

        self.version = max(
            1,
            safe_int(self.version, 1),
        )

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False

        try:
            expiration = datetime.fromisoformat(
                self.expires_at
            )

            return utc_now() >= expiration
        except ValueError:
            return False

    @property
    def searchable_text(self) -> str:
        parts = [
            self.namespace,
            self.key,
            self.memory_type,
            self.source,
            str(self.value),
            " ".join(self.tags),
        ]

        return " ".join(
            normalize_text(part)
            for part in parts
            if normalize_text(part)
        )

    def touch(self) -> None:
        self.access_count += 1
        self.accessed_at = utc_iso()

    def update(
        self,
        *,
        value: Any = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        source: Optional[str] = None,
        memory_type: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> None:
        if value is not None:
            self.value = value

        if importance is not None:
            self.importance = max(
                0.0,
                min(1.0, safe_float(importance)),
            )

        if confidence is not None:
            self.confidence = max(
                0.0,
                min(1.0, safe_float(confidence)),
            )

        if tags is not None:
            self.tags = sorted(
                {
                    normalize_key(tag)
                    for tag in tags
                    if normalize_key(tag)
                }
            )

        if metadata is not None:
            self.metadata = dict(metadata)

        if source is not None:
            self.source = normalize_text(source)

        if memory_type is not None:
            self.memory_type = (
                normalize_key(memory_type)
                or self.memory_type
            )

        if expires_at is not None:
            self.expires_at = expires_at

        self.version += 1
        self.updated_at = utc_iso()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "MemoryItem":
        if not isinstance(data, Mapping):
            raise MemoryValidationError(
                "MemoryItem harus dibuat dari mapping."
            )

        return cls(
            memory_id=normalize_text(
                data.get("memory_id")
                or str(uuid.uuid4())
            ),
            namespace=normalize_text(
                data.get("namespace")
                or "default"
            ),
            key=normalize_text(
                data.get("key")
            ),
            value=data.get("value"),
            memory_type=normalize_text(
                data.get("memory_type")
                or "fact"
            ),
            source=normalize_text(
                data.get("source")
                or "zai"
            ),
            importance=safe_float(
                data.get("importance"),
                0.5,
            ),
            confidence=safe_float(
                data.get("confidence"),
                0.5,
            ),
            tags=list(
                data.get("tags")
                or []
            ),
            metadata=dict(
                data.get("metadata")
                or {}
            ),
            created_at=normalize_text(
                data.get("created_at")
                or utc_iso()
            ),
            updated_at=normalize_text(
                data.get("updated_at")
                or utc_iso()
            ),
            accessed_at=normalize_text(
                data.get("accessed_at")
                or utc_iso()
            ),
            access_count=safe_int(
                data.get("access_count"),
                0,
            ),
            version=safe_int(
                data.get("version"),
                1,
            ),
            active=bool(
                data.get("active", True)
            ),
            expires_at=(
                normalize_text(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
        )


@dataclass(slots=True)
class MemorySearchResult:
    memory: MemoryItem
    score: float
    matched_tokens: list[str] = field(
        default_factory=list
    )
    reasons: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": round(self.score, 6),
            "matched_tokens": list(
                self.matched_tokens
            ),
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class ConversationTurn:
    turn_id: str
    session_id: str
    role: str
    content: str

    timestamp: str = field(
        default_factory=utc_iso
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    importance: float = 0.5

    def __post_init__(self) -> None:
        self.turn_id = normalize_text(
            self.turn_id
        ) or str(uuid.uuid4())

        self.session_id = normalize_text(
            self.session_id
        ) or "default"

        self.role = (
            normalize_key(self.role)
            or "user"
        )

        self.content = normalize_text(
            self.content
        )

        self.importance = max(
            0.0,
            min(1.0, safe_float(
                self.importance,
                0.5,
            )),
        )

        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ConversationTurn":
        return cls(
            turn_id=normalize_text(
                data.get("turn_id")
                or str(uuid.uuid4())
            ),
            session_id=normalize_text(
                data.get("session_id")
                or "default"
            ),
            role=normalize_text(
                data.get("role")
                or "user"
            ),
            content=normalize_text(
                data.get("content")
            ),
            timestamp=normalize_text(
                data.get("timestamp")
                or utc_iso()
            ),
            metadata=dict(
                data.get("metadata")
                or {}
            ),
            importance=safe_float(
                data.get("importance"),
                0.5,
            ),
        )


@dataclass(slots=True)
class MemoryStatistics:
    total_memories: int
    active_memories: int
    expired_memories: int

    total_conversation_turns: int
    session_count: int

    namespace_count: int
    tag_count: int

    access_count: int

    average_importance: float
    average_confidence: float

    memory_type_distribution: dict[str, int]
    namespace_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryManager:
    """
    ZAI Memory Manager.

    Layer ini bertugas menangani:

    - long-term memory
    - working memory
    - conversation history
    - memory search
    - memory ranking
    - memory update
    - memory deletion
    - namespaces
    - tags
    - importance
    - confidence
    - expiration
    - persistence
    - backup/import/export
    - statistics
    - context construction

    MemoryManager sengaja dibuat independen dari LLM agar
    dapat digunakan oleh ZAIBrain, AgentRuntime,
    Orchestrator, maupun ToolExecutionLoop.
    """

    VERSION = "1.0.0"

    DEFAULT_NAMESPACE = "default"
    WORKING_NAMESPACE = "working"
    CONVERSATION_NAMESPACE = "conversation"

    MAX_MEMORY_ITEMS = 10000
    MAX_CONVERSATION_TURNS = 5000

    def __init__(
        self,
        *,
        storage_path: str | Path | None = None,
        auto_save: bool = False,
        max_memories: int = MAX_MEMORY_ITEMS,
        max_conversation_turns: int = (
            MAX_CONVERSATION_TURNS
        ),
    ) -> None:
        self._lock = threading.RLock()

        self.storage_path = (
            Path(storage_path)
            if storage_path
            else None
        )

        self.auto_save = bool(auto_save)

        self.max_memories = max(
            100,
            safe_int(
                max_memories,
                self.MAX_MEMORY_ITEMS,
            ),
        )

        self.max_conversation_turns = max(
            100,
            safe_int(
                max_conversation_turns,
                self.MAX_CONVERSATION_TURNS,
            ),
        )

        self._memories: dict[str, MemoryItem] = {}
        self._memory_index: dict[
            str,
            set[str],
        ] = {}

        self._conversations: dict[
            str,
            list[ConversationTurn],
        ] = {}

        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0

        self._memory_created_count = 0
        self._memory_updated_count = 0
        self._memory_deleted_count = 0
        self._memory_access_count = 0

        self._search_count = 0
        self._conversation_count = 0

        if self.storage_path:
            self.load()

    # ---------------------------------------------------------
    # BASIC INFO
    # ---------------------------------------------------------

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "memory": "MemoryManager",
                "version": self.VERSION,
                "status": "READY",
                "memory_count": len(
                    self._memories
                ),
                "active_memory_count": sum(
                    1
                    for memory in self._memories.values()
                    if memory.active
                ),
                "conversation_sessions": len(
                    self._conversations
                ),
                "execution_count": (
                    self._execution_count
                ),
                "success_count": (
                    self._success_count
                ),
                "failure_count": (
                    self._failure_count
                ),
                "search_count": self._search_count,
                "auto_save": self.auto_save,
                "storage_path": (
                    str(self.storage_path)
                    if self.storage_path
                    else None
                ),
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            storage_status = "DISABLED"

            if self.storage_path:
                try:
                    parent = (
                        self.storage_path.parent
                    )

                    parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    storage_status = "READY"
                except Exception:
                    storage_status = "ERROR"

            status = (
                "HEALTHY"
                if storage_status != "ERROR"
                else "DEGRADED"
            )

            return {
                "memory": "MemoryManager",
                "version": self.VERSION,
                "status": status,
                "storage_status": storage_status,
                "memory_count": len(
                    self._memories
                ),
                "session_count": len(
                    self._conversations
                ),
                "execution_count": (
                    self._execution_count
                ),
                "success_count": (
                    self._success_count
                ),
                "failure_count": (
                    self._failure_count
                ),
                "success_rate": self.success_rate,
            }

    @property
    def success_rate(self) -> float:
        if self._execution_count <= 0:
            return 0.0

        return round(
            (
                self._success_count
                / self._execution_count
            )
            * 100.0,
            2,
        )

    # ---------------------------------------------------------
    # MEMORY CREATION
    # ---------------------------------------------------------

    def remember(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        memory_type: str = "fact",
        source: str = "zai",
        importance: float = 0.5,
        confidence: float = 0.5,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expires_at: Optional[str] = None,
        memory_id: Optional[str] = None,
        overwrite: bool = True,
    ) -> MemoryItem:
        with self._lock:
            self._execution_count += 1

            try:
                normalized_namespace = (
                    normalize_key(namespace)
                    or self.DEFAULT_NAMESPACE
                )

                normalized_key = normalize_key(
                    key
                )

                if not normalized_key:
                    raise MemoryValidationError(
                        "Memory key tidak boleh kosong."
                    )

                existing = self.get(
                    normalized_key,
                    namespace=normalized_namespace,
                    touch=False,
                )

                if existing and overwrite:
                    existing.update(
                        value=value,
                        importance=importance,
                        confidence=confidence,
                        tags=tags,
                        metadata=metadata,
                        source=source,
                        memory_type=memory_type,
                        expires_at=expires_at,
                    )

                    self._memory_updated_count += 1

                    self._success_count += 1

                    self._rebuild_index_for(
                        existing
                    )

                    self._auto_save()

                    return existing

                if existing and not overwrite:
                    raise MemoryValidationError(
                        (
                            "Memory dengan key "
                            f"'{normalized_key}' "
                            "sudah ada."
                        )
                    )

                if (
                    len(self._memories)
                    >= self.max_memories
                ):
                    self._evict_if_needed()

                item = MemoryItem(
                    memory_id=(
                        normalize_text(memory_id)
                        or str(uuid.uuid4())
                    ),
                    namespace=normalized_namespace,
                    key=normalized_key,
                    value=value,
                    memory_type=memory_type,
                    source=source,
                    importance=importance,
                    confidence=confidence,
                    tags=list(tags or []),
                    metadata=dict(
                        metadata or {}
                    ),
                    expires_at=expires_at,
                )

                self._memories[
                    item.memory_id
                ] = item

                self._index_memory(item)

                self._memory_created_count += 1
                self._success_count += 1

                self._auto_save()

                return item

            except Exception:
                self._failure_count += 1
                raise

    def remember_many(
        self,
        memories: Iterable[
            Mapping[str, Any]
        ],
        *,
        overwrite: bool = True,
    ) -> list[MemoryItem]:
        results: list[MemoryItem] = []

        for data in memories:
            if not isinstance(data, Mapping):
                raise MemoryValidationError(
                    "Setiap memory harus berupa mapping."
                )

            result = self.remember(
                key=str(
                    data.get("key", "")
                ),
                value=data.get("value"),
                namespace=str(
                    data.get(
                        "namespace",
                        self.DEFAULT_NAMESPACE,
                    )
                ),
                memory_type=str(
                    data.get(
                        "memory_type",
                        "fact",
                    )
                ),
                source=str(
                    data.get(
                        "source",
                        "zai",
                    )
                ),
                importance=safe_float(
                    data.get("importance"),
                    0.5,
                ),
                confidence=safe_float(
                    data.get("confidence"),
                    0.5,
                ),
                tags=data.get("tags") or [],
                metadata=data.get(
                    "metadata"
                )
                or {},
                expires_at=data.get(
                    "expires_at"
                ),
                memory_id=data.get(
                    "memory_id"
                ),
                overwrite=overwrite,
            )

            results.append(result)

        return results

    # ---------------------------------------------------------
    # MEMORY RETRIEVAL
    # ---------------------------------------------------------

    def get(
        self,
        key: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        touch: bool = True,
    ) -> Optional[MemoryItem]:
        with self._lock:
            normalized_key = normalize_key(key)
            normalized_namespace = (
                normalize_key(namespace)
                or self.DEFAULT_NAMESPACE
            )

            for memory in self._memories.values():
                if not memory.active:
                    continue

                if memory.namespace != (
                    normalized_namespace
                ):
                    continue

                if memory.key != normalized_key:
                    continue

                if memory.is_expired:
                    memory.active = False
                    continue

                if touch:
                    memory.touch()
                    self._memory_access_count += 1

                return memory

            return None

    def get_by_id(
        self,
        memory_id: str,
        *,
        touch: bool = True,
    ) -> Optional[MemoryItem]:
        with self._lock:
            item = self._memories.get(
                normalize_text(memory_id)
            )

            if item is None:
                return None

            if not item.active:
                return None

            if item.is_expired:
                item.active = False
                return None

            if touch:
                item.touch()
                self._memory_access_count += 1

            return item

    def require(
        self,
        key: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> MemoryItem:
        result = self.get(
            key,
            namespace=namespace,
        )

        if result is None:
            raise MemoryNotFoundError(
                (
                    "Memory tidak ditemukan: "
                    f"{namespace}:{key}"
                )
            )

        return result

    def all(
        self,
        *,
        namespace: Optional[str] = None,
        active_only: bool = True,
    ) -> list[MemoryItem]:
        with self._lock:
            normalized_namespace = (
                normalize_key(namespace)
                if namespace is not None
                else None
            )

            results = []

            for memory in self._memories.values():
                if active_only and not memory.active:
                    continue

                if memory.is_expired:
                    memory.active = False

                    if active_only:
                        continue

                if (
                    normalized_namespace is not None
                    and memory.namespace
                    != normalized_namespace
                ):
                    continue

                results.append(memory)

            results.sort(
                key=lambda item: (
                    -item.importance,
                    -item.confidence,
                    -item.access_count,
                    item.updated_at,
                )
            )

            return results

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[MemorySearchResult]:
        with self._lock:
            self._search_count += 1

            normalized_query = normalize_text(
                query
            ).lower()

            if not normalized_query:
                return []

            query_tokens = set(
                tokenize(normalized_query)
            )

            normalized_namespace = (
                normalize_key(namespace)
                if namespace
                else None
            )

            normalized_type = (
                normalize_key(memory_type)
                if memory_type
                else None
            )

            requested_tags = {
                normalize_key(tag)
                for tag in (tags or [])
                if normalize_key(tag)
            }

            results: list[
                MemorySearchResult
            ] = []

            for memory in self._memories.values():
                if not memory.active:
                    continue

                if memory.is_expired:
                    memory.active = False
                    continue

                if (
                    normalized_namespace
                    and memory.namespace
                    != normalized_namespace
                ):
                    continue

                if (
                    normalized_type
                    and memory.memory_type
                    != normalized_type
                ):
                    continue

                if requested_tags and not (
                    requested_tags.intersection(
                        memory.tags
                    )
                ):
                    continue

                score, matched, reasons = (
                    self._score_memory(
                        memory,
                        normalized_query,
                        query_tokens,
                    )
                )

                if score < min_score:
                    continue

                if score <= 0:
                    continue

                results.append(
                    MemorySearchResult(
                        memory=memory,
                        score=score,
                        matched_tokens=matched,
                        reasons=reasons,
                    )
                )

            results.sort(
                key=lambda result: (
                    -result.score,
                    -result.memory.importance,
                    -result.memory.confidence,
                    -result.memory.access_count,
                )
            )

            safe_limit = max(
                1,
                min(
                    safe_int(limit, 10),
                    100,
                ),
            )

            selected = results[
                :safe_limit
            ]

            for result in selected:
                result.memory.touch()
                self._memory_access_count += 1

            return selected

    def _score_memory(
        self,
        memory: MemoryItem,
        query: str,
        query_tokens: set[str],
    ) -> tuple[
        float,
        list[str],
        list[str],
    ]:
        searchable = memory.searchable_text.lower()

        if not searchable:
            return 0.0, [], []

        memory_tokens = set(
            tokenize(searchable)
        )

        matched = sorted(
            query_tokens.intersection(
                memory_tokens
            )
        )

        if not matched:
            return 0.0, [], []

        reasons: list[str] = []

        token_score = (
            len(matched)
            / max(
                1,
                len(query_tokens),
            )
        )

        score = token_score * 0.50

        if query in searchable:
            score += 0.25
            reasons.append(
                "exact_phrase_match"
            )

        if memory.key in query:
            score += 0.10
            reasons.append(
                "key_match"
            )

        if set(memory.tags).intersection(
            query_tokens
        ):
            score += 0.10
            reasons.append(
                "tag_match"
            )

        score += (
            memory.importance * 0.03
        )

        score += (
            memory.confidence * 0.02
        )

        if memory.access_count > 0:
            score += min(
                0.05,
                memory.access_count * 0.005,
            )

            reasons.append(
                "access_history_bonus"
            )

        reasons.append(
            "token_match"
        )

        return (
            min(1.0, score),
            matched,
            reasons,
        )

    # ---------------------------------------------------------
    # MEMORY UPDATE
    # ---------------------------------------------------------

    def update(
        self,
        memory_id: str,
        **changes: Any,
    ) -> MemoryItem:
        with self._lock:
            item = self.get_by_id(
                memory_id,
                touch=False,
            )

            if item is None:
                raise MemoryNotFoundError(
                    (
                        "Memory ID tidak ditemukan: "
                        f"{memory_id}"
                    )
                )

            item.update(
                value=changes.get("value"),
                importance=changes.get(
                    "importance"
                ),
                confidence=changes.get(
                    "confidence"
                ),
                tags=changes.get("tags"),
                metadata=changes.get(
                    "metadata"
                ),
                source=changes.get(
                    "source"
                ),
                memory_type=changes.get(
                    "memory_type"
                ),
                expires_at=changes.get(
                    "expires_at"
                ),
            )

            self._memory_updated_count += 1

            self._rebuild_index_for(item)

            self._auto_save()

            return item

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    def forget(
        self,
        key: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        hard_delete: bool = False,
    ) -> bool:
        with self._lock:
            item = self.get(
                key,
                namespace=namespace,
                touch=False,
            )

            if item is None:
                return False

            return self.forget_by_id(
                item.memory_id,
                hard_delete=hard_delete,
            )

    def forget_by_id(
        self,
        memory_id: str,
        *,
        hard_delete: bool = False,
    ) -> bool:
        with self._lock:
            normalized_id = normalize_text(
                memory_id
            )

            item = self._memories.get(
                normalized_id
            )

            if item is None:
                return False

            if hard_delete:
                del self._memories[
                    normalized_id
                ]

                self._remove_from_index(item)

            else:
                item.active = False
                item.updated_at = utc_iso()

            self._memory_deleted_count += 1

            self._auto_save()

            return True

    def clear(
        self,
        *,
        namespace: Optional[str] = None,
        hard_delete: bool = False,
    ) -> int:
        with self._lock:
            items = self.all(
                namespace=namespace,
                active_only=False,
            )

            count = 0

            for item in items:
                if hard_delete:
                    if (
                        item.memory_id
                        in self._memories
                    ):
                        del self._memories[
                            item.memory_id
                        ]

                    self._remove_from_index(
                        item
                    )
                else:
                    if item.active:
                        item.active = False
                        item.updated_at = utc_iso()

                count += 1

            self._memory_deleted_count += count

            self._auto_save()

            return count

    # ---------------------------------------------------------
    # CONVERSATION MEMORY
    # ---------------------------------------------------------

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
        importance: float = 0.5,
    ) -> ConversationTurn:
        with self._lock:
            normalized_session = (
                normalize_text(session_id)
                or "default"
            )

            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                session_id=normalized_session,
                role=role,
                content=content,
                metadata=dict(
                    metadata or {}
                ),
                importance=importance,
            )

            turns = self._conversations.setdefault(
                normalized_session,
                [],
            )

            turns.append(turn)

            if len(turns) > (
                self.max_conversation_turns
            ):
                overflow = len(turns) - (
                    self.max_conversation_turns
                )

                del turns[:overflow]

            self._conversation_count += 1

            self._auto_save()

            return turn

    def get_conversation(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> list[ConversationTurn]:
        with self._lock:
            normalized_session = (
                normalize_text(session_id)
                or "default"
            )

            turns = list(
                self._conversations.get(
                    normalized_session,
                    [],
                )
            )

            if limit is not None:
                safe_limit = max(
                    1,
                    safe_int(limit, 20),
                )

                turns = turns[
                    -safe_limit:
                ]

            return turns

    def clear_conversation(
        self,
        session_id: str,
    ) -> int:
        with self._lock:
            normalized_session = (
                normalize_text(session_id)
                or "default"
            )

            turns = self._conversations.pop(
                normalized_session,
                [],
            )

            self._auto_save()

            return len(turns)

    def sessions(self) -> list[str]:
        with self._lock:
            return sorted(
                self._conversations.keys()
            )

    # ---------------------------------------------------------
    # CONTEXT BUILDING
    # ---------------------------------------------------------

    def build_context(
        self,
        query: str,
        *,
        session_id: str = "default",
        memory_limit: int = 8,
        conversation_limit: int = 10,
    ) -> dict[str, Any]:
        with self._lock:
            memory_results = self.search(
                query,
                limit=memory_limit,
            )

            conversation = self.get_conversation(
                session_id,
                limit=conversation_limit,
            )

            memories = [
                result.to_dict()
                for result in memory_results
            ]

            turns = [
                turn.to_dict()
                for turn in conversation
            ]

            return {
                "query": normalize_text(query),
                "session_id": normalize_text(
                    session_id
                )
                or "default",
                "memories": memories,
                "conversation": turns,
                "memory_count": len(memories),
                "conversation_count": len(turns),
                "generated_at": utc_iso(),
            }

    def build_prompt_context(
        self,
        query: str,
        *,
        session_id: str = "default",
        memory_limit: int = 8,
        conversation_limit: int = 10,
    ) -> str:
        context = self.build_context(
            query,
            session_id=session_id,
            memory_limit=memory_limit,
            conversation_limit=conversation_limit,
        )

        lines: list[str] = []

        lines.append(
            "=== ZAI MEMORY CONTEXT ==="
        )

        if context["memories"]:
            lines.append(
                "\nRelevant Memories:"
            )

            for index, result in enumerate(
                context["memories"],
                start=1,
            ):
                memory = result["memory"]

                lines.append(
                    (
                        f"{index}. "
                        f"[{memory['namespace']}] "
                        f"{memory['key']}: "
                        f"{memory['value']}"
                    )
                )

        if context["conversation"]:
            lines.append(
                "\nRecent Conversation:"
            )

            for turn in context[
                "conversation"
            ]:
                lines.append(
                    (
                        f"{turn['role']}: "
                        f"{turn['content']}"
                    )
                )

        lines.append(
            "\n=== END MEMORY CONTEXT ==="
        )

        return "\n".join(lines)

    # ---------------------------------------------------------
    # INDEX MANAGEMENT
    # ---------------------------------------------------------

    def _index_memory(
        self,
        memory: MemoryItem,
    ) -> None:
        for token in set(
            tokenize(memory.searchable_text)
        ):
            bucket = self._memory_index.setdefault(
                token,
                set(),
            )

            bucket.add(
                memory.memory_id
            )

    def _remove_from_index(
        self,
        memory: MemoryItem,
    ) -> None:
        for token in set(
            tokenize(memory.searchable_text)
        ):
            bucket = self._memory_index.get(
                token
            )

            if not bucket:
                continue

            bucket.discard(
                memory.memory_id
            )

            if not bucket:
                self._memory_index.pop(
                    token,
                    None,
                )

    def _rebuild_index_for(
        self,
        memory: MemoryItem,
    ) -> None:
        self._remove_from_index(
            memory
        )

        self._index_memory(
            memory
        )

    # ---------------------------------------------------------
    # EVICTION
    # ---------------------------------------------------------

    def _evict_if_needed(self) -> None:
        if len(self._memories) < (
            self.max_memories
        ):
            return

        candidates = [
            memory
            for memory in self._memories.values()
            if memory.active
        ]

        if not candidates:
            return

        candidates.sort(
            key=lambda memory: (
                memory.importance,
                memory.confidence,
                memory.access_count,
                memory.updated_at,
            )
        )

        target_count = max(
            1,
            len(candidates) // 20,
        )

        for memory in candidates[
            :target_count
        ]:
            memory.active = False
            memory.updated_at = utc_iso()

    # ---------------------------------------------------------
    # EXPIRATION
    # ---------------------------------------------------------

    def cleanup_expired(self) -> int:
        with self._lock:
            count = 0

            for memory in self._memories.values():
                if (
                    memory.active
                    and memory.is_expired
                ):
                    memory.active = False
                    memory.updated_at = utc_iso()
                    count += 1

            if count:
                self._auto_save()

            return count

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    def statistics(self) -> MemoryStatistics:
        with self._lock:
            memories = list(
                self._memories.values()
            )

            active = [
                memory
                for memory in memories
                if memory.active
            ]

            expired = [
                memory
                for memory in memories
                if memory.is_expired
            ]

            namespace_counter = Counter(
                memory.namespace
                for memory in memories
            )

            type_counter = Counter(
                memory.memory_type
                for memory in memories
            )

            tag_set = {
                tag
                for memory in memories
                for tag in memory.tags
            }

            average_importance = (
                sum(
                    memory.importance
                    for memory in memories
                )
                / len(memories)
                if memories
                else 0.0
            )

            average_confidence = (
                sum(
                    memory.confidence
                    for memory in memories
                )
                / len(memories)
                if memories
                else 0.0
            )

            return MemoryStatistics(
                total_memories=len(
                    memories
                ),
                active_memories=len(
                    active
                ),
                expired_memories=len(
                    expired
                ),
                total_conversation_turns=sum(
                    len(turns)
                    for turns in
                    self._conversations.values()
                ),
                session_count=len(
                    self._conversations
                ),
                namespace_count=len(
                    namespace_counter
                ),
                tag_count=len(tag_set),
                access_count=(
                    self._memory_access_count
                ),
                average_importance=round(
                    average_importance,
                    4,
                ),
                average_confidence=round(
                    average_confidence,
                    4,
                ),
                memory_type_distribution=dict(
                    type_counter
                ),
                namespace_distribution=dict(
                    namespace_counter
                ),
            )

    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------

    def export_data(
        self,
        *,
        include_inactive: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            if include_inactive:
                memories = list(
                    self._memories.values()
                )
            else:
                memories = [
                    memory
                    for memory
                    in self._memories.values()
                    if memory.active
                ]

            return {
                "schema_version": "1.0",
                "memory_manager_version": (
                    self.VERSION
                ),
                "exported_at": utc_iso(),
                "memories": [
                    memory.to_dict()
                    for memory in memories
                ],
                "conversations": {
                    session: [
                        turn.to_dict()
                        for turn in turns
                    ]
                    for session, turns
                    in self._conversations.items()
                },
                "statistics": (
                    self.statistics().to_dict()
                ),
            }

    def export_json(
        self,
        *,
        indent: int = 2,
        include_inactive: bool = True,
    ) -> str:
        return json.dumps(
            self.export_data(
                include_inactive=include_inactive
            ),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    # ---------------------------------------------------------
    # IMPORT
    # ---------------------------------------------------------

    def import_data(
        self,
        data: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> dict[str, int]:
        with self._lock:
            if not isinstance(data, Mapping):
                raise MemoryValidationError(
                    "Data import harus berupa mapping."
                )

            if replace:
                self._memories.clear()
                self._memory_index.clear()
                self._conversations.clear()

            imported_memories = 0
            imported_turns = 0

            raw_memories = (
                data.get("memories")
                or []
            )

            for raw_memory in raw_memories:
                if not isinstance(
                    raw_memory,
                    Mapping,
                ):
                    continue

                item = MemoryItem.from_dict(
                    raw_memory
                )

                self._memories[
                    item.memory_id
                ] = item

                self._index_memory(
                    item
                )

                imported_memories += 1

            raw_conversations = (
                data.get("conversations")
                or {}
            )

            if isinstance(
                raw_conversations,
                Mapping,
            ):
                for (
                    session_id,
                    raw_turns,
                ) in raw_conversations.items():
                    if not isinstance(
                        raw_turns,
                        list,
                    ):
                        continue

                    session_key = (
                        normalize_text(
                            session_id
                        )
                        or "default"
                    )

                    self._conversations[
                        session_key
                    ] = []

                    for raw_turn in raw_turns:
                        if not isinstance(
                            raw_turn,
                            Mapping,
                        ):
                            continue

                        turn = (
                            ConversationTurn
                            .from_dict(raw_turn)
                        )

                        self._conversations[
                            session_key
                        ].append(turn)

                        imported_turns += 1

            return {
                "memories": imported_memories,
                "conversation_turns": (
                    imported_turns
                ),
            }

    def import_json(
        self,
        payload: str,
        *,
        replace: bool = False,
    ) -> dict[str, int]:
        try:
            data = json.loads(
                payload
            )
        except json.JSONDecodeError as exc:
            raise MemoryValidationError(
                f"JSON memory tidak valid: {exc}"
            ) from exc

        return self.import_data(
            data,
            replace=replace,
        )

    # ---------------------------------------------------------
    # PERSISTENCE
    # ---------------------------------------------------------

    def save(
        self,
        path: str | Path | None = None,
    ) -> Path:
        with self._lock:
            target = (
                Path(path)
                if path
                else self.storage_path
            )

            if target is None:
                raise MemoryStorageError(
                    "Storage path belum dikonfigurasi."
                )

            try:
                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary = target.with_suffix(
                    target.suffix + ".tmp"
                )

                temporary.write_text(
                    self.export_json(
                        indent=2,
                        include_inactive=True,
                    ),
                    encoding="utf-8",
                )

                temporary.replace(
                    target
                )

                return target

            except Exception as exc:
                raise MemoryStorageError(
                    (
                        "Gagal menyimpan memory "
                        f"ke {target}: {exc}"
                    )
                ) from exc

    def load(
        self,
        path: str | Path | None = None,
    ) -> dict[str, int]:
        with self._lock:
            target = (
                Path(path)
                if path
                else self.storage_path
            )

            if target is None:
                raise MemoryStorageError(
                    "Storage path belum dikonfigurasi."
                )

            if not target.exists():
                return {
                    "memories": 0,
                    "conversation_turns": 0,
                }

            try:
                payload = target.read_text(
                    encoding="utf-8"
                )

                data = json.loads(
                    payload
                )

                return self.import_data(
                    data,
                    replace=True,
                )

            except Exception as exc:
                raise MemoryStorageError(
                    (
                        "Gagal memuat memory "
                        f"dari {target}: {exc}"
                    )
                ) from exc

    def _auto_save(self) -> None:
        if not self.auto_save:
            return

        if self.storage_path is None:
            return

        try:
            self.save()
        except Exception:
            # Memory operation tidak boleh gagal hanya
            # karena auto-save mengalami masalah.
            pass

    # ---------------------------------------------------------
    # SNAPSHOT
    # ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "created_at": utc_iso(),
                "info": self.info(),
                "health": self.health(),
                "statistics": (
                    self.statistics().to_dict()
                ),
                "data": self.export_data(
                    include_inactive=True
                ),
            }

    # ---------------------------------------------------------
    # MEMORY PROMOTION
    # ---------------------------------------------------------

    def promote(
        self,
        memory_id: str,
        *,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
    ) -> MemoryItem:
        item = self.get_by_id(
            memory_id,
            touch=False,
        )

        if item is None:
            raise MemoryNotFoundError(
                f"Memory '{memory_id}' tidak ditemukan."
            )

        item.importance = max(
            item.importance,
            safe_float(
                importance,
                item.importance,
            ),
        )

        item.confidence = max(
            item.confidence,
            safe_float(
                confidence,
                item.confidence,
            ),
        )

        if source:
            item.source = normalize_text(
                source
            )

        item.updated_at = utc_iso()
        item.version += 1

        self._memory_updated_count += 1

        self._auto_save()

        return item

    # ---------------------------------------------------------
    # MEMORY MERGE
    # ---------------------------------------------------------

    def merge(
        self,
        memory_ids: Iterable[str],
        *,
        target_key: str,
        namespace: str = DEFAULT_NAMESPACE,
        memory_type: str = "summary",
        source: str = "memory_merge",
    ) -> MemoryItem:
        with self._lock:
            selected: list[
                MemoryItem
            ] = []

            for memory_id in memory_ids:
                item = self.get_by_id(
                    memory_id,
                    touch=False,
                )

                if item is not None:
                    selected.append(item)

            if not selected:
                raise MemoryNotFoundError(
                    "Tidak ada memory yang dapat digabung."
                )

            values = [
                item.value
                for item in selected
            ]

            tags = sorted(
                {
                    tag
                    for item in selected
                    for tag in item.tags
                }
            )

            metadata = {
                "merged_memory_ids": [
                    item.memory_id
                    for item in selected
                ],
                "merged_count": len(
                    selected
                ),
            }

            importance = max(
                item.importance
                for item in selected
            )

            confidence = min(
                item.confidence
                for item in selected
            )

            merged = self.remember(
                key=target_key,
                value=values,
                namespace=namespace,
                memory_type=memory_type,
                source=source,
                importance=importance,
                confidence=confidence,
                tags=tags,
                metadata=metadata,
                overwrite=True,
            )

            return merged

    # ---------------------------------------------------------
    # MEMORY TYPES
    # ---------------------------------------------------------

    def by_type(
        self,
        memory_type: str,
        *,
        limit: int = 100,
    ) -> list[MemoryItem]:
        normalized_type = normalize_key(
            memory_type
        )

        results = [
            memory
            for memory in self.all()
            if memory.memory_type
            == normalized_type
        ]

        return results[:max(
            1,
            safe_int(limit, 100),
        )]

    # ---------------------------------------------------------
    # TAGS
    # ---------------------------------------------------------

    def by_tag(
        self,
        tag: str,
        *,
        limit: int = 100,
    ) -> list[MemoryItem]:
        normalized_tag = normalize_key(
            tag
        )

        results = [
            memory
            for memory in self.all()
            if normalized_tag
            in memory.tags
        ]

        return results[:max(
            1,
            safe_int(limit, 100),
        )]

    def tags(self) -> list[str]:
        with self._lock:
            return sorted(
                {
                    tag
                    for memory in self._memories.values()
                    for tag in memory.tags
                }
            )

    def namespaces(self) -> list[str]:
        with self._lock:
            return sorted(
                {
                    memory.namespace
                    for memory
                    in self._memories.values()
                }
            )

    # ---------------------------------------------------------
    # EXECUTION WRAPPER
    # ---------------------------------------------------------

    def execute(
        self,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        operation_name = normalize_key(
            operation
        )

        if operation_name == "remember":
            return self.remember(**kwargs)

        if operation_name == "search":
            return self.search(**kwargs)

        if operation_name == "get":
            return self.get(**kwargs)

        if operation_name == "update":
            memory_id = kwargs.pop(
                "memory_id"
            )

            return self.update(
                memory_id,
                **kwargs,
            )

        if operation_name == "forget":
            return self.forget(**kwargs)

        if operation_name == "add_turn":
            return self.add_turn(**kwargs)

        if operation_name == "conversation":
            return self.get_conversation(
                **kwargs
            )

        if operation_name == "context":
            return self.build_context(
                **kwargs
            )

        if operation_name == "prompt_context":
            return self.build_prompt_context(
                **kwargs
            )

        if operation_name == "statistics":
            return self.statistics()

        if operation_name == "health":
            return self.health()

        if operation_name == "info":
            return self.info()

        if operation_name == "save":
            return self.save(**kwargs)

        if operation_name == "load":
            return self.load(**kwargs)

        if operation_name == "cleanup":
            return self.cleanup_expired()

        raise MemoryValidationError(
            (
                "Memory operation tidak dikenal: "
                f"{operation}"
            )
        )


__all__ = [
    "MemoryErrorBase",
    "MemoryValidationError",
    "MemoryNotFoundError",
    "MemoryStorageError",
    "MemoryItem",
    "MemorySearchResult",
    "ConversationTurn",
    "MemoryStatistics",
    "MemoryManager",
]
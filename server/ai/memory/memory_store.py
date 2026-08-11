from __future__ import annotations

"""
ZAI Memory Store
================

Central memory storage layer for Super ZAI.

Design goals
------------

1. Stable public API.
2. No mandatory external database dependency.
3. Deterministic tests.
4. Session-aware conversation memory.
5. Namespace support.
6. Tag support.
7. Importance and confidence scoring.
8. Expiration support.
9. Search and ranking.
10. Context generation.
11. Statistics.
12. Safe serialization.
13. Compatibility aliases.
14. Easy future migration to SQLite / vector DB.
15. Zero third-party dependency.

This module is intentionally self-contained.

Public classes
--------------

    MemoryRecord
    MemoryEntry
    MemorySession
    MemorySearchResult
    MemoryStore
    MemoryManager

Public helpers
--------------

    create_memory_store()
    self_test()

The implementation is designed to be compatible with the current
Super ZAI memory layer while keeping the storage engine independent
from FastAPI, agents, tools, or the brain layer.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Iterable, Iterator, Mapping, Optional
from uuid import UUID, uuid4
import math
import re


# ============================================================================
# VERSION
# ============================================================================

MEMORY_STORE_VERSION = "2.0.0"


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_NAMESPACE = "default"
DEFAULT_MEMORY_TYPE = "fact"

DEFAULT_IMPORTANCE = 1.0
DEFAULT_CONFIDENCE = 1.0

DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 1000

DEFAULT_CONTEXT_LIMIT = 10
MAX_CONTEXT_LIMIT = 100

MAX_CONTENT_LENGTH = 100_000
MAX_TAG_LENGTH = 100
MAX_NAMESPACE_LENGTH = 200
MAX_KEY_LENGTH = 500
MAX_SESSION_ID_LENGTH = 200

DEFAULT_SESSION_TITLE = "ZAI Session"

UTC = timezone.utc


# ============================================================================
# TIME HELPERS
# ============================================================================


def utc_now() -> datetime:
    """
    Return current timezone-aware UTC datetime.
    """
    return datetime.now(UTC)


def ensure_datetime(value: Any) -> Optional[datetime]:
    """
    Normalize a datetime-like value.

    Supported values:

    - None
    - datetime
    - ISO formatted string
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        try:
            parsed = datetime.fromisoformat(
                text.replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)

            return parsed.astimezone(UTC)

        except ValueError:
            return None

    return None


def iso_now() -> str:
    """
    Current UTC time in ISO format.
    """
    return utc_now().isoformat()


def datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    """
    Convert datetime to ISO string.
    """
    if value is None:
        return None

    normalized = ensure_datetime(value)

    if normalized is None:
        return None

    return normalized.isoformat()


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================


def normalize_text(value: Any) -> str:
    """
    Normalize arbitrary input into a compact string.
    """
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_namespace(value: Any) -> str:
    """
    Normalize namespace.
    """
    namespace = normalize_text(value)

    if not namespace:
        return DEFAULT_NAMESPACE

    return namespace[:MAX_NAMESPACE_LENGTH]


def normalize_key(value: Any) -> str:
    """
    Normalize memory key.
    """
    key = normalize_text(value)

    if not key:
        return ""

    return key[:MAX_KEY_LENGTH]


def normalize_session_id(value: Any) -> str:
    """
    Normalize session identifier.
    """
    session_id = normalize_text(value)

    if not session_id:
        return ""

    return session_id[:MAX_SESSION_ID_LENGTH]


def normalize_content(value: Any) -> str:
    """
    Normalize memory content.
    """
    content = normalize_text(value)

    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH]

    return content


def normalize_memory_type(value: Any) -> str:
    """
    Normalize memory type.
    """
    memory_type = normalize_text(value).lower()

    if not memory_type:
        return DEFAULT_MEMORY_TYPE

    return memory_type


def normalize_tags(
    tags: Optional[Iterable[Any]],
) -> list[str]:
    """
    Normalize tags.

    Duplicate tags are removed while preserving insertion order.
    """
    if tags is None:
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


def clamp_score(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Clamp numeric score to [0, 1].
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default

    if math.isnan(score):
        score = default

    if math.isinf(score):
        score = 1.0 if score > 0 else 0.0

    return max(
        0.0,
        min(1.0, score),
    )


def normalize_limit(
    value: Any,
    default: int = DEFAULT_SEARCH_LIMIT,
) -> int:
    """
    Normalize result limit.
    """
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = default

    if limit <= 0:
        return default

    return min(
        limit,
        MAX_SEARCH_LIMIT,
    )


# ============================================================================
# TOKENIZATION
# ============================================================================


TOKEN_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+",
    re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized tokens.
    """
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(
            normalize_text(text)
        )
        if token
    ]


def unique_tokens(text: str) -> set[str]:
    """
    Return unique normalized tokens.
    """
    return set(tokenize(text))


# ============================================================================
# MEMORY RECORD
# ============================================================================


@dataclass
class MemoryRecord:
    """
    Canonical memory object.

    A MemoryRecord represents one persistent logical memory.
    """

    id: str = field(
        default_factory=lambda: str(uuid4())
    )

    content: str = ""

    namespace: str = DEFAULT_NAMESPACE

    memory_type: str = DEFAULT_MEMORY_TYPE

    key: str = ""

    tags: list[str] = field(
        default_factory=list
    )

    importance: float = DEFAULT_IMPORTANCE

    confidence: float = DEFAULT_CONFIDENCE

    source: str = "zai"

    session_id: Optional[str] = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    accessed_at: Optional[datetime] = None

    expires_at: Optional[datetime] = None

    access_count: int = 0

    active: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.id = normalize_text(self.id) or str(uuid4())

        self.content = normalize_content(
            self.content
        )

        self.namespace = normalize_namespace(
            self.namespace
        )

        self.memory_type = normalize_memory_type(
            self.memory_type
        )

        self.key = normalize_key(
            self.key
        )

        self.tags = normalize_tags(
            self.tags
        )

        self.importance = clamp_score(
            self.importance,
            DEFAULT_IMPORTANCE,
        )

        self.confidence = clamp_score(
            self.confidence,
            DEFAULT_CONFIDENCE,
        )

        self.source = (
            normalize_text(self.source)
            or "zai"
        )

        normalized_session = normalize_session_id(
            self.session_id
        )

        self.session_id = (
            normalized_session
            if normalized_session
            else None
        )

        self.created_at = (
            ensure_datetime(self.created_at)
            or utc_now()
        )

        self.updated_at = (
            ensure_datetime(self.updated_at)
            or self.created_at
        )

        self.accessed_at = ensure_datetime(
            self.accessed_at
        )

        self.expires_at = ensure_datetime(
            self.expires_at
        )

        try:
            self.access_count = max(
                0,
                int(self.access_count),
            )
        except (TypeError, ValueError):
            self.access_count = 0

        self.active = bool(self.active)

        if not isinstance(self.metadata, dict):
            self.metadata = dict(
                self.metadata or {}
            )

    @property
    def memory_id(self) -> str:
        """
        Compatibility alias.
        """
        return self.id

    @property
    def is_expired(self) -> bool:
        """
        Whether this memory has expired.
        """
        if self.expires_at is None:
            return False

        return utc_now() >= self.expires_at

    @property
    def is_available(self) -> bool:
        """
        Whether this memory can currently be used.
        """
        return (
            self.active
            and not self.is_expired
        )

    @property
    def text(self) -> str:
        """
        Compatibility alias for content.
        """
        return self.content

    def touch(self) -> None:
        """
        Mark memory as accessed.
        """
        self.access_count += 1
        self.accessed_at = utc_now()

    def update(
        self,
        *,
        content: Optional[str] = None,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        key: Optional[str] = None,
        tags: Optional[Iterable[Any]] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        session_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        active: Optional[bool] = None,
    ) -> "MemoryRecord":
        """
        Update selected fields.
        """
        if content is not None:
            self.content = normalize_content(
                content
            )

        if namespace is not None:
            self.namespace = normalize_namespace(
                namespace
            )

        if memory_type is not None:
            self.memory_type = normalize_memory_type(
                memory_type
            )

        if key is not None:
            self.key = normalize_key(
                key
            )

        if tags is not None:
            self.tags = normalize_tags(
                tags
            )

        if importance is not None:
            self.importance = clamp_score(
                importance,
                self.importance,
            )

        if confidence is not None:
            self.confidence = clamp_score(
                confidence,
                self.confidence,
            )

        if source is not None:
            self.source = (
                normalize_text(source)
                or self.source
            )

        if session_id is not None:
            normalized_session = normalize_session_id(
                session_id
            )

            self.session_id = (
                normalized_session
                if normalized_session
                else None
            )

        if expires_at is not None:
            self.expires_at = ensure_datetime(
                expires_at
            )

        if metadata is not None:
            self.metadata = dict(metadata)

        if active is not None:
            self.active = bool(active)

        self.updated_at = utc_now()

        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize memory into JSON-friendly dictionary.
        """
        return {
            "id": self.id,
            "memory_id": self.id,
            "content": self.content,
            "text": self.content,
            "namespace": self.namespace,
            "memory_type": self.memory_type,
            "key": self.key,
            "tags": list(self.tags),
            "importance": self.importance,
            "confidence": self.confidence,
            "source": self.source,
            "session_id": self.session_id,
            "created_at": datetime_to_iso(
                self.created_at
            ),
            "updated_at": datetime_to_iso(
                self.updated_at
            ),
            "accessed_at": datetime_to_iso(
                self.accessed_at
            ),
            "expires_at": datetime_to_iso(
                self.expires_at
            ),
            "access_count": self.access_count,
            "active": self.active,
            "expired": self.is_expired,
            "available": self.is_available,
            "metadata": dict(self.metadata),
        }

    def clone(self) -> "MemoryRecord":
        """
        Return an independent copy.
        """
        return MemoryRecord(
            id=self.id,
            content=self.content,
            namespace=self.namespace,
            memory_type=self.memory_type,
            key=self.key,
            tags=list(self.tags),
            importance=self.importance,
            confidence=self.confidence,
            source=self.source,
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            accessed_at=self.accessed_at,
            expires_at=self.expires_at,
            access_count=self.access_count,
            active=self.active,
            metadata=dict(self.metadata),
        )


# ============================================================================
# COMPATIBILITY ALIAS
# ============================================================================


MemoryEntry = MemoryRecord


# ============================================================================
# MEMORY SESSION
# ============================================================================


@dataclass
class MemorySession:
    """
    Conversation/session container.

    This class is intentionally exported from memory_store because
    ai.memory.__init__ expects MemorySession to be available here.
    """

    id: str = field(
        default_factory=lambda: str(uuid4())
    )

    title: str = DEFAULT_SESSION_TITLE

    namespace: str = DEFAULT_NAMESPACE

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    closed_at: Optional[datetime] = None

    active: bool = True

    turn_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.id = normalize_session_id(
            self.id
        ) or str(uuid4())

        self.title = (
            normalize_text(self.title)
            or DEFAULT_SESSION_TITLE
        )

        self.namespace = normalize_namespace(
            self.namespace
        )

        self.created_at = (
            ensure_datetime(self.created_at)
            or utc_now()
        )

        self.updated_at = (
            ensure_datetime(self.updated_at)
            or self.created_at
        )

        self.closed_at = ensure_datetime(
            self.closed_at
        )

        try:
            self.turn_count = max(
                0,
                int(self.turn_count),
            )
        except (TypeError, ValueError):
            self.turn_count = 0

        self.active = bool(self.active)

        if not isinstance(self.metadata, dict):
            self.metadata = dict(
                self.metadata or {}
            )

    @property
    def session_id(self) -> str:
        """
        Compatibility alias.
        """
        return self.id

    @property
    def is_closed(self) -> bool:
        """
        Whether session is closed.
        """
        return (
            not self.active
            or self.closed_at is not None
        )

    def touch(self) -> None:
        """
        Mark session as used.
        """
        self.turn_count += 1
        self.updated_at = utc_now()

    def close(self) -> None:
        """
        Close session.
        """
        self.active = False
        self.closed_at = utc_now()
        self.updated_at = self.closed_at

    def reopen(self) -> None:
        """
        Reopen session.
        """
        self.active = True
        self.closed_at = None
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize session.
        """
        return {
            "id": self.id,
            "session_id": self.id,
            "title": self.title,
            "namespace": self.namespace,
            "created_at": datetime_to_iso(
                self.created_at
            ),
            "updated_at": datetime_to_iso(
                self.updated_at
            ),
            "closed_at": datetime_to_iso(
                self.closed_at
            ),
            "active": self.active,
            "closed": self.is_closed,
            "turn_count": self.turn_count,
            "metadata": dict(self.metadata),
        }

    def clone(self) -> "MemorySession":
        """
        Return independent session copy.
        """
        return MemorySession(
            id=self.id,
            title=self.title,
            namespace=self.namespace,
            created_at=self.created_at,
            updated_at=self.updated_at,
            closed_at=self.closed_at,
            active=self.active,
            turn_count=self.turn_count,
            metadata=dict(self.metadata),
        )


# ============================================================================
# SEARCH RESULT
# ============================================================================


@dataclass
class MemorySearchResult:
    """
    Ranked search result.
    """

    memory: MemoryRecord

    score: float

    matched_tokens: list[str] = field(
        default_factory=list
    )

    reason: str = "text_match"

    def __post_init__(self) -> None:
        self.score = max(
            0.0,
            min(
                1.0,
                float(self.score),
            ),
        )

        self.matched_tokens = list(
            self.matched_tokens
        )

        self.reason = (
            normalize_text(self.reason)
            or "text_match"
        )

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def content(self) -> str:
        return self.memory.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.memory.id,
            "memory_id": self.memory.id,
            "content": self.memory.content,
            "score": self.score,
            "matched_tokens": list(
                self.matched_tokens
            ),
            "reason": self.reason,
            "memory": self.memory.to_dict(),
        }


# ============================================================================
# MEMORY STORE
# ============================================================================


class MemoryStore:
    """
    Thread-safe in-memory memory store.

    The storage layer intentionally does not require SQLite.

    A future persistence adapter can be attached without changing
    the public memory API.
    """

    VERSION = MEMORY_STORE_VERSION

    def __init__(
        self,
        *,
        storage_enabled: bool = False,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        self.storage_enabled = bool(
            storage_enabled
        )

        self.default_namespace = normalize_namespace(
            namespace
        )

        self._memories: dict[
            str,
            MemoryRecord,
        ] = {}

        self._sessions: dict[
            str,
            MemorySession,
        ] = {}

        self._lock = RLock()

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.access_count = 0

    # ========================================================================
    # BASIC PROPERTIES
    # ========================================================================

    @property
    def memory_count(self) -> int:
        """
        Number of stored memories.
        """
        with self._lock:
            return len(self._memories)

    @property
    def session_count(self) -> int:
        """
        Number of sessions.
        """
        with self._lock:
            return len(self._sessions)

    @property
    def storage_status(self) -> str:
        """
        Storage status.
        """
        return (
            "ENABLED"
            if self.storage_enabled
            else "DISABLED"
        )

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    def create_session(
        self,
        *,
        session_id: Optional[str] = None,
        title: str = DEFAULT_SESSION_TITLE,
        namespace: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemorySession:
        """
        Create a new session.
        """
        with self._lock:
            normalized_id = normalize_session_id(
                session_id
            )

            if normalized_id:
                existing = self._sessions.get(
                    normalized_id
                )

                if existing is not None:
                    return existing.clone()

            session = MemorySession(
                id=normalized_id or str(uuid4()),
                title=title,
                namespace=(
                    namespace
                    or self.default_namespace
                ),
                metadata=dict(
                    metadata or {}
                ),
            )

            self._sessions[
                session.id
            ] = session

            return session.clone()

    def get_session(
        self,
        session_id: str,
    ) -> Optional[MemorySession]:
        """
        Retrieve session.
        """
        normalized_id = normalize_session_id(
            session_id
        )

        if not normalized_id:
            return None

        with self._lock:
            session = self._sessions.get(
                normalized_id
            )

            return (
                session.clone()
                if session is not None
                else None
            )

    def ensure_session(
        self,
        session_id: Optional[str] = None,
        *,
        title: str = DEFAULT_SESSION_TITLE,
        namespace: Optional[str] = None,
    ) -> MemorySession:
        """
        Return existing session or create one.
        """
        if session_id:
            existing = self.get_session(
                session_id
            )

            if existing is not None:
                return existing

        return self.create_session(
            session_id=session_id,
            title=title,
            namespace=namespace,
        )

    def update_session(
        self,
        session_id: str,
        *,
        title: Optional[str] = None,
        namespace: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        active: Optional[bool] = None,
    ) -> Optional[MemorySession]:
        """
        Update session metadata.
        """
        normalized_id = normalize_session_id(
            session_id
        )

        if not normalized_id:
            return None

        with self._lock:
            session = self._sessions.get(
                normalized_id
            )

            if session is None:
                return None

            if title is not None:
                session.title = (
                    normalize_text(title)
                    or session.title
                )

            if namespace is not None:
                session.namespace = (
                    normalize_namespace(namespace)
                )

            if metadata is not None:
                session.metadata = dict(metadata)

            if active is not None:
                session.active = bool(active)

                if not session.active:
                    session.closed_at = utc_now()
                else:
                    session.closed_at = None

            session.updated_at = utc_now()

            return session.clone()

    def close_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Close session.
        """
        normalized_id = normalize_session_id(
            session_id
        )

        if not normalized_id:
            return False

        with self._lock:
            session = self._sessions.get(
                normalized_id
            )

            if session is None:
                return False

            session.close()

            return True

    def reopen_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Reopen session.
        """
        normalized_id = normalize_session_id(
            session_id
        )

        if not normalized_id:
            return False

        with self._lock:
            session = self._sessions.get(
                normalized_id
            )

            if session is None:
                return False

            session.reopen()

            return True

    def list_sessions(
        self,
        *,
        namespace: Optional[str] = None,
        active_only: bool = False,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[MemorySession]:
        """
        List sessions.
        """
        normalized_namespace = (
            normalize_namespace(namespace)
            if namespace is not None
            else None
        )

        limit = normalize_limit(limit)

        with self._lock:
            sessions = list(
                self._sessions.values()
            )

            if normalized_namespace is not None:
                sessions = [
                    item
                    for item in sessions
                    if item.namespace
                    == normalized_namespace
                ]

            if active_only:
                sessions = [
                    item
                    for item in sessions
                    if item.active
                ]

            sessions.sort(
                key=lambda item: item.updated_at,
                reverse=True,
            )

            return [
                item.clone()
                for item in sessions[:limit]
            ]

    # ========================================================================
    # MEMORY CREATION
    # ========================================================================

    def create(
        self,
        content: str,
        *,
        namespace: Optional[str] = None,
        memory_type: str = DEFAULT_MEMORY_TYPE,
        key: str = "",
        tags: Optional[Iterable[Any]] = None,
        importance: float = DEFAULT_IMPORTANCE,
        confidence: float = DEFAULT_CONFIDENCE,
        source: str = "zai",
        session_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryRecord:
        """
        Create and store a memory.
        """
        content = normalize_content(content)

        if not content:
            raise ValueError(
                "Memory content tidak boleh kosong."
            )

        if ttl_seconds is not None:
            try:
                ttl = float(ttl_seconds)
            except (TypeError, ValueError):
                ttl = 0.0

            if ttl > 0:
                expires_at = (
                    utc_now()
                    + timedelta(
                        seconds=ttl
                    )
                )

        record = MemoryRecord(
            content=content,
            namespace=(
                namespace
                or self.default_namespace
            ),
            memory_type=memory_type,
            key=key,
            tags=list(tags or []),
            importance=importance,
            confidence=confidence,
            source=source,
            session_id=session_id,
            expires_at=expires_at,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.save(record)

    def save(
        self,
        memory: MemoryRecord,
    ) -> MemoryRecord:
        """
        Save or replace a memory.
        """
        if not isinstance(
            memory,
            MemoryRecord,
        ):
            raise TypeError(
                "memory harus merupakan MemoryRecord."
            )

        if not memory.content:
            raise ValueError(
                "Memory content tidak boleh kosong."
            )

        with self._lock:
            memory.updated_at = utc_now()

            self._memories[
                memory.id
            ] = memory.clone()

            if memory.session_id:
                session = self._sessions.get(
                    memory.session_id
                )

                if session is None:
                    self._sessions[
                        memory.session_id
                    ] = MemorySession(
                        id=memory.session_id,
                        namespace=memory.namespace,
                    )

                session = self._sessions[
                    memory.session_id
                ]

                session.touch()

            self.execution_count += 1
            self.success_count += 1

            return memory.clone()

    # ========================================================================
    # UPSERT
    # ========================================================================

    def upsert(
        self,
        content: str,
        *,
        key: str = "",
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> MemoryRecord:
        """
        Create or update by logical key.

        If key is empty, a new record is created.
        """
        normalized_key = normalize_key(key)

        if not normalized_key:
            return self.create(
                content,
                namespace=namespace,
                key=key,
                **kwargs,
            )

        existing = self.find_by_key(
            normalized_key,
            namespace=namespace,
        )

        if existing is None:
            return self.create(
                content,
                namespace=namespace,
                key=normalized_key,
                **kwargs,
            )

        return self.update(
            existing.id,
            content=content,
            **kwargs,
        ) or existing

    # ========================================================================
    # READ
    # ========================================================================

    def get(
        self,
        memory_id: str,
    ) -> Optional[MemoryRecord]:
        """
        Get memory by ID.
        """
        memory_id = normalize_text(
            memory_id
        )

        if not memory_id:
            return None

        with self._lock:
            memory = self._memories.get(
                memory_id
            )

            return (
                memory.clone()
                if memory is not None
                else None
            )

    def require(
        self,
        memory_id: str,
    ) -> MemoryRecord:
        """
        Get memory or raise KeyError.
        """
        memory = self.get(memory_id)

        if memory is None:
            raise KeyError(
                f"Memory '{memory_id}' tidak ditemukan."
            )

        return memory

    def exists(
        self,
        memory_id: str,
    ) -> bool:
        """
        Check whether memory exists.
        """
        return self.get(memory_id) is not None

    def find_by_key(
        self,
        key: str,
        *,
        namespace: Optional[str] = None,
        include_inactive: bool = False,
    ) -> Optional[MemoryRecord]:
        """
        Find one memory by logical key.

        The newest matching record is returned.
        """
        normalized_key = normalize_key(key)

        if not normalized_key:
            return None

        normalized_namespace = (
            normalize_namespace(namespace)
            if namespace is not None
            else None
        )

        with self._lock:
            matches = [
                memory
                for memory in self._memories.values()
                if memory.key == normalized_key
                and (
                    normalized_namespace is None
                    or memory.namespace
                    == normalized_namespace
                )
                and (
                    include_inactive
                    or memory.is_available
                )
            ]

            if not matches:
                return None

            matches.sort(
                key=lambda item: item.updated_at,
                reverse=True,
            )

            return matches[0].clone()

    def all(
        self,
        *,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        include_inactive: bool = False,
        include_expired: bool = False,
        limit: Optional[int] = None,
    ) -> list[MemoryRecord]:
        """
        Return memories matching filters.
        """
        normalized_namespace = (
            normalize_namespace(namespace)
            if namespace is not None
            else None
        )

        normalized_type = (
            normalize_memory_type(memory_type)
            if memory_type is not None
            else None
        )

        normalized_session = (
            normalize_session_id(session_id)
            if session_id is not None
            else None
        )

        with self._lock:
            memories = list(
                self._memories.values()
            )

            filtered: list[
                MemoryRecord
            ] = []

            for memory in memories:
                if (
                    normalized_namespace is not None
                    and memory.namespace
                    != normalized_namespace
                ):
                    continue

                if (
                    normalized_type is not None
                    and memory.memory_type
                    != normalized_type
                ):
                    continue

                if (
                    normalized_session is not None
                    and memory.session_id
                    != normalized_session
                ):
                    continue

                if (
                    not include_inactive
                    and not memory.active
                ):
                    continue

                if (
                    not include_expired
                    and memory.is_expired
                ):
                    continue

                filtered.append(
                    memory
                )

            filtered.sort(
                key=lambda item: item.updated_at,
                reverse=True,
            )

            if limit is not None:
                normalized_limit = normalize_limit(
                    limit
                )

                filtered = filtered[
                    :normalized_limit
                ]

            return [
                memory.clone()
                for memory in filtered
            ]

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        memory_id: str,
        **kwargs: Any,
    ) -> Optional[MemoryRecord]:
        """
        Update a memory.
        """
        normalized_id = normalize_text(
            memory_id
        )

        if not normalized_id:
            return None

        with self._lock:
            memory = self._memories.get(
                normalized_id
            )

            if memory is None:
                return None

            memory.update(
                **kwargs
            )

            self._memories[
                normalized_id
            ] = memory.clone()

            self.execution_count += 1
            self.success_count += 1

            return memory.clone()

    # ========================================================================
    # ACCESS
    # ========================================================================

    def touch(
        self,
        memory_id: str,
    ) -> Optional[MemoryRecord]:
        """
        Record a memory access.
        """
        normalized_id = normalize_text(
            memory_id
        )

        if not normalized_id:
            return None

        with self._lock:
            memory = self._memories.get(
                normalized_id
            )

            if memory is None:
                return None

            memory.touch()

            self.access_count += 1

            self._memories[
                normalized_id
            ] = memory.clone()

            return memory.clone()

    # ========================================================================
    # DELETE
    # ========================================================================

    def delete(
        self,
        memory_id: str,
        *,
        hard: bool = False,
    ) -> bool:
        """
        Delete memory.

        hard=False:
            soft delete

        hard=True:
            remove completely
        """
        normalized_id = normalize_text(
            memory_id
        )

        if not normalized_id:
            return False

        with self._lock:
            memory = self._memories.get(
                normalized_id
            )

            if memory is None:
                return False

            if hard:
                del self._memories[
                    normalized_id
                ]

            else:
                memory.active = False
                memory.updated_at = utc_now()

                self._memories[
                    normalized_id
                ] = memory

            self.execution_count += 1
            self.success_count += 1

            return True

    def clear(
        self,
        *,
        namespace: Optional[str] = None,
        hard: bool = True,
    ) -> int:
        """
        Clear memories.

        Returns number of affected memories.
        """
        normalized_namespace = (
            normalize_namespace(namespace)
            if namespace is not None
            else None
        )

        with self._lock:
            matching_ids = [
                memory.id
                for memory in self._memories.values()
                if (
                    normalized_namespace is None
                    or memory.namespace
                    == normalized_namespace
                )
            ]

            if hard:
                for memory_id in matching_ids:
                    self._memories.pop(
                        memory_id,
                        None,
                    )

            else:
                for memory_id in matching_ids:
                    memory = self._memories.get(
                        memory_id
                    )

                    if memory is None:
                        continue

                    memory.active = False
                    memory.updated_at = utc_now()

            affected = len(
                matching_ids
            )

            self.execution_count += 1
            self.success_count += 1

            return affected

    # ========================================================================
    # EXPIRATION
    # ========================================================================

    def purge_expired(
        self,
        *,
        hard: bool = True,
    ) -> int:
        """
        Remove or deactivate expired memories.
        """
        with self._lock:
            expired_ids = [
                memory.id
                for memory in self._memories.values()
                if memory.is_expired
            ]

            if hard:
                for memory_id in expired_ids:
                    self._memories.pop(
                        memory_id,
                        None,
                    )

            else:
                for memory_id in expired_ids:
                    memory = self._memories.get(
                        memory_id
                    )

                    if memory is None:
                        continue

                    memory.active = False
                    memory.updated_at = utc_now()

            return len(
                expired_ids
            )

    # ========================================================================
    # SEARCH
    # ========================================================================

    def search(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[Iterable[Any]] = None,
        min_score: float = 0.0,
        limit: int = DEFAULT_SEARCH_LIMIT,
        include_inactive: bool = False,
        touch_results: bool = False,
    ) -> list[MemorySearchResult]:
        """
        Search memories using deterministic lexical ranking.

        Ranking factors:

        - exact phrase
        - token overlap
        - tag overlap
        - key overlap
        - importance
        - confidence
        """
        normalized_query = normalize_text(
            query
        )

        if not normalized_query:
            return []

        query_tokens = unique_tokens(
            normalized_query
        )

        query_token_list = list(
            query_tokens
        )

        normalized_namespace = (
            normalize_namespace(namespace)
            if namespace is not None
            else None
        )

        normalized_type = (
            normalize_memory_type(memory_type)
            if memory_type is not None
            else None
        )

        normalized_session = (
            normalize_session_id(session_id)
            if session_id is not None
            else None
        )

        requested_tags = set(
            normalize_tags(tags)
        )

        threshold = max(
            0.0,
            min(
                1.0,
                float(min_score),
            ),
        )

        normalized_limit = normalize_limit(
            limit
        )

        with self._lock:
            memories = list(
                self._memories.values()
            )

        results: list[
            MemorySearchResult
        ] = []

        query_lower = normalized_query.lower()

        for memory in memories:
            if (
                normalized_namespace is not None
                and memory.namespace
                != normalized_namespace
            ):
                continue

            if (
                normalized_type is not None
                and memory.memory_type
                != normalized_type
            ):
                continue

            if (
                normalized_session is not None
                and memory.session_id
                != normalized_session
            ):
                continue

            if (
                not include_inactive
                and not memory.is_available
            ):
                continue

            content_lower = (
                memory.content.lower()
            )

            content_tokens = unique_tokens(
                memory.content
            )

            matched = (
                query_tokens
                & content_tokens
            )

            token_score = 0.0

            if query_tokens:
                token_score = (
                    len(matched)
                    / len(query_tokens)
                )

            phrase_score = (
                1.0
                if query_lower in content_lower
                else 0.0
            )

            tag_score = 0.0

            if requested_tags:
                tag_matches = (
                    requested_tags
                    & set(memory.tags)
                )

                tag_score = (
                    len(tag_matches)
                    / len(requested_tags)
                )

            key_score = 0.0

            if memory.key:
                key_tokens = unique_tokens(
                    memory.key
                )

                if key_tokens:
                    key_matches = (
                        query_tokens
                        & key_tokens
                    )

                    key_score = (
                        len(key_matches)
                        / len(key_tokens)
                    )

            base_score = (
                token_score * 0.55
                + phrase_score * 0.20
                + tag_score * 0.10
                + key_score * 0.05
                + memory.importance * 0.05
                + memory.confidence * 0.05
            )

            score = max(
                0.0,
                min(
                    1.0,
                    base_score,
                ),
            )

            if score < threshold:
                continue

            reason_parts: list[str] = []

            if phrase_score:
                reason_parts.append(
                    "exact_phrase"
                )

            if matched:
                reason_parts.append(
                    "token_match"
                )

            if tag_score:
                reason_parts.append(
                    "tag_match"
                )

            if key_score:
                reason_parts.append(
                    "key_match"
                )

            if not reason_parts:
                reason_parts.append(
                    "metadata_match"
                )

            results.append(
                MemorySearchResult(
                    memory=memory.clone(),
                    score=score,
                    matched_tokens=sorted(
                        matched
                    ),
                    reason=",".join(
                        reason_parts
                    ),
                )
            )

        results.sort(
            key=lambda item: (
                item.score,
                item.memory.importance,
                item.memory.confidence,
                item.memory.updated_at,
            ),
            reverse=True,
        )

        results = results[
            :normalized_limit
        ]

        if touch_results:
            for result in results:
                self.touch(
                    result.memory.id
                )

        return results

    # ========================================================================
    # EXACT SEARCH
    # ========================================================================

    def search_exact(
        self,
        content: str,
        *,
        namespace: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[MemoryRecord]:
        """
        Exact normalized content search.

        Duplicate IDs are impossible in the returned list.
        """
        normalized_content = normalize_content(
            content
        )

        if not normalized_content:
            return []

        normalized_namespace = (
            normalize_namespace(namespace)
            if namespace is not None
            else None
        )

        with self._lock:
            matches = [
                memory
                for memory in self._memories.values()
                if (
                    memory.content
                    == normalized_content
                    and memory.is_available
                    and (
                        normalized_namespace
                        is None
                        or memory.namespace
                        == normalized_namespace
                    )
                )
            ]

            matches.sort(
                key=lambda item: item.updated_at,
                reverse=True,
            )

            unique: dict[
                str,
                MemoryRecord,
            ] = {}

            for memory in matches:
                unique[
                    memory.id
                ] = memory

            result = list(
                unique.values()
            )

            return [
                memory.clone()
                for memory in result[
                    :normalize_limit(limit)
                ]
            ]

    # ========================================================================
    # TAG SEARCH
    # ========================================================================

    def search_tags(
        self,
        tags: Iterable[Any],
        *,
        namespace: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[MemoryRecord]:
        """
        Find memories containing requested tags.
        """
        requested = set(
            normalize_tags(tags)
        )

        if not requested:
            return []

        normalized_namespace = (
            normalize_namespace(namespace)
            if namespace is not None
            else None
        )

        with self._lock:
            results = []

            for memory in self._memories.values():
                if not memory.is_available:
                    continue

                if (
                    normalized_namespace is not None
                    and memory.namespace
                    != normalized_namespace
                ):
                    continue

                if not (
                    requested
                    & set(memory.tags)
                ):
                    continue

                results.append(
                    memory
                )

            results.sort(
                key=lambda item: (
                    len(
                        requested
                        & set(item.tags)
                    ),
                    item.importance,
                    item.updated_at,
                ),
                reverse=True,
            )

            return [
                item.clone()
                for item in results[
                    :normalize_limit(limit)
                ]
            ]

    # ========================================================================
    # CONTEXT
    # ========================================================================

    def build_context(
        self,
        query: str = "",
        *,
        namespace: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = DEFAULT_CONTEXT_LIMIT,
        max_chars: int = 12_000,
    ) -> str:
        """
        Build a prompt-friendly memory context.
        """
        normalized_limit = min(
            normalize_limit(limit),
            MAX_CONTEXT_LIMIT,
        )

        if normalize_text(query):
            results = self.search(
                query,
                namespace=namespace,
                session_id=session_id,
                limit=normalized_limit,
            )
        else:
            memories = self.all(
                namespace=namespace,
                session_id=session_id,
                limit=normalized_limit,
            )

            results = [
                MemorySearchResult(
                    memory=memory,
                    score=1.0,
                    matched_tokens=[],
                    reason="recent_memory",
                )
                for memory in memories
            ]

        if not results:
            return ""

        lines: list[str] = []

        lines.append(
            "=== ZAI MEMORY CONTEXT ==="
        )

        lines.append("")

        lines.append(
            "Relevant Memories:"
        )

        lines.append("")

        for index, result in enumerate(
            results,
            start=1,
        ):
            memory = result.memory

            label = (
                memory.key
                or memory.memory_type
                or "memory"
            )

            lines.append(
                f"{index}. "
                f"[{memory.namespace}] "
                f"{label}: "
                f"{memory.content}"
            )

        lines.append("")

        lines.append(
            "=== END MEMORY CONTEXT ==="
        )

        context = "\n".join(
            lines
        )

        if len(context) > max_chars:
            context = context[
                :max_chars
            ]

            context = (
                context.rstrip()
                + "\n=== END MEMORY CONTEXT ==="
            )

        return context

    # ========================================================================
    # MEMORY COMMAND DETECTION
    # ========================================================================

    def detect_memory_command(
        self,
        text: str,
    ) -> Optional[dict[str, Any]]:
        """
        Detect simple natural-language memory commands.

        Supported examples:

            ingat bahwa saya suka X
            ingat saya suka X
            lupakan X
            hapus memory X
            apa yang kamu ingat tentang X
        """
        normalized = normalize_text(
            text
        )

        if not normalized:
            return None

        lower = normalized.lower()

        save_prefixes = (
            "ingat bahwa ",
            "ingat ",
            "simpan bahwa ",
            "simpan ",
        )

        for prefix in save_prefixes:
            if lower.startswith(prefix):
                content = normalized[
                    len(prefix):
                ].strip()

                if content:
                    return {
                        "action": "save",
                        "content": content,
                    }

        forget_prefixes = (
            "lupakan ",
            "hapus memory ",
            "hapus memori ",
            "hapus ingatan ",
        )

        for prefix in forget_prefixes:
            if lower.startswith(prefix):
                content = normalized[
                    len(prefix):
                ].strip()

                if content:
                    return {
                        "action": "delete",
                        "query": content,
                    }

        search_prefixes = (
            "apa yang kamu ingat tentang ",
            "apa yang kamu ingat mengenai ",
            "ingatanku tentang ",
        )

        for prefix in search_prefixes:
            if lower.startswith(prefix):
                query = normalized[
                    len(prefix):
                ].strip()

                if query:
                    return {
                        "action": "search",
                        "query": query,
                    }

        return None

    # ========================================================================
    # MEMORY COMMAND EXECUTION
    # ========================================================================

    def execute_memory_command(
        self,
        command: Mapping[str, Any],
    ) -> Optional[Any]:
        """
        Execute a detected memory command.
        """
        action = normalize_text(
            command.get("action")
        ).lower()

        if action == "save":
            content = normalize_content(
                command.get(
                    "content",
                    "",
                )
            )

            if not content:
                return None

            return self.create(
                content,
                namespace=command.get(
                    "namespace",
                    self.default_namespace,
                ),
                memory_type=command.get(
                    "memory_type",
                    DEFAULT_MEMORY_TYPE,
                ),
                key=command.get(
                    "key",
                    "",
                ),
                tags=command.get(
                    "tags",
                    [],
                ),
                importance=command.get(
                    "importance",
                    DEFAULT_IMPORTANCE,
                ),
                confidence=command.get(
                    "confidence",
                    DEFAULT_CONFIDENCE,
                ),
                source=command.get(
                    "source",
                    "zai",
                ),
                session_id=command.get(
                    "session_id"
                ),
            )

        if action == "delete":
            query = normalize_text(
                command.get(
                    "query",
                    "",
                )
            )

            if not query:
                return False

            results = self.search(
                query,
                namespace=command.get(
                    "namespace"
                ),
                limit=1,
            )

            if not results:
                return False

            return self.delete(
                results[0].memory.id
            )

        if action == "search":
            query = normalize_text(
                command.get(
                    "query",
                    "",
                )
            )

            if not query:
                return []

            return self.search(
                query,
                namespace=command.get(
                    "namespace"
                ),
                limit=normalize_limit(
                    command.get(
                        "limit",
                        DEFAULT_SEARCH_LIMIT,
                    )
                ),
            )

        if action == "clear":
            return self.clear(
                namespace=command.get(
                    "namespace"
                )
            )

        return None

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Return detailed memory statistics.
        """
        with self._lock:
            memories = list(
                self._memories.values()
            )

            sessions = list(
                self._sessions.values()
            )

        active_memories = [
            item
            for item in memories
            if item.active
            and not item.is_expired
        ]

        expired_memories = [
            item
            for item in memories
            if item.is_expired
        ]

        memory_type_distribution: dict[
            str,
            int,
        ] = {}

        namespace_distribution: dict[
            str,
            int,
        ] = {}

        tag_distribution: dict[
            str,
            int,
        ] = {}

        for memory in memories:
            memory_type_distribution[
                memory.memory_type
            ] = (
                memory_type_distribution.get(
                    memory.memory_type,
                    0,
                )
                + 1
            )

            namespace_distribution[
                memory.namespace
            ] = (
                namespace_distribution.get(
                    memory.namespace,
                    0,
                )
                + 1
            )

            for tag in memory.tags:
                tag_distribution[
                    tag
                ] = (
                    tag_distribution.get(
                        tag,
                        0,
                    )
                    + 1
                )

        if memories:
            average_importance = sum(
                memory.importance
                for memory in memories
            ) / len(memories)

            average_confidence = sum(
                memory.confidence
                for memory in memories
            ) / len(memories)

        else:
            average_importance = 0.0
            average_confidence = 0.0

        return {
            "total_memories": len(
                memories
            ),
            "active_memories": len(
                active_memories
            ),
            "expired_memories": len(
                expired_memories
            ),
            "inactive_memories": (
                len(memories)
                - len(active_memories)
            ),
            "total_conversation_turns": sum(
                session.turn_count
                for session in sessions
            ),
            "session_count": len(
                sessions
            ),
            "active_session_count": sum(
                1
                for session in sessions
                if session.active
            ),
            "namespace_count": len(
                namespace_distribution
            ),
            "tag_count": len(
                tag_distribution
            ),
            "access_count": self.access_count,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "average_importance": round(
                average_importance,
                4,
            ),
            "average_confidence": round(
                average_confidence,
                4,
            ),
            "memory_type_distribution": (
                memory_type_distribution
            ),
            "namespace_distribution": (
                namespace_distribution
            ),
            "tag_distribution": (
                tag_distribution
            ),
            "storage_status": (
                self.storage_status
            ),
        }

    def stats(self) -> dict[str, Any]:
        """
        Compatibility alias.
        """
        return self.statistics()

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health(self) -> dict[str, Any]:
        """
        Health report.
        """
        statistics = self.statistics()

        success_total = (
            self.success_count
            + self.failure_count
        )

        success_rate = (
            (
                self.success_count
                / success_total
            )
            * 100.0
            if success_total
            else 0.0
        )

        return {
            "memory": "MemoryStore",
            "version": self.VERSION,
            "status": "HEALTHY",
            "storage_status": (
                self.storage_status
            ),
            "memory_count": (
                statistics[
                    "total_memories"
                ]
            ),
            "session_count": (
                statistics[
                    "session_count"
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
            "success_rate": round(
                success_rate,
                4,
            ),
        }

    def info(self) -> dict[str, Any]:
        """
        Information report.
        """
        return {
            "store": "MemoryStore",
            "version": self.VERSION,
            "status": "READY",
            "storage_status": (
                self.storage_status
            ),
            "memory_count": self.memory_count,
            "session_count": self.session_count,
        }

    # ========================================================================
    # ITERATION
    # ========================================================================

    def __iter__(
        self,
    ) -> Iterator[MemoryRecord]:
        """
        Iterate through available memories.
        """
        for memory in self.all():
            yield memory

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def snapshot(self) -> dict[str, Any]:
        """
        Export complete in-memory state.
        """
        with self._lock:
            return {
                "version": self.VERSION,
                "storage_enabled": (
                    self.storage_enabled
                ),
                "default_namespace": (
                    self.default_namespace
                ),
                "memories": [
                    memory.to_dict()
                    for memory
                    in self._memories.values()
                ],
                "sessions": [
                    session.to_dict()
                    for session
                    in self._sessions.values()
                ],
                "statistics": self.statistics(),
            }

    def restore(
        self,
        snapshot: Mapping[str, Any],
        *,
        clear_existing: bool = True,
    ) -> None:
        """
        Restore a snapshot.
        """
        if not isinstance(
            snapshot,
            Mapping,
        ):
            raise TypeError(
                "snapshot harus berupa mapping."
            )

        with self._lock:
            if clear_existing:
                self._memories.clear()
                self._sessions.clear()

            for raw_session in (
                snapshot.get(
                    "sessions",
                    [],
                )
                or []
            ):
                if not isinstance(
                    raw_session,
                    Mapping,
                ):
                    continue

                session = MemorySession(
                    id=raw_session.get(
                        "id"
                    )
                    or raw_session.get(
                        "session_id"
                    )
                    or str(uuid4()),
                    title=raw_session.get(
                        "title",
                        DEFAULT_SESSION_TITLE,
                    ),
                    namespace=raw_session.get(
                        "namespace",
                        DEFAULT_NAMESPACE,
                    ),
                    created_at=ensure_datetime(
                        raw_session.get(
                            "created_at"
                        )
                    ),
                    updated_at=ensure_datetime(
                        raw_session.get(
                            "updated_at"
                        )
                    ),
                    closed_at=ensure_datetime(
                        raw_session.get(
                            "closed_at"
                        )
                    ),
                    active=raw_session.get(
                        "active",
                        True,
                    ),
                    turn_count=raw_session.get(
                        "turn_count",
                        0,
                    ),
                    metadata=raw_session.get(
                        "metadata",
                        {},
                    ),
                )

                self._sessions[
                    session.id
                ] = session

            for raw_memory in (
                snapshot.get(
                    "memories",
                    [],
                )
                or []
            ):
                if not isinstance(
                    raw_memory,
                    Mapping,
                ):
                    continue

                memory = MemoryRecord(
                    id=raw_memory.get(
                        "id"
                    )
                    or raw_memory.get(
                        "memory_id"
                    )
                    or str(uuid4()),
                    content=raw_memory.get(
                        "content",
                        raw_memory.get(
                            "text",
                            "",
                        ),
                    ),
                    namespace=raw_memory.get(
                        "namespace",
                        DEFAULT_NAMESPACE,
                    ),
                    memory_type=raw_memory.get(
                        "memory_type",
                        DEFAULT_MEMORY_TYPE,
                    ),
                    key=raw_memory.get(
                        "key",
                        "",
                    ),
                    tags=raw_memory.get(
                        "tags",
                        [],
                    ),
                    importance=raw_memory.get(
                        "importance",
                        DEFAULT_IMPORTANCE,
                    ),
                    confidence=raw_memory.get(
                        "confidence",
                        DEFAULT_CONFIDENCE,
                    ),
                    source=raw_memory.get(
                        "source",
                        "zai",
                    ),
                    session_id=raw_memory.get(
                        "session_id"
                    ),
                    created_at=ensure_datetime(
                        raw_memory.get(
                            "created_at"
                        )
                    ),
                    updated_at=ensure_datetime(
                        raw_memory.get(
                            "updated_at"
                        )
                    ),
                    accessed_at=ensure_datetime(
                        raw_memory.get(
                            "accessed_at"
                        )
                    ),
                    expires_at=ensure_datetime(
                        raw_memory.get(
                            "expires_at"
                        )
                    ),
                    access_count=raw_memory.get(
                        "access_count",
                        0,
                    ),
                    active=raw_memory.get(
                        "active",
                        True,
                    ),
                    metadata=raw_memory.get(
                        "metadata",
                        {},
                    ),
                )

                self._memories[
                    memory.id
                ] = memory

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:
        """
        Completely reset runtime state.
        """
        with self._lock:
            self._memories.clear()
            self._sessions.clear()

            self.execution_count = 0
            self.success_count = 0
            self.failure_count = 0
            self.access_count = 0


# ============================================================================
# MEMORY MANAGER
# ============================================================================


class MemoryManager:
    """
    High-level compatibility facade.

    This class keeps the same general role used by the existing
    Super ZAI pipeline while delegating storage operations to
    MemoryStore.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        *,
        storage_enabled: bool = False,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        self.store = (
            store
            if store is not None
            else MemoryStore(
                storage_enabled=storage_enabled,
                namespace=namespace,
            )
        )

    # ========================================================================
    # MEMORY WRITE
    # ========================================================================

    def remember(
        self,
        content: str,
        **kwargs: Any,
    ) -> MemoryRecord:
        """
        Save memory.
        """
        return self.store.create(
            content,
            **kwargs,
        )

    def save(
        self,
        content: str,
        **kwargs: Any,
    ) -> MemoryRecord:
        """
        Compatibility alias.
        """
        return self.remember(
            content,
            **kwargs,
        )

    def add(
        self,
        content: str,
        **kwargs: Any,
    ) -> MemoryRecord:
        """
        Compatibility alias.
        """
        return self.remember(
            content,
            **kwargs,
        )

    # ========================================================================
    # MEMORY READ
    # ========================================================================

    def get(
        self,
        memory_id: str,
    ) -> Optional[MemoryRecord]:
        """
        Retrieve memory.
        """
        return self.store.get(
            memory_id
        )

    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """
        Search memory.
        """
        return self.store.search(
            query,
            **kwargs,
        )

    def retrieve(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """
        Compatibility alias.
        """
        return self.search(
            query,
            **kwargs,
        )

    def build_context(
        self,
        query: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Build LLM memory context.
        """
        return self.store.build_context(
            query,
            **kwargs,
        )

    # ========================================================================
    # MEMORY DELETE
    # ========================================================================

    def delete(
        self,
        memory_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Delete memory.
        """
        return self.store.delete(
            memory_id,
            **kwargs,
        )

    def forget(
        self,
        memory_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Compatibility alias.
        """
        return self.delete(
            memory_id,
            **kwargs,
        )

    # ========================================================================
    # SESSION
    # ========================================================================

    def create_session(
        self,
        **kwargs: Any,
    ) -> MemorySession:
        """
        Create memory session.
        """
        return self.store.create_session(
            **kwargs
        )

    def get_session(
        self,
        session_id: str,
    ) -> Optional[MemorySession]:
        """
        Get session.
        """
        return self.store.get_session(
            session_id
        )

    def close_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Close session.
        """
        return self.store.close_session(
            session_id
        )

    # ========================================================================
    # COMMANDS
    # ========================================================================

    def detect_memory_command(
        self,
        text: str,
    ) -> Optional[dict[str, Any]]:
        """
        Detect memory command.
        """
        return self.store.detect_memory_command(
            text
        )

    def execute_memory_command(
        self,
        command: Mapping[str, Any],
    ) -> Optional[Any]:
        """
        Execute memory command.
        """
        return self.store.execute_memory_command(
            command
        )

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Memory statistics.
        """
        return self.store.statistics()

    def stats(self) -> dict[str, Any]:
        """
        Compatibility alias.
        """
        return self.statistics()

    def health(self) -> dict[str, Any]:
        """
        Memory health.
        """
        health = self.store.health()

        return {
            "memory": "MemoryManager",
            "version": self.VERSION,
            "status": health.get(
                "status",
                "HEALTHY",
            ),
            "storage_status": health.get(
                "storage_status",
                "DISABLED",
            ),
            "memory_count": health.get(
                "memory_count",
                0,
            ),
            "session_count": health.get(
                "session_count",
                0,
            ),
            "execution_count": health.get(
                "execution_count",
                0,
            ),
            "success_count": health.get(
                "success_count",
                0,
            ),
            "failure_count": health.get(
                "failure_count",
                0,
            ),
            "success_rate": health.get(
                "success_rate",
                0.0,
            ),
        }

    def info(self) -> dict[str, Any]:
        """
        Information report.
        """
        return {
            "manager": "MemoryManager",
            "version": self.VERSION,
            "status": "READY",
            "store_version": (
                self.store.VERSION
            ),
            "storage_status": (
                self.store.storage_status
            ),
            "memory_count": (
                self.store.memory_count
            ),
            "session_count": (
                self.store.session_count
            ),
        }

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:
        """
        Reset memory manager.
        """
        self.store.reset()


# ============================================================================
# FACTORY
# ============================================================================


def create_memory_store(
    *,
    storage_enabled: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> MemoryStore:
    """
    Create a MemoryStore instance.
    """
    return MemoryStore(
        storage_enabled=storage_enabled,
        namespace=namespace,
    )


def create_memory_manager(
    *,
    storage_enabled: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> MemoryManager:
    """
    Create a MemoryManager instance.
    """
    return MemoryManager(
        storage_enabled=storage_enabled,
        namespace=namespace,
    )


# ============================================================================
# SELF TEST HELPERS
# ============================================================================


def _assert(
    condition: bool,
    message: str,
) -> None:
    """
    Internal deterministic assertion helper.
    """
    if not condition:
        raise AssertionError(
            message
        )


def self_test() -> dict[str, Any]:
    """
    Complete deterministic memory store self-test.

    Important:
        The test deliberately uses unique content and namespaces.
        This prevents accidental duplicate search results.

    Returns:
        A structured test report.
    """
    store = MemoryStore(
        storage_enabled=False
    )

    test_results: list[str] = []

    # ------------------------------------------------------------------------
    # TEST 1: EMPTY STORE
    # ------------------------------------------------------------------------

    _assert(
        store.memory_count == 0,
        "Initial memory count harus 0.",
    )

    _assert(
        store.session_count == 0,
        "Initial session count harus 0.",
    )

    test_results.append(
        "EMPTY_STORE_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 2: SESSION
    # ------------------------------------------------------------------------

    session = store.create_session(
        title="ZAI Self Test",
        namespace="test",
    )

    _assert(
        session.id,
        "Session ID tidak boleh kosong.",
    )

    _assert(
        store.session_count == 1,
        "Session count harus 1.",
    )

    test_results.append(
        "SESSION_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 3: CREATE MEMORY
    # ------------------------------------------------------------------------

    memory = store.create(
        "Super ZAI adalah project AI pribadi.",
        namespace="test",
        memory_type="fact",
        key="project",
        tags=[
            "zai",
            "project",
        ],
        importance=1.0,
        confidence=1.0,
        session_id=session.id,
    )

    _assert(
        isinstance(
            memory,
            MemoryRecord,
        ),
        "Create harus menghasilkan MemoryRecord.",
    )

    _assert(
        store.memory_count == 1,
        "Memory count harus 1.",
    )

    test_results.append(
        "CREATE_MEMORY_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 4: GET
    # ------------------------------------------------------------------------

    loaded = store.get(
        memory.id
    )

    _assert(
        loaded is not None,
        "Memory harus dapat diambil.",
    )

    _assert(
        loaded.content
        == memory.content,
        "Content hasil get berbeda.",
    )

    test_results.append(
        "GET_MEMORY_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 5: SEARCH
    # ------------------------------------------------------------------------

    results = store.search(
        "Super ZAI",
        namespace="test",
        limit=10,
    )

    _assert(
        len(results) == 1,
        (
            "Search harus menghasilkan "
            "tepat 1 memory pada self_test."
        ),
    )

    _assert(
        results[0].memory.id
        == memory.id,
        "Search result ID salah.",
    )

    _assert(
        results[0].score > 0,
        "Search score harus > 0.",
    )

    test_results.append(
        "SEARCH_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 6: EXACT SEARCH
    # ------------------------------------------------------------------------

    exact_results = store.search_exact(
        memory.content,
        namespace="test",
    )

    _assert(
        len(exact_results) == 1,
        "Exact search harus menghasilkan 1.",
    )

    test_results.append(
        "EXACT_SEARCH_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 7: TAG SEARCH
    # ------------------------------------------------------------------------

    tag_results = store.search_tags(
        ["zai"],
        namespace="test",
    )

    _assert(
        len(tag_results) == 1,
        "Tag search harus menghasilkan 1.",
    )

    test_results.append(
        "TAG_SEARCH_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 8: TOUCH
    # ------------------------------------------------------------------------

    before_access = (
        store.get(
            memory.id
        ).access_count
    )

    store.touch(
        memory.id
    )

    after_access = (
        store.get(
            memory.id
        ).access_count
    )

    _assert(
        after_access
        == before_access + 1,
        "Access count tidak bertambah.",
    )

    test_results.append(
        "TOUCH_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 9: UPDATE
    # ------------------------------------------------------------------------

    updated = store.update(
        memory.id,
        content=(
            "Super ZAI adalah "
            "platform AI pribadi "
            "multi-agent."
        ),
        tags=[
            "zai",
            "ai",
            "project",
        ],
    )

    _assert(
        updated is not None,
        "Update harus menghasilkan memory.",
    )

    _assert(
        "multi-agent"
        in updated.content,
        "Update content gagal.",
    )

    test_results.append(
        "UPDATE_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 10: UPSERT
    # ------------------------------------------------------------------------

    upserted = store.upsert(
        "Super ZAI adalah platform AI "
        "multi-agent yang berkembang.",
        key="project",
        namespace="test",
        tags=[
            "zai",
            "project",
        ],
    )

    _assert(
        upserted.id
        == memory.id,
        "Upsert key seharusnya mempertahankan ID.",
    )

    _assert(
        store.memory_count == 1,
        "Upsert tidak boleh membuat duplicate.",
    )

    test_results.append(
        "UPSERT_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 11: CONTEXT
    # ------------------------------------------------------------------------

    context = store.build_context(
        "Super ZAI",
        namespace="test",
    )

    _assert(
        "ZAI MEMORY CONTEXT"
        in context,
        "Context header tidak ditemukan.",
    )

    _assert(
        "Super ZAI"
        in context,
        "Memory tidak masuk context.",
    )

    test_results.append(
        "CONTEXT_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 12: COMMAND DETECTION
    # ------------------------------------------------------------------------

    command = store.detect_memory_command(
        "ingat bahwa ZAI sedang dibangun"
    )

    _assert(
        command is not None,
        "Save memory command tidak terdeteksi.",
    )

    _assert(
        command.get("action")
        == "save",
        "Command action harus save.",
    )

    test_results.append(
        "COMMAND_DETECTION_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 13: COMMAND EXECUTION
    # ------------------------------------------------------------------------

    command_result = (
        store.execute_memory_command(
            {
                "action": "save",
                "content": (
                    "ZAI menggunakan "
                    "arsitektur multi-agent."
                ),
                "namespace": "test",
                "tags": [
                    "zai",
                    "architecture",
                ],
            }
        )
    )

    _assert(
        isinstance(
            command_result,
            MemoryRecord,
        ),
        "Command execution save gagal.",
    )

    _assert(
        store.memory_count == 2,
        "Command save seharusnya menambah memory.",
    )

    test_results.append(
        "COMMAND_EXECUTION_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 14: DELETE
    # ------------------------------------------------------------------------

    delete_target = store.search_exact(
        "ZAI menggunakan arsitektur "
        "multi-agent.",
        namespace="test",
    )

    _assert(
        len(delete_target) == 1,
        "Delete target tidak ditemukan.",
    )

    deleted = store.delete(
        delete_target[0].id
    )

    _assert(
        deleted is True,
        "Delete harus True.",
    )

    _assert(
        store.get(
            delete_target[0].id
        ).active is False,
        "Soft delete harus membuat active=False.",
    )

    test_results.append(
        "DELETE_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 15: EXPIRATION
    # ------------------------------------------------------------------------

    expired = store.create(
        "Temporary ZAI memory.",
        namespace="expiration_test",
        ttl_seconds=0.001,
    )

    # Force expiration for deterministic test.
    with store._lock:
        internal = store._memories[
            expired.id
        ]

        internal.expires_at = (
            utc_now()
            - timedelta(
                seconds=1
            )
        )

        store._memories[
            expired.id
        ] = internal

    _assert(
        expired.id
        in store._memories,
        "Expired memory tidak tersimpan.",
    )

    purged = store.purge_expired(
        hard=True
    )

    _assert(
        purged >= 1,
        "Expired memory tidak berhasil dipurge.",
    )

    test_results.append(
        "EXPIRATION_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 16: SNAPSHOT
    # ------------------------------------------------------------------------

    snapshot = store.snapshot()

    _assert(
        isinstance(
            snapshot,
            dict,
        ),
        "Snapshot harus dictionary.",
    )

    _assert(
        "memories"
        in snapshot,
        "Snapshot tidak memiliki memories.",
    )

    _assert(
        "sessions"
        in snapshot,
        "Snapshot tidak memiliki sessions.",
    )

    test_results.append(
        "SNAPSHOT_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 17: RESTORE
    # ------------------------------------------------------------------------

    restored_store = MemoryStore()

    restored_store.restore(
        snapshot
    )

    _assert(
        restored_store.session_count
        == store.session_count,
        "Session restore gagal.",
    )

    test_results.append(
        "RESTORE_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 18: MANAGER
    # ------------------------------------------------------------------------

    manager = MemoryManager()

    manager_memory = manager.remember(
        "Memory Manager ZAI test.",
        namespace="manager_test",
    )

    _assert(
        manager_memory.content
        == "Memory Manager ZAI test.",
        "MemoryManager remember gagal.",
    )

    manager_results = manager.search(
        "Memory Manager",
        namespace="manager_test",
    )

    _assert(
        len(manager_results) == 1,
        "MemoryManager search gagal.",
    )

    test_results.append(
        "MEMORY_MANAGER_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 19: HEALTH
    # ------------------------------------------------------------------------

    health = manager.health()

    _assert(
        health["status"]
        == "HEALTHY",
        "MemoryManager health gagal.",
    )

    test_results.append(
        "HEALTH_OK"
    )

    # ------------------------------------------------------------------------
    # TEST 20: STATISTICS
    # ------------------------------------------------------------------------

    statistics = manager.statistics()

    _assert(
        statistics["total_memories"]
        == 1,
        "Manager statistics memory count salah.",
    )

    _assert(
        statistics["session_count"]
        == 0,
        "Manager statistics session count salah.",
    )

    test_results.append(
        "STATISTICS_OK"
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    final_statistics = store.statistics()

    return {
        "success": True,
        "store": "MemoryStore",
        "version": MEMORY_STORE_VERSION,
        "status": "PASS",
        "tests_passed": len(
            test_results
        ),
        "tests": test_results,
        "memory_count": (
            store.memory_count
        ),
        "session_count": (
            store.session_count
        ),
        "statistics": final_statistics,
    }


# ============================================================================
# MODULE LEVEL COMPATIBILITY HELPERS
# ============================================================================


def save_memory(
    content: str,
    *,
    store: Optional[MemoryStore] = None,
    **kwargs: Any,
) -> MemoryRecord:
    """
    Convenience helper.
    """
    target = (
        store
        if store is not None
        else MemoryStore()
    )

    return target.create(
        content,
        **kwargs,
    )


def search_memory(
    query: str,
    *,
    store: Optional[MemoryStore] = None,
    **kwargs: Any,
) -> list[MemorySearchResult]:
    """
    Convenience helper.
    """
    target = (
        store
        if store is not None
        else MemoryStore()
    )

    return target.search(
        query,
        **kwargs,
    )


def build_memory_context(
    query: str = "",
    *,
    store: Optional[MemoryStore] = None,
    **kwargs: Any,
) -> str:
    """
    Convenience helper.
    """
    target = (
        store
        if store is not None
        else MemoryStore()
    )

    return target.build_context(
        query,
        **kwargs,
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "MEMORY_STORE_VERSION",
    "DEFAULT_NAMESPACE",
    "DEFAULT_MEMORY_TYPE",
    "MemoryRecord",
    "MemoryEntry",
    "MemorySession",
    "MemorySearchResult",
    "MemoryStore",
    "MemoryManager",
    "create_memory_store",
    "create_memory_manager",
    "save_memory",
    "search_memory",
    "build_memory_context",
    "self_test",
]


# ============================================================================
# DIRECT EXECUTION
# ============================================================================


if __name__ == "__main__":
    import pprint

    print("=" * 72)
    print(" ZAI MEMORY STORE SELF TEST")
    print("=" * 72)

    try:
        result = self_test()

        pprint.pp(
            result
        )

        print("=" * 72)
        print("MEMORY_STORE_OK")
        print("=" * 72)

    except Exception as exc:
        print("=" * 72)
        print("MEMORY_STORE_FAILED")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 72)

        raise
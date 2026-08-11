from __future__ import annotations

"""
ZAI Memory Brain Bridge
=======================

Layer integrasi antara:

    ZAIBrain
        |
        v
    MemoryBrainBridge
        |
        +---- MemoryManager
        |
        +---- MemoryStore
        |
        +---- MemoryRecord
        |
        +---- MemorySession
        |
        +---- MemorySearchResult

Tujuan utama modul ini:

1. Menyediakan interface stabil antara Brain dan Memory.
2. Mengisolasi Brain dari detail implementasi MemoryStore.
3. Mengambil memory yang relevan sebelum Brain memproses task.
4. Menyusun memory context yang aman untuk LLM.
5. Menyimpan interaksi setelah task selesai.
6. Menjaga session conversation.
7. Menyediakan memory statistics.
8. Menyediakan health check.
9. Menyediakan fallback ketika API memory berubah.
10. Menjaga backward compatibility dengan MemoryManager lama.
11. Mencegah memory kosong / invalid merusak Brain.
12. Menyediakan sanitasi context.
13. Membatasi ukuran context.
14. Menyediakan observability.
15. Menjadi fondasi semantic/vector memory di masa depan.

Design goals:

- dependency injection
- duck typing untuk compatibility
- defensive programming
- no hidden global state
- deterministic behavior
- safe context generation
- easy unit testing
- production-oriented
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4


# ============================================================================
# CONSTANTS
# ============================================================================

BRIDGE_VERSION = "1.0.0"

DEFAULT_NAMESPACE = "default"

DEFAULT_SESSION_ID = "default"

DEFAULT_MAX_RESULTS = 8

DEFAULT_MAX_CONTEXT_CHARS = 6000

DEFAULT_MAX_MEMORY_CHARS = 1200

DEFAULT_MIN_RELEVANCE = 0.0

DEFAULT_MIN_IMPORTANCE = 0.0

DEFAULT_CONFIDENCE = 1.0

DEFAULT_MEMORY_TYPE = "conversation"

DEFAULT_ROLE_USER = "user"

DEFAULT_ROLE_ASSISTANT = "assistant"

DEFAULT_ROLE_SYSTEM = "system"

DEFAULT_ROLE_TOOL = "tool"

MAX_SESSION_ID_LENGTH = 200

MAX_NAMESPACE_LENGTH = 100

MAX_QUERY_LENGTH = 4000

MAX_CONTENT_LENGTH = 20000

MAX_KEY_LENGTH = 200

MAX_TAG_LENGTH = 100

MAX_TAGS = 20

SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
    "private_key",
    "credit_card",
    "card_number",
    "cvv",
    "pin",
}


# ============================================================================
# TIME HELPERS
# ============================================================================


def utc_now() -> datetime:
    """
    Return timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    """
    Return current UTC timestamp as ISO string.
    """
    return utc_now().isoformat()


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================


def normalize_text(
    value: Any,
    *,
    max_length: int,
) -> str:
    """
    Normalize arbitrary input into bounded text.
    """
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\x00", "")

    text = text.strip()

    if len(text) > max_length:
        text = text[:max_length].rstrip() + "..."

    return text


def normalize_session_id(
    value: Any,
) -> str:
    """
    Normalize session identifier.
    """
    result = normalize_text(
        value,
        max_length=MAX_SESSION_ID_LENGTH,
    )

    return result or DEFAULT_SESSION_ID


def normalize_namespace(
    value: Any,
) -> str:
    """
    Normalize memory namespace.
    """
    result = normalize_text(
        value,
        max_length=MAX_NAMESPACE_LENGTH,
    )

    return result or DEFAULT_NAMESPACE


def normalize_key(
    value: Any,
) -> str:
    """
    Normalize memory key.
    """
    result = normalize_text(
        value,
        max_length=MAX_KEY_LENGTH,
    )

    return result.lower()


def normalize_role(
    value: Any,
) -> str:
    """
    Normalize conversation role.
    """
    role = normalize_text(
        value,
        max_length=50,
    ).lower()

    if role in {
        DEFAULT_ROLE_USER,
        DEFAULT_ROLE_ASSISTANT,
        DEFAULT_ROLE_SYSTEM,
        DEFAULT_ROLE_TOOL,
    }:
        return role

    return DEFAULT_ROLE_USER


def normalize_tags(
    values: Any,
) -> list[str]:
    """
    Normalize memory tags.
    """
    if values is None:
        return []

    if isinstance(values, str):
        values = [
            item.strip()
            for item in values.split(",")
        ]

    if not isinstance(values, Iterable):
        return []

    result: list[str] = []

    for value in values:
        tag = normalize_text(
            value,
            max_length=MAX_TAG_LENGTH,
        ).lower()

        if not tag:
            continue

        if tag not in result:
            result.append(tag)

        if len(result) >= MAX_TAGS:
            break

    return result


# ============================================================================
# SAFE SERIALIZATION
# ============================================================================


def safe_value(
    value: Any,
    *,
    key: str = "",
    depth: int = 0,
) -> Any:
    """
    Convert arbitrary values into safe JSON-like structures.

    Sensitive keys are masked.
    """

    if depth > 6:
        return "<max-depth>"

    normalized_key = str(key or "").lower()

    if any(
        sensitive in normalized_key
        for sensitive in SENSITIVE_KEYS
    ):
        return "[REDACTED]"

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

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for item_key, item_value in value.items():
            result[str(item_key)] = safe_value(
                item_value,
                key=str(item_key),
                depth=depth + 1,
            )

        return result

    if isinstance(value, (list, tuple, set)):
        return [
            safe_value(
                item,
                key=normalized_key,
                depth=depth + 1,
            )
            for item in value
        ]

    if hasattr(value, "to_dict"):
        try:
            return safe_value(
                value.to_dict(),
                key=normalized_key,
                depth=depth + 1,
            )
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            return safe_value(
                value.model_dump(),
                key=normalized_key,
                depth=depth + 1,
            )
        except Exception:
            pass

    return str(value)


# ============================================================================
# DTO: MEMORY CONTEXT ITEM
# ============================================================================


@dataclass(slots=True)
class MemoryContextItem:
    """
    Normalized memory item exposed to Brain.
    """

    memory_id: str

    content: str

    memory_type: str = DEFAULT_MEMORY_TYPE

    namespace: str = DEFAULT_NAMESPACE

    key: str = ""

    relevance: float = DEFAULT_MIN_RELEVANCE

    importance: float = DEFAULT_MIN_IMPORTANCE

    confidence: float = DEFAULT_CONFIDENCE

    tags: list[str] = field(
        default_factory=list,
    )

    created_at: Optional[str] = None

    updated_at: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.memory_id = normalize_text(
            self.memory_id,
            max_length=200,
        )

        self.content = normalize_text(
            self.content,
            max_length=DEFAULT_MAX_MEMORY_CHARS,
        )

        self.memory_type = normalize_text(
            self.memory_type,
            max_length=100,
        ) or DEFAULT_MEMORY_TYPE

        self.namespace = normalize_namespace(
            self.namespace,
        )

        self.key = normalize_key(
            self.key,
        )

        self.relevance = clamp_score(
            self.relevance,
        )

        self.importance = clamp_score(
            self.importance,
        )

        self.confidence = clamp_score(
            self.confidence,
        )

        self.tags = normalize_tags(
            self.tags,
        )

        self.metadata = safe_value(
            self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize context item.
        """
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "namespace": self.namespace,
            "key": self.key,
            "relevance": self.relevance,
            "importance": self.importance,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


# ============================================================================
# DTO: MEMORY CONTEXT
# ============================================================================


@dataclass(slots=True)
class MemoryContext:
    """
    Aggregated memory context for Brain/LLM.
    """

    query: str

    session_id: str

    namespace: str

    items: list[MemoryContextItem] = field(
        default_factory=list,
    )

    text: str = ""

    total_candidates: int = 0

    selected_count: int = 0

    truncated: bool = False

    created_at: str = field(
        default_factory=utc_iso,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    @property
    def has_memory(self) -> bool:
        """
        Return True when at least one memory exists.
        """
        return bool(self.items)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize complete context.
        """
        return {
            "query": self.query,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "items": [
                item.to_dict()
                for item in self.items
            ],
            "text": self.text,
            "total_candidates": self.total_candidates,
            "selected_count": self.selected_count,
            "truncated": self.truncated,
            "created_at": self.created_at,
            "metadata": safe_value(
                self.metadata,
            ),
        }


# ============================================================================
# DTO: MEMORY OPERATION RESULT
# ============================================================================


@dataclass(slots=True)
class MemoryBridgeResult:
    """
    Standard result returned by MemoryBrainBridge.
    """

    success: bool

    operation: str

    message: str = ""

    data: Any = None

    error: Optional[str] = None

    execution_id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    created_at: str = field(
        default_factory=utc_iso,
    )

    latency_ms: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    warnings: list[str] = field(
        default_factory=list,
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize operation result.
        """
        return {
            "success": self.success,
            "operation": self.operation,
            "message": self.message,
            "data": safe_value(self.data),
            "error": self.error,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "latency_ms": self.latency_ms,
            "metadata": safe_value(
                self.metadata,
            ),
            "warnings": list(self.warnings),
        }


# ============================================================================
# SCORE HELPERS
# ============================================================================


def clamp_score(
    value: Any,
) -> float:
    """
    Clamp numeric score into 0..1.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if number < 0:
        return 0.0

    if number > 1:
        return 1.0

    return number


def calculate_memory_score(
    *,
    relevance: float,
    importance: float,
    confidence: float,
) -> float:
    """
    Calculate ranking score.

    Relevance receives the highest weight because the Brain
    should prioritize memories useful for the current task.
    """
    relevance = clamp_score(relevance)

    importance = clamp_score(importance)

    confidence = clamp_score(confidence)

    score = (
        relevance * 0.55
        + importance * 0.25
        + confidence * 0.20
    )

    return round(
        clamp_score(score),
        6,
    )


# ============================================================================
# MEMORY OBJECT EXTRACTION
# ============================================================================


def object_to_mapping(
    value: Any,
) -> dict[str, Any]:
    """
    Convert memory object into dictionary-like structure.
    """

    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "to_dict"):
        try:
            result = value.to_dict()

            if isinstance(result, Mapping):
                return dict(result)
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            result = value.model_dump()

            if isinstance(result, Mapping):
                return dict(result)
        except Exception:
            pass

    fields = (
        "id",
        "memory_id",
        "key",
        "content",
        "value",
        "text",
        "memory_type",
        "type",
        "namespace",
        "importance",
        "confidence",
        "relevance",
        "score",
        "tags",
        "created_at",
        "updated_at",
        "metadata",
    )

    result: dict[str, Any] = {}

    for field_name in fields:
        try:
            field_value = getattr(
                value,
                field_name,
            )
        except Exception:
            continue

        if field_value is not None:
            result[field_name] = field_value

    return result


def extract_memory_content(
    value: Any,
) -> str:
    """
    Extract human-readable content from memory object.
    """

    data = object_to_mapping(value)

    for key in (
        "content",
        "text",
        "value",
        "memory",
        "summary",
        "description",
    ):
        candidate = data.get(key)

        if candidate is None:
            continue

        text = normalize_text(
            candidate,
            max_length=DEFAULT_MAX_MEMORY_CHARS,
        )

        if text:
            return text

    return ""


def extract_memory_id(
    value: Any,
) -> str:
    """
    Extract memory identifier.
    """

    data = object_to_mapping(value)

    for key in (
        "memory_id",
        "id",
        "uuid",
        "execution_id",
    ):
        candidate = data.get(key)

        if candidate:
            return str(candidate)

    return str(uuid4())


def extract_memory_type(
    value: Any,
) -> str:
    """
    Extract memory type.
    """

    data = object_to_mapping(value)

    result = (
        data.get("memory_type")
        or data.get("type")
        or DEFAULT_MEMORY_TYPE
    )

    return normalize_text(
        result,
        max_length=100,
    )


def extract_memory_score(
    value: Any,
    field_name: str,
) -> float:
    """
    Extract score from arbitrary memory object.
    """

    data = object_to_mapping(value)

    return clamp_score(
        data.get(
            field_name,
            0.0,
        )
    )


# ============================================================================
# MEMORY BRAIN BRIDGE
# ============================================================================


class MemoryBrainBridge:
    """
    Production-oriented compatibility layer between Brain and Memory.

    The bridge intentionally does not require a specific MemoryStore
    implementation. This allows MemoryStore and MemoryManager to evolve
    without requiring ZAIBrain to be rewritten.

    Supported provider styles:

        manager.search(...)
        manager.recall(...)
        manager.retrieve(...)
        manager.query(...)
        store.search(...)
        store.retrieve(...)
        store.add(...)
        store.remember(...)
        store.save(...)
        manager.remember(...)

    The bridge attempts known APIs in a controlled order.
    """

    VERSION = BRIDGE_VERSION

    def __init__(
        self,
        memory_manager: Any = None,
        memory_store: Any = None,
        *,
        default_namespace: str = DEFAULT_NAMESPACE,
        default_session_id: str = DEFAULT_SESSION_ID,
        max_results: int = DEFAULT_MAX_RESULTS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        min_importance: float = DEFAULT_MIN_IMPORTANCE,
    ) -> None:
        self.memory_manager = memory_manager

        self.memory_store = memory_store

        self.default_namespace = normalize_namespace(
            default_namespace,
        )

        self.default_session_id = normalize_session_id(
            default_session_id,
        )

        self.max_results = max(
            1,
            int(max_results),
        )

        self.max_context_chars = max(
            500,
            int(max_context_chars),
        )

        self.min_relevance = clamp_score(
            min_relevance,
        )

        self.min_importance = clamp_score(
            min_importance,
        )

        self.execution_count = 0

        self.success_count = 0

        self.failure_count = 0

        self.retrieval_count = 0

        self.save_count = 0

        self.context_count = 0

        self.session_count = 0

        self.last_operation: Optional[str] = None

        self.last_error: Optional[str] = None

    # ========================================================================
    # PROVIDER
    # ========================================================================

    def provider(self) -> Any:
        """
        Return the preferred memory provider.
        """

        if self.memory_manager is not None:
            return self.memory_manager

        if self.memory_store is not None:
            return self.memory_store

        return None

    def is_available(self) -> bool:
        """
        Return whether a memory provider is available.
        """
        return self.provider() is not None

    # ========================================================================
    # INFO
    # ========================================================================

    def info(self) -> dict[str, Any]:
        """
        Return bridge information.
        """

        return {
            "bridge": "MemoryBrainBridge",
            "version": self.VERSION,
            "status": (
                "READY"
                if self.is_available()
                else "DEGRADED"
            ),
            "provider": (
                type(self.provider()).__name__
                if self.provider() is not None
                else None
            ),
            "default_namespace": self.default_namespace,
            "default_session_id": self.default_session_id,
            "max_results": self.max_results,
            "max_context_chars": self.max_context_chars,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "retrieval_count": self.retrieval_count,
            "save_count": self.save_count,
            "context_count": self.context_count,
        }

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health(self) -> dict[str, Any]:
        """
        Return health state.
        """

        provider = self.provider()

        provider_health: Any = None

        if provider is not None:
            health_method = getattr(
                provider,
                "health",
                None,
            )

            if callable(health_method):
                try:
                    provider_health = health_method()
                except Exception as exc:
                    provider_health = {
                        "status": "ERROR",
                        "error": str(exc),
                    }

        status = "HEALTHY"

        if provider is None:
            status = "DEGRADED"

        elif isinstance(
            provider_health,
            Mapping,
        ):
            provider_status = str(
                provider_health.get(
                    "status",
                    "READY",
                )
            ).upper()

            if provider_status in {
                "ERROR",
                "FAILED",
                "UNHEALTHY",
            }:
                status = "DEGRADED"

        return {
            "bridge": "MemoryBrainBridge",
            "version": self.VERSION,
            "status": status,
            "provider_available": provider is not None,
            "provider_health": provider_health,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
        }

    # ========================================================================
    # SUCCESS RATE
    # ========================================================================

    def success_rate(self) -> float:
        """
        Calculate bridge success rate.
        """

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

    # ========================================================================
    # FAILURE RATE
    # ========================================================================

    def failure_rate(self) -> float:
        """
        Calculate bridge failure rate.
        """

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

    # ========================================================================
    # SESSION
    # ========================================================================

    def resolve_session_id(
        self,
        session_id: Optional[str],
    ) -> str:
        """
        Resolve effective session id.
        """

        return normalize_session_id(
            session_id
            or self.default_session_id,
        )

    # ========================================================================
    # NAMESPACE
    # ========================================================================

    def resolve_namespace(
        self,
        namespace: Optional[str],
    ) -> str:
        """
        Resolve effective namespace.
        """

        return normalize_namespace(
            namespace
            or self.default_namespace,
        )

    # ========================================================================
    # SEARCH
    # ========================================================================

    def search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: Optional[int] = None,
        min_relevance: Optional[float] = None,
        min_importance: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryBridgeResult:
        """
        Retrieve relevant memories.

        This is intentionally synchronous because it adapts the
        current MemoryManager/MemoryStore API. If the underlying
        provider exposes an async method, use async_search().
        """

        started = utc_now()

        self.execution_count += 1

        self.retrieval_count += 1

        self.last_operation = "search"

        normalized_query = normalize_text(
            query,
            max_length=MAX_QUERY_LENGTH,
        )

        effective_session = self.resolve_session_id(
            session_id,
        )

        effective_namespace = self.resolve_namespace(
            namespace,
        )

        effective_limit = (
            self.max_results
            if limit is None
            else max(1, int(limit))
        )

        effective_relevance = (
            self.min_relevance
            if min_relevance is None
            else clamp_score(min_relevance)
        )

        effective_importance = (
            self.min_importance
            if min_importance is None
            else clamp_score(min_importance)
        )

        if not normalized_query:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="search",
                message="Query memory kosong.",
                error="Memory query tidak boleh kosong.",
                started=started,
            )

        provider = self.provider()

        if provider is None:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="search",
                message="Memory provider belum tersedia.",
                error="Memory provider tidak dikonfigurasi.",
                started=started,
            )

        try:
            raw_results = self._provider_search(
                provider,
                normalized_query,
                session_id=effective_session,
                namespace=effective_namespace,
                limit=effective_limit,
                metadata=metadata,
            )

            normalized_results = self._normalize_results(
                raw_results,
                min_relevance=effective_relevance,
                min_importance=effective_importance,
                limit=effective_limit,
            )

            self.success_count += 1

            return self._result(
                success=True,
                operation="search",
                message=(
                    "Memory berhasil diambil."
                ),
                data=normalized_results,
                started=started,
                metadata={
                    "query": normalized_query,
                    "session_id": effective_session,
                    "namespace": effective_namespace,
                    "result_count": len(
                        normalized_results,
                    ),
                },
            )

        except Exception as exc:
            self.failure_count += 1

            self.last_error = str(exc)

            return self._result(
                success=False,
                operation="search",
                message="Memory retrieval gagal.",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                started=started,
            )

    # ========================================================================
    # ASYNC SEARCH
    # ========================================================================

    async def async_search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: Optional[int] = None,
        min_relevance: Optional[float] = None,
        min_importance: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryBridgeResult:
        """
        Async version of memory search.

        Uses async provider methods when available.
        Falls back to sync search when necessary.
        """

        started = utc_now()

        provider = self.provider()

        if provider is None:
            return self.search(
                query,
                session_id=session_id,
                namespace=namespace,
                limit=limit,
                min_relevance=min_relevance,
                min_importance=min_importance,
                metadata=metadata,
            )

        method = self._find_callable(
            provider,
            (
                "async_search",
                "asearch",
                "async_recall",
                "arecall",
                "async_retrieve",
                "aretrieve",
            ),
        )

        if method is None:
            return self.search(
                query,
                session_id=session_id,
                namespace=namespace,
                limit=limit,
                min_relevance=min_relevance,
                min_importance=min_importance,
                metadata=metadata,
            )

        self.execution_count += 1

        self.retrieval_count += 1

        self.last_operation = "async_search"

        normalized_query = normalize_text(
            query,
            max_length=MAX_QUERY_LENGTH,
        )

        if not normalized_query:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="async_search",
                message="Query memory kosong.",
                error="Memory query tidak boleh kosong.",
                started=started,
            )

        effective_session = self.resolve_session_id(
            session_id,
        )

        effective_namespace = self.resolve_namespace(
            namespace,
        )

        effective_limit = (
            self.max_results
            if limit is None
            else max(1, int(limit))
        )

        try:
            raw_results = await self._call_provider_async(
                method,
                normalized_query,
                session_id=effective_session,
                namespace=effective_namespace,
                limit=effective_limit,
                metadata=metadata,
            )

            normalized_results = self._normalize_results(
                raw_results,
                min_relevance=(
                    self.min_relevance
                    if min_relevance is None
                    else clamp_score(min_relevance)
                ),
                min_importance=(
                    self.min_importance
                    if min_importance is None
                    else clamp_score(min_importance)
                ),
                limit=effective_limit,
            )

            self.success_count += 1

            return self._result(
                success=True,
                operation="async_search",
                message="Memory berhasil diambil.",
                data=normalized_results,
                started=started,
            )

        except Exception as exc:
            self.failure_count += 1

            self.last_error = str(exc)

            return self._result(
                success=False,
                operation="async_search",
                message="Async memory retrieval gagal.",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                started=started,
            )

    # ========================================================================
    # PROVIDER SEARCH
    # ========================================================================

    def _provider_search(
        self,
        provider: Any,
        query: str,
        *,
        session_id: str,
        namespace: str,
        limit: int,
        metadata: Optional[Mapping[str, Any]],
    ) -> Any:
        """
        Call compatible synchronous provider API.
        """

        method = self._find_callable(
            provider,
            (
                "search",
                "retrieve",
                "recall",
                "query",
                "find",
            ),
        )

        if method is None:
            raise AttributeError(
                "Memory provider tidak memiliki "
                "method search/retrieve/recall/query/find."
            )

        kwargs = {
            "query": query,
            "session_id": session_id,
            "namespace": namespace,
            "limit": limit,
            "metadata": dict(
                metadata or {},
            ),
        }

        return self._call_provider_sync(
            method,
            query,
            kwargs,
        )

    # ========================================================================
    # CALL SYNC PROVIDER
    # ========================================================================

    @staticmethod
    def _call_provider_sync(
        method: Any,
        query: str,
        kwargs: Mapping[str, Any],
    ) -> Any:
        """
        Call provider while tolerating older signatures.
        """

        attempts = [
            lambda: method(**kwargs),
            lambda: method(
                query,
                limit=kwargs["limit"],
            ),
            lambda: method(query),
        ]

        last_error: Optional[Exception] = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Memory provider gagal dipanggil."
        )

    # ========================================================================
    # CALL ASYNC PROVIDER
    # ========================================================================

    @staticmethod
    async def _call_provider_async(
        method: Any,
        query: str,
        *,
        session_id: str,
        namespace: str,
        limit: int,
        metadata: Optional[Mapping[str, Any]],
    ) -> Any:
        """
        Call async provider while tolerating older signatures.
        """

        kwargs = {
            "query": query,
            "session_id": session_id,
            "namespace": namespace,
            "limit": limit,
            "metadata": dict(
                metadata or {},
            ),
        }

        attempts = [
            lambda: method(**kwargs),
            lambda: method(
                query,
                limit=limit,
            ),
            lambda: method(query),
        ]

        last_error: Optional[Exception] = None

        for attempt in attempts:
            try:
                result = attempt()

                if hasattr(
                    result,
                    "__await__",
                ):
                    return await result

                return result

            except TypeError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Async memory provider gagal dipanggil."
        )

    # ========================================================================
    # NORMALIZE SEARCH RESULTS
    # ========================================================================

    def _normalize_results(
        self,
        raw_results: Any,
        *,
        min_relevance: float,
        min_importance: float,
        limit: int,
    ) -> list[MemoryContextItem]:
        """
        Convert provider-specific result objects into MemoryContextItem.
        """

        if raw_results is None:
            return []

        if isinstance(
            raw_results,
            Mapping,
        ):
            for key in (
                "results",
                "items",
                "memories",
                "data",
            ):
                if key in raw_results:
                    raw_results = raw_results[key]
                    break
            else:
                raw_results = [
                    raw_results,
                ]

        if isinstance(
            raw_results,
            (str, bytes),
        ):
            raw_results = [
                raw_results,
            ]

        try:
            iterator = iter(
                raw_results,
            )
        except TypeError:
            iterator = iter(
                [
                    raw_results,
                ]
            )

        normalized: list[
            MemoryContextItem
        ] = []

        for raw_item in iterator:
            item = self._normalize_memory_item(
                raw_item,
            )

            if not item.content:
                continue

            if item.relevance < min_relevance:
                continue

            if item.importance < min_importance:
                continue

            normalized.append(item)

        normalized.sort(
            key=lambda item: calculate_memory_score(
                relevance=item.relevance,
                importance=item.importance,
                confidence=item.confidence,
            ),
            reverse=True,
        )

        return normalized[:limit]

    # ========================================================================
    # NORMALIZE MEMORY ITEM
    # ========================================================================

    def _normalize_memory_item(
        self,
        raw_item: Any,
    ) -> MemoryContextItem:
        """
        Normalize one provider result.
        """

        data = object_to_mapping(
            raw_item,
        )

        memory_id = extract_memory_id(
            raw_item,
        )

        content = extract_memory_content(
            raw_item,
        )

        memory_type = extract_memory_type(
            raw_item,
        )

        namespace = normalize_namespace(
            data.get(
                "namespace",
                self.default_namespace,
            )
        )

        key = normalize_key(
            data.get(
                "key",
                "",
            )
        )

        relevance = extract_memory_score(
            raw_item,
            "relevance",
        )

        if relevance == 0:
            relevance = clamp_score(
                data.get(
                    "score",
                    0.0,
                )
            )

        importance = extract_memory_score(
            raw_item,
            "importance",
        )

        confidence = extract_memory_score(
            raw_item,
            "confidence",
        )

        if confidence == 0:
            confidence = DEFAULT_CONFIDENCE

        tags = normalize_tags(
            data.get(
                "tags",
                [],
            )
        )

        metadata = data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            Mapping,
        ):
            metadata = {}

        return MemoryContextItem(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            namespace=namespace,
            key=key,
            relevance=relevance,
            importance=importance,
            confidence=confidence,
            tags=tags,
            created_at=self._optional_string(
                data.get("created_at"),
            ),
            updated_at=self._optional_string(
                data.get("updated_at"),
            ),
            metadata=dict(metadata),
        )

    # ========================================================================
    # OPTIONAL STRING
    # ========================================================================

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> Optional[str]:
        """
        Normalize optional string values.
        """

        if value is None:
            return None

        return str(value)

    # ========================================================================
    # CONTEXT BUILD
    # ========================================================================

    def build_context(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: Optional[int] = None,
        max_chars: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryBridgeResult:
        """
        Retrieve memories and construct a bounded LLM context.
        """

        started = utc_now()

        self.context_count += 1

        result = self.search(
            query,
            session_id=session_id,
            namespace=namespace,
            limit=limit,
            metadata=metadata,
        )

        if not result.success:
            return self._result(
                success=False,
                operation="build_context",
                message="Memory context gagal dibuat.",
                error=result.error,
                started=started,
                warnings=result.warnings,
            )

        items = result.data or []

        effective_max_chars = (
            self.max_context_chars
            if max_chars is None
            else max(
                500,
                int(max_chars),
            )
        )

        context_text, selected, truncated = (
            self._render_context(
                items,
                max_chars=effective_max_chars,
            )
        )

        context = MemoryContext(
            query=normalize_text(
                query,
                max_length=MAX_QUERY_LENGTH,
            ),
            session_id=self.resolve_session_id(
                session_id,
            ),
            namespace=self.resolve_namespace(
                namespace,
            ),
            items=selected,
            text=context_text,
            total_candidates=len(items),
            selected_count=len(selected),
            truncated=truncated,
            metadata={
                "bridge_version": self.VERSION,
                "max_chars": effective_max_chars,
            },
        )

        self.success_count += 1

        return self._result(
            success=True,
            operation="build_context",
            message=(
                "Memory context berhasil dibuat."
            ),
            data=context,
            started=started,
            metadata={
                "memory_count": len(selected),
                "truncated": truncated,
            },
        )

    # ========================================================================
    # RENDER CONTEXT
    # ========================================================================

    def _render_context(
        self,
        items: list[MemoryContextItem],
        *,
        max_chars: int,
    ) -> tuple[
        str,
        list[MemoryContextItem],
        bool,
    ]:
        """
        Render bounded memory context.
        """

        if not items:
            return (
                "",
                [],
                False,
            )

        lines = [
            "===== ZAI MEMORY CONTEXT =====",
        ]

        selected: list[
            MemoryContextItem
        ] = []

        current_length = len(
            lines[0],
        )

        truncated = False

        for index, item in enumerate(
            items,
            start=1,
        ):
            line = (
                f"{index}. "
                f"[{item.memory_type}] "
                f"{item.content}"
            )

            if item.key:
                line = (
                    f"{index}. "
                    f"[{item.key}] "
                    f"{item.content}"
                )

            projected = (
                current_length
                + len(line)
                + 1
            )

            if projected > max_chars:
                truncated = True
                break

            lines.append(line)

            current_length = projected

            selected.append(item)

        if not selected:
            truncated = True

            first = items[0]

            content = normalize_text(
                first.content,
                max_length=max(
                    100,
                    max_chars - 100,
                ),
            )

            lines.append(
                f"1. {content}"
            )

            selected.append(first)

        lines.append(
            "===== END MEMORY CONTEXT ====="
        )

        result = "\n".join(
            lines,
        )

        if len(result) > max_chars:
            result = result[:max_chars].rstrip()

            truncated = True

        return (
            result,
            selected,
            truncated,
        )

    # ========================================================================
    # SAVE MEMORY
    # ========================================================================

    def save(
        self,
        content: str,
        *,
        key: Optional[str] = None,
        memory_type: str = DEFAULT_MEMORY_TYPE,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        importance: float = DEFAULT_MIN_IMPORTANCE,
        confidence: float = DEFAULT_CONFIDENCE,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryBridgeResult:
        """
        Save one memory through the available provider.
        """

        started = utc_now()

        self.execution_count += 1

        self.save_count += 1

        self.last_operation = "save"

        normalized_content = normalize_text(
            content,
            max_length=MAX_CONTENT_LENGTH,
        )

        if not normalized_content:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="save",
                message="Memory kosong tidak disimpan.",
                error="Memory content tidak boleh kosong.",
                started=started,
            )

        provider = self.provider()

        if provider is None:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="save",
                message="Memory provider belum tersedia.",
                error="Memory provider tidak dikonfigurasi.",
                started=started,
            )

        effective_session = self.resolve_session_id(
            session_id,
        )

        effective_namespace = self.resolve_namespace(
            namespace,
        )

        normalized_key = normalize_key(
            key,
        )

        normalized_tags = normalize_tags(
            tags,
        )

        clean_metadata = safe_value(
            metadata or {},
        )

        payload = {
            "content": normalized_content,
            "key": normalized_key,
            "memory_type": normalize_text(
                memory_type,
                max_length=100,
            ) or DEFAULT_MEMORY_TYPE,
            "session_id": effective_session,
            "namespace": effective_namespace,
            "importance": clamp_score(
                importance,
            ),
            "confidence": clamp_score(
                confidence,
            ),
            "tags": normalized_tags,
            "metadata": clean_metadata,
        }

        try:
            raw_result = self._provider_save(
                provider,
                payload,
            )

            self.success_count += 1

            return self._result(
                success=True,
                operation="save",
                message="Memory berhasil disimpan.",
                data=raw_result,
                started=started,
                metadata={
                    "session_id": effective_session,
                    "namespace": effective_namespace,
                    "memory_type": payload[
                        "memory_type"
                    ],
                },
            )

        except Exception as exc:
            self.failure_count += 1

            self.last_error = str(exc)

            return self._result(
                success=False,
                operation="save",
                message="Memory gagal disimpan.",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                started=started,
            )

    # ========================================================================
    # PROVIDER SAVE
    # ========================================================================

    def _provider_save(
        self,
        provider: Any,
        payload: Mapping[str, Any],
    ) -> Any:
        """
        Call compatible save API.
        """

        method = self._find_callable(
            provider,
            (
                "remember",
                "save",
                "add_memory",
                "store",
                "add",
                "create",
            ),
        )

        if method is None:
            raise AttributeError(
                "Memory provider tidak memiliki "
                "method remember/save/add_memory/store/add/create."
            )

        attempts = [
            lambda: method(**payload),
            lambda: method(
                payload["content"],
                key=payload["key"],
                namespace=payload["namespace"],
            ),
            lambda: method(
                payload["content"],
            ),
        ]

        last_error: Optional[Exception] = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Memory save gagal."
        )

    # ========================================================================
    # SAVE INTERACTION
    # ========================================================================

    def save_interaction(
        self,
        *,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryBridgeResult:
        """
        Save a conversation turn as memory.

        The default policy uses a conversation memory type and
        moderate importance.
        """

        started = utc_now()

        user_text = normalize_text(
            user_message,
            max_length=MAX_CONTENT_LENGTH,
        )

        assistant_text = normalize_text(
            assistant_response,
            max_length=MAX_CONTENT_LENGTH,
        )

        if not user_text and not assistant_text:
            return self._result(
                success=False,
                operation="save_interaction",
                message="Interaction kosong.",
                error=(
                    "User message dan assistant response "
                    "sama-sama kosong."
                ),
                started=started,
            )

        interaction = (
            f"User: {user_text}\n"
            f"ZAI: {assistant_text}"
        )

        result = self.save(
            interaction,
            memory_type="conversation",
            session_id=session_id,
            namespace=namespace,
            importance=0.35,
            confidence=1.0,
            tags=[
                "conversation",
                "interaction",
            ],
            metadata={
                **dict(metadata or {}),
                "source": "brain",
            },
        )

        if result.success:
            result.operation = (
                "save_interaction"
            )

        return result

    # ========================================================================
    # CREATE SESSION
    # ========================================================================

    def create_session(
        self,
        *,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryBridgeResult:
        """
        Create or initialize a memory session.

        If provider supports explicit session creation it is used.
        Otherwise the bridge maintains the logical identifier.
        """

        started = utc_now()

        self.execution_count += 1

        self.session_count += 1

        effective_session = self.resolve_session_id(
            session_id,
        )

        effective_namespace = self.resolve_namespace(
            namespace,
        )

        provider = self.provider()

        if provider is not None:
            method = self._find_callable(
                provider,
                (
                    "create_session",
                    "start_session",
                    "open_session",
                    "get_or_create_session",
                ),
            )

            if method is not None:
                try:
                    payload = {
                        "session_id": effective_session,
                        "namespace": effective_namespace,
                        "metadata": dict(
                            metadata or {},
                        ),
                    }

                    try:
                        raw = method(
                            **payload,
                        )
                    except TypeError:
                        raw = method(
                            effective_session,
                        )

                    self.success_count += 1

                    return self._result(
                        success=True,
                        operation="create_session",
                        message="Session memory siap.",
                        data=raw,
                        started=started,
                    )

                except Exception as exc:
                    self.failure_count += 1

                    return self._result(
                        success=False,
                        operation="create_session",
                        message=(
                            "Session provider gagal."
                        ),
                        error=(
                            f"{type(exc).__name__}: {exc}"
                        ),
                        started=started,
                    )

        self.success_count += 1

        return self._result(
            success=True,
            operation="create_session",
            message="Logical memory session siap.",
            data={
                "session_id": effective_session,
                "namespace": effective_namespace,
                "metadata": safe_value(
                    metadata or {},
                ),
            },
            started=started,
        )

    # ========================================================================
    # CLEAR SESSION
    # ========================================================================

    def clear_session(
        self,
        session_id: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
    ) -> MemoryBridgeResult:
        """
        Clear session memory if provider supports it.
        """

        started = utc_now()

        self.execution_count += 1

        provider = self.provider()

        effective_session = self.resolve_session_id(
            session_id,
        )

        effective_namespace = self.resolve_namespace(
            namespace,
        )

        if provider is None:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="clear_session",
                message="Memory provider tidak tersedia.",
                error="Memory provider tidak dikonfigurasi.",
                started=started,
            )

        method = self._find_callable(
            provider,
            (
                "clear_session",
                "delete_session",
                "reset_session",
            ),
        )

        if method is None:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="clear_session",
                message="Provider belum mendukung clear session.",
                error=(
                    "Method clear_session/delete_session/reset_session "
                    "tidak tersedia."
                ),
                started=started,
            )

        try:
            try:
                data = method(
                    session_id=effective_session,
                    namespace=effective_namespace,
                )
            except TypeError:
                data = method(
                    effective_session,
                )

            self.success_count += 1

            return self._result(
                success=True,
                operation="clear_session",
                message="Session memory berhasil dibersihkan.",
                data=data,
                started=started,
            )

        except Exception as exc:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="clear_session",
                message="Session gagal dibersihkan.",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                started=started,
            )

    # ========================================================================
    # DELETE MEMORY
    # ========================================================================

    def delete(
        self,
        memory_id: str,
        *,
        namespace: Optional[str] = None,
    ) -> MemoryBridgeResult:
        """
        Delete one memory.
        """

        started = utc_now()

        self.execution_count += 1

        normalized_id = normalize_text(
            memory_id,
            max_length=200,
        )

        if not normalized_id:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="delete",
                message="Memory ID kosong.",
                error="memory_id wajib diisi.",
                started=started,
            )

        provider = self.provider()

        if provider is None:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="delete",
                message="Memory provider tidak tersedia.",
                error="Memory provider tidak dikonfigurasi.",
                started=started,
            )

        method = self._find_callable(
            provider,
            (
                "delete",
                "delete_memory",
                "remove",
                "forget",
            ),
        )

        if method is None:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="delete",
                message="Provider belum mendukung delete.",
                error=(
                    "Method delete/delete_memory/remove/forget "
                    "tidak tersedia."
                ),
                started=started,
            )

        try:
            try:
                data = method(
                    normalized_id,
                    namespace=self.resolve_namespace(
                        namespace,
                    ),
                )
            except TypeError:
                data = method(
                    normalized_id,
                )

            self.success_count += 1

            return self._result(
                success=True,
                operation="delete",
                message="Memory berhasil dihapus.",
                data=data,
                started=started,
            )

        except Exception as exc:
            self.failure_count += 1

            return self._result(
                success=False,
                operation="delete",
                message="Memory gagal dihapus.",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                started=started,
            )

    # ========================================================================
    # MEMORY STATISTICS
    # ========================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Return bridge and provider statistics.
        """

        provider = self.provider()

        provider_stats: Any = None

        if provider is not None:
            method = self._find_callable(
                provider,
                (
                    "statistics",
                    "stats",
                    "summary",
                ),
            )

            if method is not None:
                try:
                    provider_stats = method()
                except Exception as exc:
                    provider_stats = {
                        "status": "ERROR",
                        "error": str(exc),
                    }

        return {
            "bridge": "MemoryBrainBridge",
            "version": self.VERSION,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
            "failure_rate": self.failure_rate(),
            "retrieval_count": self.retrieval_count,
            "save_count": self.save_count,
            "context_count": self.context_count,
            "session_count": self.session_count,
            "provider_statistics": safe_value(
                provider_stats,
            ),
        }

    # ========================================================================
    # PROVIDER METHODS
    # ========================================================================

    @staticmethod
    def _find_callable(
        provider: Any,
        names: Iterable[str],
    ) -> Optional[Any]:
        """
        Find first callable method from provider.
        """

        for name in names:
            method = getattr(
                provider,
                name,
                None,
            )

            if callable(method):
                return method

        return None

    # ========================================================================
    # RESULT FACTORY
    # ========================================================================

    def _result(
        self,
        *,
        success: bool,
        operation: str,
        message: str,
        started: datetime,
        data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        warnings: Optional[list[str]] = None,
    ) -> MemoryBridgeResult:
        """
        Build standardized bridge result.
        """

        elapsed = (
            utc_now()
            - started
        ).total_seconds() * 1000.0

        return MemoryBridgeResult(
            success=success,
            operation=operation,
            message=message,
            data=data,
            error=error,
            latency_ms=round(
                elapsed,
                4,
            ),
            metadata={
                "bridge_version": self.VERSION,
                **dict(
                    metadata or {},
                ),
            },
            warnings=list(
                warnings or [],
            ),
        )


# ============================================================================
# FACTORY
# ============================================================================


_default_bridge: Optional[
    MemoryBrainBridge
] = None


def get_memory_brain_bridge(
    memory_manager: Any = None,
    memory_store: Any = None,
) -> MemoryBrainBridge:
    """
    Return singleton MemoryBrainBridge.

    Explicit dependencies replace the existing singleton.
    This allows tests and application startup to inject dependencies.
    """

    global _default_bridge

    if (
        memory_manager is not None
        or memory_store is not None
    ):
        _default_bridge = MemoryBrainBridge(
            memory_manager=memory_manager,
            memory_store=memory_store,
        )

        return _default_bridge

    if _default_bridge is None:
        _default_bridge = MemoryBrainBridge()

    return _default_bridge


def reset_memory_brain_bridge() -> None:
    """
    Reset singleton bridge.

    Primarily intended for tests and controlled application restart.
    """

    global _default_bridge

    _default_bridge = None


# ============================================================================
# MEMORY CONTEXT SHORTCUT
# ============================================================================


def build_memory_context(
    query: str,
    *,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = DEFAULT_MAX_RESULTS,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    memory_manager: Any = None,
    memory_store: Any = None,
) -> MemoryBridgeResult:
    """
    Convenience wrapper for Brain.
    """

    bridge = get_memory_brain_bridge(
        memory_manager=memory_manager,
        memory_store=memory_store,
    )

    return bridge.build_context(
        query,
        session_id=session_id,
        namespace=namespace,
        limit=limit,
        max_chars=max_chars,
    )


# ============================================================================
# MEMORY SAVE SHORTCUT
# ============================================================================


def remember_for_brain(
    content: str,
    *,
    key: Optional[str] = None,
    memory_type: str = DEFAULT_MEMORY_TYPE,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    importance: float = 0.0,
    confidence: float = 1.0,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    memory_manager: Any = None,
    memory_store: Any = None,
) -> MemoryBridgeResult:
    """
    Convenience wrapper for saving memory.
    """

    bridge = get_memory_brain_bridge(
        memory_manager=memory_manager,
        memory_store=memory_store,
    )

    return bridge.save(
        content,
        key=key,
        memory_type=memory_type,
        session_id=session_id,
        namespace=namespace,
        importance=importance,
        confidence=confidence,
        tags=tags,
        metadata=metadata,
    )


# ============================================================================
# SELF TEST
# ============================================================================


class _SelfTestMemoryProvider:
    """
    Small in-memory provider used only by self_test().
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def remember(
        self,
        content: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        item = {
            "memory_id": str(uuid4()),
            "content": content,
            "memory_type": kwargs.get(
                "memory_type",
                "conversation",
            ),
            "namespace": kwargs.get(
                "namespace",
                "default",
            ),
            "session_id": kwargs.get(
                "session_id",
                "default",
            ),
            "key": kwargs.get(
                "key",
                "",
            ),
            "importance": kwargs.get(
                "importance",
                0.0,
            ),
            "confidence": kwargs.get(
                "confidence",
                1.0,
            ),
            "relevance": 1.0,
            "tags": kwargs.get(
                "tags",
                [],
            ),
            "metadata": kwargs.get(
                "metadata",
                {},
            ),
            "created_at": utc_iso(),
        }

        self.items.append(item)

        return item

    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        normalized = query.lower()

        results = []

        for item in self.items:
            content = item["content"].lower()

            if normalized in content:
                results.append(item)
                continue

            words = normalized.split()

            if any(
                word in content
                for word in words
                if word
            ):
                results.append(item)

        return results[
            : kwargs.get(
                "limit",
                8,
            )
        ]

    def delete(
        self,
        memory_id: str,
        **_: Any,
    ) -> bool:
        before = len(
            self.items,
        )

        self.items = [
            item
            for item in self.items
            if item["memory_id"] != memory_id
        ]

        return len(
            self.items,
        ) < before

    def statistics(self) -> dict[str, Any]:
        return {
            "memory_count": len(
                self.items,
            ),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "HEALTHY",
        }


def self_test() -> dict[str, Any]:
    """
    Execute deterministic MemoryBrainBridge self test.
    """

    provider = _SelfTestMemoryProvider()

    bridge = MemoryBrainBridge(
        memory_manager=provider,
        max_results=5,
        max_context_chars=2000,
    )

    save_result = bridge.save(
        "Project Super ZAI menggunakan arsitektur multi agent.",
        key="project",
        memory_type="fact",
        importance=0.9,
        confidence=1.0,
        tags=[
            "zai",
            "project",
        ],
    )

    assert save_result.success is True

    interaction_result = bridge.save_interaction(
        user_message="Lanjut pembangunan ZAI",
        assistant_response="Siap, kita lanjut.",
    )

    assert interaction_result.success is True

    search_result = bridge.search(
        "Super ZAI",
    )

    assert search_result.success is True

    assert len(
        search_result.data,
    ) >= 1

    context_result = bridge.build_context(
        "ZAI project",
    )

    assert context_result.success is True

    assert isinstance(
        context_result.data,
        MemoryContext,
    )

    assert context_result.data.has_memory is True

    assert (
        "MEMORY CONTEXT"
        in context_result.data.text
    )

    statistics = bridge.statistics()

    assert statistics[
        "execution_count"
    ] > 0

    health = bridge.health()

    assert health[
        "status"
    ] == "HEALTHY"

    memory_id = (
        search_result.data[0].memory_id
    )

    delete_result = bridge.delete(
        memory_id,
    )

    assert delete_result.success is True

    return {
        "bridge": "MemoryBrainBridge",
        "version": BRIDGE_VERSION,
        "status": "PASS",
        "save_success": save_result.success,
        "interaction_success": (
            interaction_result.success
        ),
        "search_success": search_result.success,
        "context_success": context_result.success,
        "delete_success": delete_result.success,
        "statistics": statistics,
        "health": health,
    }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "BRIDGE_VERSION",
    "MemoryContextItem",
    "MemoryContext",
    "MemoryBridgeResult",
    "MemoryBrainBridge",
    "get_memory_brain_bridge",
    "reset_memory_brain_bridge",
    "build_memory_context",
    "remember_for_brain",
    "self_test",
]
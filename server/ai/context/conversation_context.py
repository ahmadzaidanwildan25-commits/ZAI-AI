from __future__ import annotations

"""
ZAI Conversation Context Engine
================================

File:
    ai/context/conversation_context.py

Purpose:
    Menyediakan context engine untuk percakapan ZAI.

Responsibilities:
    - Menyimpan conversation/session context.
    - Menyimpan message history.
    - Menyimpan metadata percakapan.
    - Menyimpan active topic.
    - Menyimpan active agent.
    - Menyimpan active intent.
    - Menyimpan active task.
    - Menyimpan context variables.
    - Menyimpan conversation summary.
    - Mencari message berdasarkan keyword.
    - Membatasi history.
    - Serialisasi/deserialisasi.
    - Snapshot context.
    - Restore context.
    - Menghapus context.
    - Statistik conversation.
    - Context validation.
    - Thread-safe mutation.
    - Export/import dictionary.
    - Self-test.

Design goals:
    1. Tidak bergantung pada module ZAI lain.
    2. Aman di-import oleh brain/manager/memory layer.
    3. API sederhana.
    4. Production-oriented.
    5. Backward-friendly.
    6. Mudah dikembangkan ke persistent storage.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4


# ============================================================================
# CONSTANTS
# ============================================================================

CONVERSATION_CONTEXT_VERSION = "1.0.0"

DEFAULT_MAX_MESSAGES = 100

MAX_ALLOWED_MESSAGES = 10_000

DEFAULT_NAMESPACE = "default"

DEFAULT_ROLE = "user"

VALID_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
    "agent",
    "developer",
}

VALID_CONTEXT_STATUSES = {
    "active",
    "paused",
    "completed",
    "archived",
    "cleared",
}

DEFAULT_CONTEXT_STATUS = "active"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def utc_now() -> datetime:
    """
    Menghasilkan waktu UTC aware.
    """
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    """
    Menghasilkan timestamp UTC ISO-8601.
    """
    return utc_now().isoformat()


def generate_id(prefix: str) -> str:
    """
    Membuat identifier unik.

    Example:
        generate_id("msg")
        -> msg-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    """
    return f"{prefix}-{uuid4()}"


def normalize_text(value: Any) -> str:
    """
    Normalisasi input menjadi string.

    None:
        menjadi ""

    String:
        whitespace awal/akhir dihapus.

    Non-string:
        dikonversi dengan str().
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def normalize_namespace(value: Any) -> str:
    """
    Normalisasi namespace context.
    """
    value = normalize_text(value)

    if not value:
        return DEFAULT_NAMESPACE

    return value


def normalize_role(value: Any) -> str:
    """
    Normalisasi role message.
    """
    role = normalize_text(value).lower()

    if not role:
        return DEFAULT_ROLE

    return role


def safe_copy_mapping(
    value: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Membuat shallow copy mapping menjadi dict.

    Digunakan agar caller tidak memodifikasi internal state
    secara tidak sengaja.
    """
    if value is None:
        return {}

    return dict(value)


# ============================================================================
# CONVERSATION MESSAGE
# ============================================================================


@dataclass
class ConversationMessage:
    """
    Representasi satu pesan percakapan.
    """

    role: str

    content: str

    message_id: str = field(
        default_factory=lambda: generate_id("msg")
    )

    timestamp: str = field(
        default_factory=utc_iso
    )

    name: Optional[str] = None

    agent: Optional[str] = None

    tool: Optional[str] = None

    intent: Optional[str] = None

    task_id: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    sequence: int = 0

    def __post_init__(self) -> None:
        self.role = normalize_role(self.role)

        self.content = normalize_text(self.content)

        if self.role not in VALID_ROLES:
            self.role = "user"

        self.name = (
            normalize_text(self.name)
            if self.name is not None
            else None
        )

        self.agent = (
            normalize_text(self.agent)
            if self.agent is not None
            else None
        )

        self.tool = (
            normalize_text(self.tool)
            if self.tool is not None
            else None
        )

        self.intent = (
            normalize_text(self.intent)
            if self.intent is not None
            else None
        )

        self.task_id = (
            normalize_text(self.task_id)
            if self.task_id is not None
            else None
        )

        self.metadata = dict(self.metadata or {})

        try:
            self.sequence = int(self.sequence)
        except (TypeError, ValueError):
            self.sequence = 0

    @property
    def length(self) -> int:
        """
        Panjang content.
        """
        return len(self.content)

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role == "assistant"

    @property
    def is_system(self) -> bool:
        return self.role == "system"

    @property
    def is_tool(self) -> bool:
        return self.role == "tool"

    @property
    def is_agent(self) -> bool:
        return self.role == "agent"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisasi message.
        """
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "name": self.name,
            "agent": self.agent,
            "tool": self.tool,
            "intent": self.intent,
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ConversationMessage":
        """
        Membuat ConversationMessage dari dictionary.

        Raises:
            TypeError:
                Jika data bukan mapping.
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                "ConversationMessage membutuhkan Mapping."
            )

        return cls(
            role=data.get("role", DEFAULT_ROLE),
            content=data.get("content", ""),
            message_id=data.get(
                "message_id",
                generate_id("msg"),
            ),
            timestamp=data.get(
                "timestamp",
                utc_iso(),
            ),
            name=data.get("name"),
            agent=data.get("agent"),
            tool=data.get("tool"),
            intent=data.get("intent"),
            task_id=data.get("task_id"),
            metadata=safe_copy_mapping(
                data.get("metadata")
            ),
            sequence=data.get("sequence", 0),
        )


# ============================================================================
# CONTEXT SNAPSHOT
# ============================================================================


@dataclass
class ConversationContextSnapshot:
    """
    Snapshot immutable-style dari context.

    Digunakan untuk:
        - debugging
        - rollback
        - checkpoint
        - orchestration
        - testing
    """

    context_id: str

    session_id: str

    namespace: str

    status: str

    active_topic: Optional[str]

    active_intent: Optional[str]

    active_agent: Optional[str]

    active_task: Optional[str]

    message_count: int

    context_variables: dict[str, Any]

    summary: str

    created_at: str = field(
        default_factory=utc_iso
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "status": self.status,
            "active_topic": self.active_topic,
            "active_intent": self.active_intent,
            "active_agent": self.active_agent,
            "active_task": self.active_task,
            "message_count": self.message_count,
            "context_variables": dict(
                self.context_variables
            ),
            "summary": self.summary,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# CONVERSATION CONTEXT
# ============================================================================


class ConversationContext:
    """
    Core conversation context untuk ZAI.

    Object ini mewakili satu konteks percakapan.

    Contoh:

        context = ConversationContext()

        context.add_user_message(
            "Halo ZAI"
        )

        context.add_assistant_message(
            "Halo, saya siap membantu."
        )

        print(context.history())
    """

    VERSION = CONVERSATION_CONTEXT_VERSION

    def __init__(
        self,
        context_id: Optional[str] = None,
        session_id: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.context_id = (
            normalize_text(context_id)
            or generate_id("ctx")
        )

        self.session_id = (
            normalize_text(session_id)
            or generate_id("session")
        )

        self.namespace = normalize_namespace(
            namespace
        )

        self.max_messages = self._normalize_max_messages(
            max_messages
        )

        self.status = DEFAULT_CONTEXT_STATUS

        self.created_at = utc_iso()

        self.updated_at = self.created_at

        self.active_topic: Optional[str] = None

        self.active_intent: Optional[str] = None

        self.active_agent: Optional[str] = None

        self.active_task: Optional[str] = None

        self.summary = ""

        self.metadata: dict[str, Any] = dict(
            metadata or {}
        )

        self.variables: dict[str, Any] = {}

        self.tags: set[str] = set()

        self._messages: list[
            ConversationMessage
        ] = []

        self._snapshots: list[
            ConversationContextSnapshot
        ] = []

        self._lock = RLock()

        self._operation_count = 0

    # ------------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------------

    @staticmethod
    def _normalize_max_messages(
        value: Any,
    ) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = DEFAULT_MAX_MESSAGES

        if value <= 0:
            value = DEFAULT_MAX_MESSAGES

        return min(
            value,
            MAX_ALLOWED_MESSAGES,
        )

    def _touch(self) -> None:
        self.updated_at = utc_iso()

        self._operation_count += 1

    def _trim_history_if_needed(self) -> None:
        """
        Menjaga jumlah message sesuai max_messages.
        """
        if len(self._messages) <= self.max_messages:
            return

        overflow = (
            len(self._messages)
            - self.max_messages
        )

        del self._messages[:overflow]

        self._renumber_sequences()

    def _renumber_sequences(self) -> None:
        for index, message in enumerate(
            self._messages,
            start=1,
        ):
            message.sequence = index

    def _find_message_index(
        self,
        message_id: str,
    ) -> int:
        for index, message in enumerate(
            self._messages
        ):
            if message.message_id == message_id:
                return index

        return -1

    # ------------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        with self._lock:
            return len(self._messages)

    @property
    def user_message_count(self) -> int:
        with self._lock:
            return sum(
                1
                for message in self._messages
                if message.is_user
            )

    @property
    def assistant_message_count(self) -> int:
        with self._lock:
            return sum(
                1
                for message in self._messages
                if message.is_assistant
            )

    @property
    def tool_message_count(self) -> int:
        with self._lock:
            return sum(
                1
                for message in self._messages
                if message.is_tool
            )

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_empty(self) -> bool:
        return self.message_count == 0

    @property
    def last_message(self) -> Optional[
        ConversationMessage
    ]:
        with self._lock:
            if not self._messages:
                return None

            return self._messages[-1]

    @property
    def first_message(self) -> Optional[
        ConversationMessage
    ]:
        with self._lock:
            if not self._messages:
                return None

            return self._messages[0]

    # ------------------------------------------------------------------------
    # MESSAGE MANAGEMENT
    # ------------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        *,
        name: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        intent: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> ConversationMessage:
        """
        Menambahkan message ke context.
        """
        with self._lock:
            message = ConversationMessage(
                role=role,
                content=content,
                name=name,
                agent=agent,
                tool=tool,
                intent=intent,
                task_id=task_id,
                metadata=safe_copy_mapping(
                    metadata
                ),
                sequence=len(
                    self._messages
                ) + 1,
            )

            self._messages.append(message)

            self._trim_history_if_needed()

            self._touch()

            return message

    def add_user_message(
        self,
        content: str,
        *,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
        intent: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> ConversationMessage:
        return self.add_message(
            "user",
            content,
            metadata=metadata,
            intent=intent,
            task_id=task_id,
        )

    def add_assistant_message(
        self,
        content: str,
        *,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
        agent: Optional[str] = None,
        intent: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> ConversationMessage:
        return self.add_message(
            "assistant",
            content,
            metadata=metadata,
            agent=agent,
            intent=intent,
            task_id=task_id,
        )

    def add_system_message(
        self,
        content: str,
        *,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> ConversationMessage:
        return self.add_message(
            "system",
            content,
            metadata=metadata,
        )

    def add_tool_message(
        self,
        content: str,
        *,
        tool: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
        task_id: Optional[str] = None,
    ) -> ConversationMessage:
        return self.add_message(
            "tool",
            content,
            tool=tool,
            metadata=metadata,
            task_id=task_id,
        )

    def add_agent_message(
        self,
        content: str,
        *,
        agent: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
        task_id: Optional[str] = None,
    ) -> ConversationMessage:
        return self.add_message(
            "agent",
            content,
            agent=agent,
            metadata=metadata,
            task_id=task_id,
        )

    def add_messages(
        self,
        messages: Iterable[
            Mapping[str, Any] | ConversationMessage
        ],
    ) -> list[ConversationMessage]:
        """
        Menambahkan banyak message sekaligus.
        """
        results: list[
            ConversationMessage
        ] = []

        for item in messages:
            if isinstance(
                item,
                ConversationMessage,
            ):
                message = self.add_message(
                    role=item.role,
                    content=item.content,
                    name=item.name,
                    agent=item.agent,
                    tool=item.tool,
                    intent=item.intent,
                    task_id=item.task_id,
                    metadata=item.metadata,
                )

            elif isinstance(item, Mapping):
                message = self.add_message(
                    role=item.get(
                        "role",
                        DEFAULT_ROLE,
                    ),
                    content=item.get(
                        "content",
                        "",
                    ),
                    name=item.get("name"),
                    agent=item.get("agent"),
                    tool=item.get("tool"),
                    intent=item.get("intent"),
                    task_id=item.get("task_id"),
                    metadata=item.get(
                        "metadata"
                    ),
                )

            else:
                raise TypeError(
                    "Message harus Mapping atau "
                    "ConversationMessage."
                )

            results.append(message)

        return results

    # ------------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------------

    def history(
        self,
        limit: Optional[int] = None,
        *,
        role: Optional[str] = None,
    ) -> list[ConversationMessage]:
        """
        Mengambil history.

        Return berupa copy list sehingga caller tidak
        dapat merusak internal list secara langsung.
        """
        with self._lock:
            messages = list(self._messages)

            if role is not None:
                normalized_role = normalize_role(
                    role
                )

                messages = [
                    message
                    for message in messages
                    if message.role
                    == normalized_role
                ]

            if limit is not None:
                try:
                    limit = int(limit)
                except (TypeError, ValueError):
                    limit = None

                if limit is not None:
                    if limit <= 0:
                        return []

                    messages = messages[-limit:]

            return list(messages)

    def history_dict(
        self,
        limit: Optional[int] = None,
        *,
        role: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return [
            message.to_dict()
            for message in self.history(
                limit=limit,
                role=role,
            )
        ]

    def clear_history(self) -> int:
        """
        Menghapus seluruh message.

        Return:
            jumlah message yang dihapus.
        """
        with self._lock:
            count = len(self._messages)

            self._messages.clear()

            self._touch()

            return count

    def remove_message(
        self,
        message_id: str,
    ) -> bool:
        """
        Menghapus message berdasarkan ID.
        """
        message_id = normalize_text(
            message_id
        )

        if not message_id:
            return False

        with self._lock:
            index = self._find_message_index(
                message_id
            )

            if index < 0:
                return False

            del self._messages[index]

            self._renumber_sequences()

            self._touch()

            return True

    # ------------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        role: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> list[ConversationMessage]:
        """
        Mencari message berdasarkan substring.
        """
        query = normalize_text(query)

        if not query:
            return []

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20

        if limit <= 0:
            return []

        if case_sensitive:
            needle = query
        else:
            needle = query.lower()

        results: list[
            ConversationMessage
        ] = []

        with self._lock:
            messages = self._messages

            normalized_role = (
                normalize_role(role)
                if role is not None
                else None
            )

            for message in reversed(messages):
                if (
                    normalized_role is not None
                    and message.role
                    != normalized_role
                ):
                    continue

                haystack = (
                    message.content
                    if case_sensitive
                    else message.content.lower()
                )

                if needle in haystack:
                    results.append(message)

                if len(results) >= limit:
                    break

        results.reverse()

        return results

    def contains(
        self,
        query: str,
    ) -> bool:
        return bool(
            self.search(
                query,
                limit=1,
            )
        )

    # ------------------------------------------------------------------------
    # CONTEXT STATE
    # ------------------------------------------------------------------------

    def set_topic(
        self,
        topic: Optional[str],
    ) -> "ConversationContext":
        with self._lock:
            value = normalize_text(topic)

            self.active_topic = (
                value or None
            )

            self._touch()

            return self

    def set_intent(
        self,
        intent: Optional[str],
    ) -> "ConversationContext":
        with self._lock:
            value = normalize_text(intent)

            self.active_intent = (
                value or None
            )

            self._touch()

            return self

    def set_agent(
        self,
        agent: Optional[str],
    ) -> "ConversationContext":
        with self._lock:
            value = normalize_text(agent)

            self.active_agent = (
                value or None
            )

            self._touch()

            return self

    def set_task(
        self,
        task: Optional[str],
    ) -> "ConversationContext":
        with self._lock:
            value = normalize_text(task)

            self.active_task = (
                value or None
            )

            self._touch()

            return self

    def set_summary(
        self,
        summary: Optional[str],
    ) -> "ConversationContext":
        with self._lock:
            self.summary = normalize_text(
                summary
            )

            self._touch()

            return self

    def set_status(
        self,
        status: str,
    ) -> "ConversationContext":
        status = normalize_text(
            status
        ).lower()

        if status not in VALID_CONTEXT_STATUSES:
            raise ValueError(
                f"Status context '{status}' tidak valid."
            )

        with self._lock:
            self.status = status

            self._touch()

            return self

    # ------------------------------------------------------------------------
    # VARIABLES
    # ------------------------------------------------------------------------

    def set_variable(
        self,
        key: str,
        value: Any,
    ) -> "ConversationContext":
        key = normalize_text(key)

        if not key:
            raise ValueError(
                "Nama variable tidak boleh kosong."
            )

        with self._lock:
            self.variables[key] = value

            self._touch()

            return self

    def get_variable(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        key = normalize_text(key)

        with self._lock:
            return self.variables.get(
                key,
                default,
            )

    def has_variable(
        self,
        key: str,
    ) -> bool:
        key = normalize_text(key)

        with self._lock:
            return key in self.variables

    def remove_variable(
        self,
        key: str,
    ) -> bool:
        key = normalize_text(key)

        with self._lock:
            if key not in self.variables:
                return False

            del self.variables[key]

            self._touch()

            return True

    def clear_variables(self) -> int:
        with self._lock:
            count = len(self.variables)

            self.variables.clear()

            self._touch()

            return count

    # ------------------------------------------------------------------------
    # TAGS
    # ------------------------------------------------------------------------

    def add_tag(
        self,
        tag: str,
    ) -> "ConversationContext":
        tag = normalize_text(tag)

        if not tag:
            return self

        with self._lock:
            self.tags.add(tag)

            self._touch()

            return self

    def add_tags(
        self,
        tags: Iterable[str],
    ) -> "ConversationContext":
        for tag in tags:
            self.add_tag(tag)

        return self

    def remove_tag(
        self,
        tag: str,
    ) -> bool:
        tag = normalize_text(tag)

        with self._lock:
            if tag not in self.tags:
                return False

            self.tags.remove(tag)

            self._touch()

            return True

    def has_tag(
        self,
        tag: str,
    ) -> bool:
        tag = normalize_text(tag)

        with self._lock:
            return tag in self.tags

    def get_tags(self) -> list[str]:
        with self._lock:
            return sorted(self.tags)

    # ------------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------------

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> "ConversationContext":
        key = normalize_text(key)

        if not key:
            raise ValueError(
                "Metadata key tidak boleh kosong."
            )

        with self._lock:
            self.metadata[key] = value

            self._touch()

            return self

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        key = normalize_text(key)

        with self._lock:
            return self.metadata.get(
                key,
                default,
            )

    def remove_metadata(
        self,
        key: str,
    ) -> bool:
        key = normalize_text(key)

        with self._lock:
            if key not in self.metadata:
                return False

            del self.metadata[key]

            self._touch()

            return True

    # ------------------------------------------------------------------------
    # SNAPSHOT
    # ------------------------------------------------------------------------

    def snapshot(self) -> ConversationContextSnapshot:
        """
        Membuat snapshot state context.
        """
        with self._lock:
            snapshot = (
                ConversationContextSnapshot(
                    context_id=self.context_id,
                    session_id=self.session_id,
                    namespace=self.namespace,
                    status=self.status,
                    active_topic=self.active_topic,
                    active_intent=self.active_intent,
                    active_agent=self.active_agent,
                    active_task=self.active_task,
                    message_count=len(
                        self._messages
                    ),
                    context_variables=dict(
                        self.variables
                    ),
                    summary=self.summary,
                    metadata={
                        "created_at": self.created_at,
                        "updated_at": self.updated_at,
                        "message_count": len(
                            self._messages
                        ),
                        "tags": sorted(
                            self.tags
                        ),
                    },
                )
            )

            self._snapshots.append(
                snapshot
            )

            return snapshot

    def snapshots(self) -> list[
        ConversationContextSnapshot
    ]:
        with self._lock:
            return list(self._snapshots)

    # ------------------------------------------------------------------------
    # RESTORE
    # ------------------------------------------------------------------------

    def restore_snapshot(
        self,
        snapshot: ConversationContextSnapshot,
    ) -> "ConversationContext":
        if not isinstance(
            snapshot,
            ConversationContextSnapshot,
        ):
            raise TypeError(
                "Snapshot harus berupa "
                "ConversationContextSnapshot."
            )

        with self._lock:
            self.status = snapshot.status

            self.active_topic = (
                snapshot.active_topic
            )

            self.active_intent = (
                snapshot.active_intent
            )

            self.active_agent = (
                snapshot.active_agent
            )

            self.active_task = (
                snapshot.active_task
            )

            self.variables = dict(
                snapshot.context_variables
            )

            self.summary = snapshot.summary

            tags = snapshot.metadata.get(
                "tags",
                [],
            )

            self.tags = set(
                str(tag)
                for tag in tags
            )

            self._touch()

            return self

    # ------------------------------------------------------------------------
    # CONTEXT FORMATTING
    # ------------------------------------------------------------------------

    def format_history(
        self,
        limit: Optional[int] = None,
    ) -> str:
        """
        Format history menjadi text yang mudah diberikan
        kepada LLM.
        """
        messages = self.history(
            limit=limit
        )

        if not messages:
            return ""

        lines: list[str] = []

        for message in messages:
            role = message.role.upper()

            lines.append(
                f"{role}: {message.content}"
            )

        return "\n".join(lines)

    def build_prompt_context(
        self,
        limit: Optional[int] = None,
    ) -> str:
        """
        Membuat context block untuk brain/LLM.
        """
        with self._lock:
            lines: list[str] = []

            lines.append(
                "=== ZAI CONVERSATION CONTEXT ==="
            )

            lines.append(
                f"Context ID: {self.context_id}"
            )

            lines.append(
                f"Session ID: {self.session_id}"
            )

            lines.append(
                f"Namespace: {self.namespace}"
            )

            lines.append(
                f"Status: {self.status}"
            )

            if self.active_topic:
                lines.append(
                    f"Active Topic: "
                    f"{self.active_topic}"
                )

            if self.active_intent:
                lines.append(
                    f"Active Intent: "
                    f"{self.active_intent}"
                )

            if self.active_agent:
                lines.append(
                    f"Active Agent: "
                    f"{self.active_agent}"
                )

            if self.active_task:
                lines.append(
                    f"Active Task: "
                    f"{self.active_task}"
                )

            if self.summary:
                lines.append(
                    f"Summary: {self.summary}"
                )

            if self.variables:
                lines.append(
                    "Variables:"
                )

                for key, value in (
                    self.variables.items()
                ):
                    lines.append(
                        f"- {key}: {value}"
                    )

            lines.append("")

            history_text = (
                self.format_history(
                    limit=limit
                )
            )

            if history_text:
                lines.append(
                    "Conversation History:"
                )

                lines.append(
                    history_text
                )

            lines.append(
                "=== END ZAI CONVERSATION CONTEXT ==="
            )

            return "\n".join(lines)

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    def generate_basic_summary(self) -> str:
        """
        Membuat summary sederhana tanpa LLM.

        Ini bukan summarization AI.
        Tujuannya adalah menyediakan fallback context summary.
        """
        with self._lock:
            if not self._messages:
                return "Belum ada percakapan."

            latest_user = None

            latest_assistant = None

            for message in reversed(
                self._messages
            ):
                if (
                    latest_user is None
                    and message.is_user
                ):
                    latest_user = message.content

                if (
                    latest_assistant is None
                    and message.is_assistant
                ):
                    latest_assistant = (
                        message.content
                    )

                if (
                    latest_user is not None
                    and latest_assistant is not None
                ):
                    break

            parts: list[str] = []

            parts.append(
                f"Total pesan: "
                f"{len(self._messages)}."
            )

            if self.active_topic:
                parts.append(
                    f"Topik aktif: "
                    f"{self.active_topic}."
                )

            if self.active_intent:
                parts.append(
                    f"Intent aktif: "
                    f"{self.active_intent}."
                )

            if latest_user:
                parts.append(
                    f"Pesan user terakhir: "
                    f"{latest_user[:200]}."
                )

            if latest_assistant:
                parts.append(
                    f"Respons assistant terakhir: "
                    f"{latest_assistant[:200]}."
                )

            return " ".join(parts)

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            role_distribution: dict[
                str,
                int,
            ] = {}

            for message in self._messages:
                role_distribution[
                    message.role
                ] = (
                    role_distribution.get(
                        message.role,
                        0,
                    )
                    + 1
                )

            total_characters = sum(
                message.length
                for message in self._messages
            )

            return {
                "context_id": self.context_id,
                "session_id": self.session_id,
                "namespace": self.namespace,
                "status": self.status,
                "message_count": len(
                    self._messages
                ),
                "user_message_count": sum(
                    1
                    for message
                    in self._messages
                    if message.is_user
                ),
                "assistant_message_count": sum(
                    1
                    for message
                    in self._messages
                    if message.is_assistant
                ),
                "tool_message_count": sum(
                    1
                    for message
                    in self._messages
                    if message.is_tool
                ),
                "agent_message_count": sum(
                    1
                    for message
                    in self._messages
                    if message.is_agent
                ),
                "system_message_count": sum(
                    1
                    for message
                    in self._messages
                    if message.is_system
                ),
                "total_characters": total_characters,
                "role_distribution": role_distribution,
                "variable_count": len(
                    self.variables
                ),
                "metadata_count": len(
                    self.metadata
                ),
                "tag_count": len(
                    self.tags
                ),
                "snapshot_count": len(
                    self._snapshots
                ),
                "operation_count": (
                    self._operation_count
                ),
                "active_topic": (
                    self.active_topic
                ),
                "active_intent": (
                    self.active_intent
                ),
                "active_agent": (
                    self.active_agent
                ),
                "active_task": (
                    self.active_task
                ),
                "version": self.VERSION,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    def validate(
        self,
    ) -> dict[str, Any]:
        """
        Validasi internal state context.
        """
        errors: list[str] = []

        warnings: list[str] = []

        if not self.context_id:
            errors.append(
                "context_id kosong."
            )

        if not self.session_id:
            errors.append(
                "session_id kosong."
            )

        if not self.namespace:
            errors.append(
                "namespace kosong."
            )

        if self.status not in (
            VALID_CONTEXT_STATUSES
        ):
            errors.append(
                "status context tidak valid."
            )

        if (
            self.max_messages <= 0
            or self.max_messages
            > MAX_ALLOWED_MESSAGES
        ):
            errors.append(
                "max_messages berada di luar "
                "batas valid."
            )

        if len(self._messages) > (
            self.max_messages
        ):
            errors.append(
                "Jumlah message melebihi "
                "max_messages."
            )

        for index, message in enumerate(
            self._messages,
            start=1,
        ):
            if not message.message_id:
                errors.append(
                    f"Message #{index} "
                    "tidak memiliki ID."
                )

            if not message.content:
                warnings.append(
                    f"Message #{index} "
                    "memiliki content kosong."
                )

            if message.role not in (
                VALID_ROLES
            ):
                errors.append(
                    f"Message #{index} "
                    f"memiliki role "
                    f"'{message.role}'."
                )

        return {
            "valid": not errors,
            "status": (
                "VALID"
                if not errors
                else "INVALID"
            ),
            "errors": errors,
            "warnings": warnings,
            "message_count": len(
                self._messages
            ),
            "version": self.VERSION,
        }

    # ------------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------------

    def to_dict(
        self,
        *,
        include_history: bool = True,
    ) -> dict[str, Any]:
        """
        Serialisasi lengkap context.
        """
        with self._lock:
            data: dict[str, Any] = {
                "context_version": self.VERSION,
                "context_id": self.context_id,
                "session_id": self.session_id,
                "namespace": self.namespace,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "active_topic": self.active_topic,
                "active_intent": self.active_intent,
                "active_agent": self.active_agent,
                "active_task": self.active_task,
                "summary": self.summary,
                "max_messages": self.max_messages,
                "metadata": dict(
                    self.metadata
                ),
                "variables": dict(
                    self.variables
                ),
                "tags": sorted(
                    self.tags
                ),
                "message_count": len(
                    self._messages
                ),
                "operation_count": (
                    self._operation_count
                ),
            }

            if include_history:
                data["messages"] = [
                    message.to_dict()
                    for message
                    in self._messages
                ]
            else:
                data["messages"] = []

            return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ConversationContext":
        """
        Reconstruct context dari dictionary.
        """
        if not isinstance(
            data,
            Mapping,
        ):
            raise TypeError(
                "ConversationContext.from_dict "
                "membutuhkan Mapping."
            )

        context = cls(
            context_id=data.get(
                "context_id"
            ),
            session_id=data.get(
                "session_id"
            ),
            namespace=data.get(
                "namespace",
                DEFAULT_NAMESPACE,
            ),
            max_messages=data.get(
                "max_messages",
                DEFAULT_MAX_MESSAGES,
            ),
            metadata=data.get(
                "metadata"
            ),
        )

        context.status = data.get(
            "status",
            DEFAULT_CONTEXT_STATUS,
        )

        if context.status not in (
            VALID_CONTEXT_STATUSES
        ):
            context.status = (
                DEFAULT_CONTEXT_STATUS
            )

        context.created_at = data.get(
            "created_at",
            context.created_at,
        )

        context.updated_at = data.get(
            "updated_at",
            context.updated_at,
        )

        context.active_topic = (
            normalize_text(
                data.get("active_topic")
            )
            or None
        )

        context.active_intent = (
            normalize_text(
                data.get("active_intent")
            )
            or None
        )

        context.active_agent = (
            normalize_text(
                data.get("active_agent")
            )
            or None
        )

        context.active_task = (
            normalize_text(
                data.get("active_task")
            )
            or None
        )

        context.summary = normalize_text(
            data.get("summary")
        )

        context.variables = dict(
            data.get(
                "variables",
                {},
            )
            or {}
        )

        context.tags = set(
            str(tag)
            for tag in (
                data.get(
                    "tags",
                    [],
                )
                or []
            )
        )

        raw_messages = data.get(
            "messages",
            [],
        )

        if isinstance(
            raw_messages,
            Iterable,
        ) and not isinstance(
            raw_messages,
            (str, bytes),
        ):
            for item in raw_messages:
                if isinstance(
                    item,
                    ConversationMessage,
                ):
                    message = item

                elif isinstance(
                    item,
                    Mapping,
                ):
                    message = (
                        ConversationMessage.from_dict(
                            item
                        )
                    )

                else:
                    continue

                context._messages.append(
                    message
                )

        context._renumber_sequences()

        context._trim_history_if_needed()

        context._operation_count = int(
            data.get(
                "operation_count",
                0,
            )
        )

        return context

    # ------------------------------------------------------------------------
    # CLONE
    # ------------------------------------------------------------------------

    def clone(
        self,
        *,
        new_context_id: bool = True,
        new_session_id: bool = False,
    ) -> "ConversationContext":
        """
        Membuat salinan context.
        """
        data = self.to_dict(
            include_history=True
        )

        if new_context_id:
            data["context_id"] = (
                generate_id("ctx")
            )

        if new_session_id:
            data["session_id"] = (
                generate_id("session")
            )

        return self.from_dict(data)

    # ------------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------------

    def reset(
        self,
        *,
        clear_metadata: bool = False,
        clear_variables: bool = True,
        clear_tags: bool = True,
    ) -> "ConversationContext":
        """
        Reset conversation state.
        """
        with self._lock:
            self._messages.clear()

            self.active_topic = None

            self.active_intent = None

            self.active_agent = None

            self.active_task = None

            self.summary = ""

            if clear_variables:
                self.variables.clear()

            if clear_tags:
                self.tags.clear()

            if clear_metadata:
                self.metadata.clear()

            self.status = "active"

            self._touch()

            return self

    def archive(self) -> "ConversationContext":
        return self.set_status(
            "archived"
        )

    def pause(self) -> "ConversationContext":
        return self.set_status(
            "paused"
        )

    def complete(self) -> "ConversationContext":
        return self.set_status(
            "completed"
        )

    def activate(self) -> "ConversationContext":
        return self.set_status(
            "active"
        )

    # ------------------------------------------------------------------------
    # LAST CONTENT HELPERS
    # ------------------------------------------------------------------------

    def last_user_message(
        self,
    ) -> Optional[ConversationMessage]:
        with self._lock:
            for message in reversed(
                self._messages
            ):
                if message.is_user:
                    return message

        return None

    def last_assistant_message(
        self,
    ) -> Optional[ConversationMessage]:
        with self._lock:
            for message in reversed(
                self._messages
            ):
                if message.is_assistant:
                    return message

        return None

    def last_tool_message(
        self,
    ) -> Optional[ConversationMessage]:
        with self._lock:
            for message in reversed(
                self._messages
            ):
                if message.is_tool:
                    return message

        return None

    def latest_user_text(
        self,
    ) -> str:
        message = (
            self.last_user_message()
        )

        if message is None:
            return ""

        return message.content

    def latest_assistant_text(
        self,
    ) -> str:
        message = (
            self.last_assistant_message()
        )

        if message is None:
            return ""

        return message.content

    # ------------------------------------------------------------------------
    # CONTEXT RELEVANCE
    # ------------------------------------------------------------------------

    def relevant_messages(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        """
        Lightweight relevance retrieval.

        Belum menggunakan embedding/vector database.
        Untuk tahap awal digunakan token overlap.
        """
        query = normalize_text(query)

        if not query:
            return []

        query_tokens = {
            token.lower()
            for token in query.split()
            if token.strip()
        }

        if not query_tokens:
            return []

        scored: list[
            tuple[int, ConversationMessage]
        ] = []

        with self._lock:
            for message in self._messages:
                content_tokens = {
                    token.lower()
                    for token in message.content.split()
                    if token.strip()
                }

                score = len(
                    query_tokens
                    & content_tokens
                )

                if score > 0:
                    scored.append(
                        (
                            score,
                            message,
                        )
                    )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].sequence,
            ),
            reverse=True,
        )

        return [
            message
            for _, message
            in scored[:limit]
        ]

    # ------------------------------------------------------------------------
    # CONTEXT PACK
    # ------------------------------------------------------------------------

    def build_context_pack(
        self,
        query: Optional[str] = None,
        *,
        history_limit: int = 20,
        relevant_limit: int = 10,
    ) -> dict[str, Any]:
        """
        Membuat paket context yang nantinya dapat langsung
        dipakai oleh ZAIBrain.
        """
        with self._lock:
            data: dict[str, Any] = {
                "context_id": self.context_id,
                "session_id": self.session_id,
                "namespace": self.namespace,
                "status": self.status,
                "active_topic": self.active_topic,
                "active_intent": self.active_intent,
                "active_agent": self.active_agent,
                "active_task": self.active_task,
                "summary": self.summary,
                "variables": dict(
                    self.variables
                ),
                "tags": sorted(
                    self.tags
                ),
                "history": self.history_dict(
                    limit=history_limit
                ),
                "latest_user_message": (
                    self.latest_user_text()
                ),
                "latest_assistant_message": (
                    self.latest_assistant_text()
                ),
            }

        if query:
            data["relevant_messages"] = [
                message.to_dict()
                for message
                in self.relevant_messages(
                    query,
                    limit=relevant_limit,
                )
            ]
        else:
            data["relevant_messages"] = []

        data["statistics"] = (
            self.statistics()
        )

        data["validation"] = (
            self.validate()
        )

        return data

    # ------------------------------------------------------------------------
    # INFO / HEALTH
    # ------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "context": "ConversationContext",
            "version": self.VERSION,
            "context_id": self.context_id,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "status": self.status,
            "message_count": self.message_count,
            "active_topic": self.active_topic,
            "active_intent": self.active_intent,
            "active_agent": self.active_agent,
            "active_task": self.active_task,
            "max_messages": self.max_messages,
            "variable_count": len(
                self.variables
            ),
            "tag_count": len(
                self.tags
            ),
        }

    def health(self) -> dict[str, Any]:
        validation = self.validate()

        return {
            "context": "ConversationContext",
            "version": self.VERSION,
            "status": (
                "HEALTHY"
                if validation["valid"]
                else "UNHEALTHY"
            ),
            "context_status": self.status,
            "message_count": self.message_count,
            "validation_errors": validation[
                "errors"
            ],
            "validation_warnings": validation[
                "warnings"
            ],
        }

    # ------------------------------------------------------------------------
    # REPRESENTATION
    # ------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.message_count

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            "ConversationContext("
            f"context_id={self.context_id!r}, "
            f"session_id={self.session_id!r}, "
            f"messages={self.message_count}, "
            f"status={self.status!r}"
            ")"
        )


# ============================================================================
# CONVERSATION CONTEXT MANAGER
# ============================================================================


class ConversationContextManager:
    """
    Manager untuk banyak ConversationContext.

    Digunakan ketika ZAI memiliki banyak session.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        max_contexts: int = 1000,
    ) -> None:
        try:
            max_contexts = int(
                max_contexts
            )
        except (TypeError, ValueError):
            max_contexts = 1000

        if max_contexts <= 0:
            max_contexts = 1000

        self.max_contexts = max_contexts

        self._contexts: dict[
            str,
            ConversationContext,
        ] = {}

        self._lock = RLock()

        self._created_count = 0

        self._deleted_count = 0

    # ------------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------------

    def create(
        self,
        *,
        session_id: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
        context_id: Optional[str] = None,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> ConversationContext:
        with self._lock:
            if (
                len(self._contexts)
                >= self.max_contexts
            ):
                raise RuntimeError(
                    "Batas maksimum conversation "
                    "context telah tercapai."
                )

            context = ConversationContext(
                context_id=context_id,
                session_id=session_id,
                namespace=namespace,
                max_messages=max_messages,
                metadata=metadata,
            )

            self._contexts[
                context.context_id
            ] = context

            self._created_count += 1

            return context

    # ------------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------------

    def get(
        self,
        context_id: str,
    ) -> ConversationContext:
        context_id = normalize_text(
            context_id
        )

        with self._lock:
            try:
                return self._contexts[
                    context_id
                ]
            except KeyError as exc:
                raise KeyError(
                    f"Conversation context "
                    f"'{context_id}' tidak ditemukan."
                ) from exc

    def get_or_create(
        self,
        session_id: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> ConversationContext:
        session_id = normalize_text(
            session_id
        )

        with self._lock:
            for context in (
                self._contexts.values()
            ):
                if (
                    context.session_id
                    == session_id
                    and context.namespace
                    == normalize_namespace(
                        namespace
                    )
                    and context.status
                    != "archived"
                ):
                    return context

            return self.create(
                session_id=session_id,
                namespace=namespace,
            )

    # ------------------------------------------------------------------------
    # REGISTER
    # ------------------------------------------------------------------------

    def register(
        self,
        context: ConversationContext,
    ) -> ConversationContext:
        if not isinstance(
            context,
            ConversationContext,
        ):
            raise TypeError(
                "context harus berupa "
                "ConversationContext."
            )

        with self._lock:
            if (
                context.context_id
                not in self._contexts
                and len(self._contexts)
                >= self.max_contexts
            ):
                raise RuntimeError(
                    "Batas maksimum context "
                    "telah tercapai."
                )

            self._contexts[
                context.context_id
            ] = context

            return context

    # ------------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------------

    def delete(
        self,
        context_id: str,
    ) -> bool:
        context_id = normalize_text(
            context_id
        )

        with self._lock:
            if context_id not in self._contexts:
                return False

            del self._contexts[
                context_id
            ]

            self._deleted_count += 1

            return True

    # ------------------------------------------------------------------------
    # CLEAR
    # ------------------------------------------------------------------------

    def clear(self) -> int:
        with self._lock:
            count = len(
                self._contexts
            )

            self._contexts.clear()

            self._deleted_count += count

            return count

    # ------------------------------------------------------------------------
    # LIST
    # ------------------------------------------------------------------------

    def all(
        self,
    ) -> list[ConversationContext]:
        with self._lock:
            return list(
                self._contexts.values()
            )

    def ids(self) -> list[str]:
        with self._lock:
            return list(
                self._contexts.keys()
            )

    def count(self) -> int:
        with self._lock:
            return len(
                self._contexts
            )

    # ------------------------------------------------------------------------
    # SESSION
    # ------------------------------------------------------------------------

    def find_by_session(
        self,
        session_id: str,
    ) -> list[ConversationContext]:
        session_id = normalize_text(
            session_id
        )

        with self._lock:
            return [
                context
                for context
                in self._contexts.values()
                if context.session_id
                == session_id
            ]

    def find_by_namespace(
        self,
        namespace: str,
    ) -> list[ConversationContext]:
        namespace = normalize_namespace(
            namespace
        )

        with self._lock:
            return [
                context
                for context
                in self._contexts.values()
                if context.namespace
                == namespace
            ]

    # ------------------------------------------------------------------------
    # EXPORT / IMPORT
    # ------------------------------------------------------------------------

    def to_dict(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "manager": (
                    "ConversationContextManager"
                ),
                "version": self.VERSION,
                "max_contexts": self.max_contexts,
                "created_count": (
                    self._created_count
                ),
                "deleted_count": (
                    self._deleted_count
                ),
                "contexts": [
                    context.to_dict(
                        include_history=True
                    )
                    for context
                    in self._contexts.values()
                ],
            }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ConversationContextManager":
        if not isinstance(
            data,
            Mapping,
        ):
            raise TypeError(
                "ConversationContextManager."
                "from_dict membutuhkan Mapping."
            )

        manager = cls(
            max_contexts=data.get(
                "max_contexts",
                1000,
            )
        )

        raw_contexts = data.get(
            "contexts",
            [],
        )

        if isinstance(
            raw_contexts,
            Iterable,
        ) and not isinstance(
            raw_contexts,
            (str, bytes),
        ):
            for item in raw_contexts:
                if not isinstance(
                    item,
                    Mapping,
                ):
                    continue

                context = (
                    ConversationContext.from_dict(
                        item
                    )
                )

                manager.register(
                    context
                )

        manager._created_count = int(
            data.get(
                "created_count",
                len(manager._contexts),
            )
        )

        manager._deleted_count = int(
            data.get(
                "deleted_count",
                0,
            )
        )

        return manager

    # ------------------------------------------------------------------------
    # INFO / HEALTH
    # ------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "manager": (
                    "ConversationContextManager"
                ),
                "version": self.VERSION,
                "status": "READY",
                "context_count": len(
                    self._contexts
                ),
                "max_contexts": (
                    self.max_contexts
                ),
                "created_count": (
                    self._created_count
                ),
                "deleted_count": (
                    self._deleted_count
                ),
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            unhealthy: list[str] = []

            for context in (
                self._contexts.values()
            ):
                health = context.health()

                if health["status"] != "HEALTHY":
                    unhealthy.append(
                        context.context_id
                    )

            return {
                "manager": (
                    "ConversationContextManager"
                ),
                "version": self.VERSION,
                "status": (
                    "HEALTHY"
                    if not unhealthy
                    else "DEGRADED"
                ),
                "context_count": len(
                    self._contexts
                ),
                "unhealthy_contexts": unhealthy,
            }


# ============================================================================
# GLOBAL MANAGER
# ============================================================================


_default_context_manager: Optional[
    ConversationContextManager
] = None

_default_context_manager_lock = RLock()


def get_context_manager() -> ConversationContextManager:
    """
    Mengambil singleton context manager.
    """
    global _default_context_manager

    with _default_context_manager_lock:
        if _default_context_manager is None:
            _default_context_manager = (
                ConversationContextManager()
            )

        return _default_context_manager


def reset_context_manager() -> ConversationContextManager:
    """
    Reset singleton manager.
    """
    global _default_context_manager

    with _default_context_manager_lock:
        _default_context_manager = (
            ConversationContextManager()
        )

        return _default_context_manager


def get_or_create_context(
    session_id: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> ConversationContext:
    manager = get_context_manager()

    return manager.get_or_create(
        session_id,
        namespace=namespace,
    )


# ============================================================================
# CONVENIENCE API
# ============================================================================


def create_context(
    *,
    session_id: Optional[str] = None,
    namespace: str = DEFAULT_NAMESPACE,
    context_id: Optional[str] = None,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> ConversationContext:
    return get_context_manager().create(
        session_id=session_id,
        namespace=namespace,
        context_id=context_id,
        max_messages=max_messages,
        metadata=metadata,
    )


# ============================================================================
# SELF TEST
# ============================================================================


def self_test() -> dict[str, Any]:
    """
    Test internal conversation context engine.

    Return dictionary agar dapat digunakan dari PowerShell:
        python -c "from ai.context.conversation_context import self_test; print(self_test())"
    """

    # ------------------------------------------------------------------------
    # 1. CREATE
    # ------------------------------------------------------------------------

    context = ConversationContext(
        session_id="test-session",
        namespace="test",
        max_messages=20,
    )

    assert context.context_id

    assert context.session_id == (
        "test-session"
    )

    assert context.namespace == "test"

    # ------------------------------------------------------------------------
    # 2. MESSAGE
    # ------------------------------------------------------------------------

    user_message = (
        context.add_user_message(
            "Halo ZAI"
        )
    )

    assert isinstance(
        user_message,
        ConversationMessage,
    )

    assert user_message.role == "user"

    assert user_message.content == (
        "Halo ZAI"
    )

    assistant_message = (
        context.add_assistant_message(
            "Halo, saya siap membantu."
        )
    )

    assert assistant_message.role == (
        "assistant"
    )

    assert context.message_count == 2

    print(
        "CONVERSATION_CONTEXT_MESSAGE_OK"
    )

    # ------------------------------------------------------------------------
    # 3. HISTORY
    # ------------------------------------------------------------------------

    history = context.history()

    assert len(history) == 2

    assert history[0].content == (
        "Halo ZAI"
    )

    assert history[1].content == (
        "Halo, saya siap membantu."
    )

    print(
        "CONVERSATION_CONTEXT_HISTORY_OK"
    )

    # ------------------------------------------------------------------------
    # 4. ROLE FILTER
    # ------------------------------------------------------------------------

    user_history = context.history(
        role="user"
    )

    assert len(user_history) == 1

    assert user_history[0].is_user

    assistant_history = context.history(
        role="assistant"
    )

    assert len(assistant_history) == 1

    assert assistant_history[0].is_assistant

    # ------------------------------------------------------------------------
    # 5. SEARCH
    # ------------------------------------------------------------------------

    results = context.search(
        "Halo",
        limit=10,
    )

    assert len(results) == 2

    print(
        "CONVERSATION_CONTEXT_SEARCH_OK"
    )

    # ------------------------------------------------------------------------
    # 6. STATE
    # ------------------------------------------------------------------------

    context.set_topic(
        "Pembangunan Super ZAI"
    )

    context.set_intent(
        "general"
    )

    context.set_agent(
        "general_agent"
    )

    context.set_task(
        "Lanjut pembangunan ZAI"
    )

    assert context.active_topic == (
        "Pembangunan Super ZAI"
    )

    assert context.active_intent == (
        "general"
    )

    assert context.active_agent == (
        "general_agent"
    )

    # ------------------------------------------------------------------------
    # 7. VARIABLES
    # ------------------------------------------------------------------------

    context.set_variable(
        "project",
        "Super ZAI",
    )

    context.set_variable(
        "phase",
        "context_engine",
    )

    assert context.get_variable(
        "project"
    ) == "Super ZAI"

    assert context.has_variable(
        "phase"
    )

    # ------------------------------------------------------------------------
    # 8. TAGS
    # ------------------------------------------------------------------------

    context.add_tag(
        "zai"
    )

    context.add_tag(
        "development"
    )

    assert context.has_tag(
        "zai"
    )

    assert len(
        context.get_tags()
    ) == 2

    # ------------------------------------------------------------------------
    # 9. SUMMARY
    # ------------------------------------------------------------------------

    context.set_summary(
        "ZAI sedang membangun context engine."
    )

    assert context.summary

    generated_summary = (
        context.generate_basic_summary()
    )

    assert generated_summary

    print(
        "CONVERSATION_CONTEXT_SUMMARY_OK"
    )

    # ------------------------------------------------------------------------
    # 10. SERIALIZATION
    # ------------------------------------------------------------------------

    data = context.to_dict()

    assert isinstance(
        data,
        dict,
    )

    assert data["context_id"] == (
        context.context_id
    )

    restored = (
        ConversationContext.from_dict(
            data
        )
    )

    assert restored.context_id == (
        context.context_id
    )

    assert restored.message_count == (
        context.message_count
    )

    assert restored.active_topic == (
        context.active_topic
    )

    assert restored.get_variable(
        "project"
    ) == "Super ZAI"

    print(
        "CONVERSATION_CONTEXT_SERIALIZATION_OK"
    )

    # ------------------------------------------------------------------------
    # 11. CLONE
    # ------------------------------------------------------------------------

    clone = context.clone()

    assert clone.context_id != (
        context.context_id
    )

    assert clone.session_id == (
        context.session_id
    )

    assert clone.message_count == (
        context.message_count
    )

    # ------------------------------------------------------------------------
    # 12. SNAPSHOT
    # ------------------------------------------------------------------------

    snapshot = context.snapshot()

    assert isinstance(
        snapshot,
        ConversationContextSnapshot,
    )

    assert snapshot.message_count == (
        context.message_count
    )

    # ------------------------------------------------------------------------
    # 13. CONTEXT PACK
    # ------------------------------------------------------------------------

    pack = context.build_context_pack(
        query="ZAI"
    )

    assert isinstance(
        pack,
        dict,
    )

    assert (
        "history"
        in pack
    )

    assert (
        "relevant_messages"
        in pack
    )

    assert (
        "statistics"
        in pack
    )

    # ------------------------------------------------------------------------
    # 14. VALIDATION
    # ------------------------------------------------------------------------

    validation = context.validate()

    assert validation["valid"] is True

    # ------------------------------------------------------------------------
    # 15. HEALTH
    # ------------------------------------------------------------------------

    health = context.health()

    assert health["status"] == (
        "HEALTHY"
    )

    # ------------------------------------------------------------------------
    # 16. MANAGER
    # ------------------------------------------------------------------------

    manager = (
        ConversationContextManager(
            max_contexts=10
        )
    )

    managed_context = manager.create(
        session_id="manager-session",
        namespace="default",
    )

    assert manager.count() == 1

    fetched = manager.get(
        managed_context.context_id
    )

    assert fetched.context_id == (
        managed_context.context_id
    )

    by_session = (
        manager.find_by_session(
            "manager-session"
        )
    )

    assert len(by_session) == 1

    print(
        "CONVERSATION_CONTEXT_MANAGER_OK"
    )

    # ------------------------------------------------------------------------
    # 17. MANAGER SERIALIZATION
    # ------------------------------------------------------------------------

    manager_data = manager.to_dict()

    assert isinstance(
        manager_data,
        dict,
    )

    restored_manager = (
        ConversationContextManager.from_dict(
            manager_data
        )
    )

    assert restored_manager.count() == 1

    # ------------------------------------------------------------------------
    # 18. GLOBAL API
    # ------------------------------------------------------------------------

    reset_context_manager()

    global_context = create_context(
        session_id="global-session"
    )

    assert global_context

    same_context = get_or_create_context(
        "global-session"
    )

    assert same_context.context_id == (
        global_context.context_id
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    final_statistics = (
        context.statistics()
    )

    return {
        "context": (
            "ConversationContext"
        ),
        "version": (
            CONVERSATION_CONTEXT_VERSION
        ),
        "status": "PASS",
        "context_id": context.context_id,
        "session_id": context.session_id,
        "message_count": context.message_count,
        "statistics": final_statistics,
        "validation": validation,
        "health": health,
        "manager": manager.health(),
    }


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    "CONVERSATION_CONTEXT_VERSION",
    "ConversationMessage",
    "ConversationContextSnapshot",
    "ConversationContext",
    "ConversationContextManager",
    "get_context_manager",
    "reset_context_manager",
    "get_or_create_context",
    "create_context",
    "self_test",
]


# ============================================================================
# DIRECT EXECUTION
# ============================================================================


if __name__ == "__main__":
    import pprint

    print(
        "=== ZAI CONVERSATION CONTEXT SELF TEST ==="
    )

    result = self_test()

    pprint.pp(result)

    print(
        "CONVERSATION_CONTEXT_OK"
    )
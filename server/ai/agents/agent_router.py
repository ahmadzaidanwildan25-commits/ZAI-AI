from __future__ import annotations

"""
ZAI Agent Router
================

Production-oriented routing layer untuk Super ZAI.

Tanggung jawab utama:

1. Menganalisis task pengguna.
2. Menentukan agent yang paling sesuai.
3. Mendukung explicit agent selection.
4. Mendukung keyword-based routing.
5. Mendukung capability-based routing.
6. Mendukung priority routing.
7. Mendukung fallback agent.
8. Menyimpan statistik routing.
9. Menyimpan histori routing.
10. Menyediakan health/status information.
11. Menyediakan explainability untuk setiap keputusan routing.
12. Menyediakan API yang kompatibel dengan AgentRuntime.
13. Aman digunakan secara async maupun synchronous.
14. Tidak bergantung pada LLM eksternal.
15. Siap dikembangkan menjadi intelligent router berbasis Ollama/LLM.

Arsitektur:

    User Task
        |
        v
    AgentRuntime
        |
        v
    AgentRouter
        |
        +---- explicit agent
        |
        +---- command routing
        |
        +---- keyword routing
        |
        +---- capability routing
        |
        +---- priority routing
        |
        +---- default agent
        |
        v
    Selected Agent

Versi:
    AgentRouter 2.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
import re
import time
import uuid


# ============================================================================
# CONSTANTS
# ============================================================================

ROUTER_VERSION = "2.0.0"

DEFAULT_AGENT_NAME = "general_agent"

DEFAULT_MIN_SCORE = 0.01

MAX_HISTORY_SIZE = 500

MAX_TASK_LENGTH = 20_000

UNKNOWN_AGENT = "unknown"

STATUS_READY = "READY"
STATUS_HEALTHY = "HEALTHY"
STATUS_DEGRADED = "DEGRADED"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def utc_now() -> datetime:
    """
    Return timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """
    Return current UTC timestamp in ISO-8601 format.
    """
    return utc_now().isoformat()


def normalize_text(value: Any) -> str:
    """
    Normalize arbitrary input into a clean lowercase string.

    This function intentionally performs conservative normalization.

    Example:

        "  Halo   ZAI  "

    becomes:

        "halo zai"
    """

    if value is None:
        return ""

    text = str(value)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    """
    Convert text into simple normalized tokens.
    """

    normalized = normalize_text(text)

    if not normalized:
        return []

    return re.findall(r"[a-zA-Z0-9_À-ÿ]+", normalized)


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value into integer.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value into float.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# ROUTE RULE
# ============================================================================


@dataclass(slots=True)
class RouteRule:
    """
    Defines one routing rule.

    A rule can match:

    - keywords
    - phrases
    - commands
    - capabilities
    - task types
    """

    agent_name: str

    keywords: tuple[str, ...] = ()

    phrases: tuple[str, ...] = ()

    commands: tuple[str, ...] = ()

    capabilities: tuple[str, ...] = ()

    task_types: tuple[str, ...] = ()

    priority: int = 0

    weight: float = 1.0

    enabled: bool = True

    description: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_name = str(self.agent_name).strip()

        self.keywords = tuple(
            normalize_text(item)
            for item in self.keywords
            if normalize_text(item)
        )

        self.phrases = tuple(
            normalize_text(item)
            for item in self.phrases
            if normalize_text(item)
        )

        self.commands = tuple(
            normalize_text(item)
            for item in self.commands
            if normalize_text(item)
        )

        self.capabilities = tuple(
            normalize_text(item)
            for item in self.capabilities
            if normalize_text(item)
        )

        self.task_types = tuple(
            normalize_text(item)
            for item in self.task_types
            if normalize_text(item)
        )

        self.priority = safe_int(self.priority)

        self.weight = max(
            0.0,
            safe_float(self.weight, 1.0),
        )

    def info(self) -> dict[str, Any]:
        """
        Serialize route rule.
        """

        return {
            "agent_name": self.agent_name,
            "keywords": list(self.keywords),
            "phrases": list(self.phrases),
            "commands": list(self.commands),
            "capabilities": list(self.capabilities),
            "task_types": list(self.task_types),
            "priority": self.priority,
            "weight": self.weight,
            "enabled": self.enabled,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# ROUTING CANDIDATE
# ============================================================================


@dataclass(slots=True)
class RouteCandidate:
    """
    Represents one candidate agent generated by the router.
    """

    agent_name: str

    score: float = 0.0

    priority: int = 0

    matched_keywords: list[str] = field(default_factory=list)

    matched_phrases: list[str] = field(default_factory=list)

    matched_commands: list[str] = field(default_factory=list)

    matched_capabilities: list[str] = field(default_factory=list)

    matched_task_types: list[str] = field(default_factory=list)

    reasons: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def add_reason(self, reason: str) -> None:
        """
        Add a routing explanation without duplication.
        """

        reason = str(reason).strip()

        if not reason:
            return

        if reason not in self.reasons:
            self.reasons.append(reason)

    def info(self) -> dict[str, Any]:
        """
        Serialize candidate.
        """

        return {
            "agent_name": self.agent_name,
            "score": round(self.score, 4),
            "priority": self.priority,
            "matched_keywords": list(self.matched_keywords),
            "matched_phrases": list(self.matched_phrases),
            "matched_commands": list(self.matched_commands),
            "matched_capabilities": list(
                self.matched_capabilities
            ),
            "matched_task_types": list(
                self.matched_task_types
            ),
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# ROUTING DECISION
# ============================================================================


@dataclass(slots=True)
class RouteDecision:
    """
    Final routing decision.
    """

    selected_agent: str

    task: str

    strategy: str

    confidence: float

    candidates: list[RouteCandidate] = field(
        default_factory=list
    )

    reason: str = ""

    fallback_used: bool = False

    route_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    created_at: str = field(
        default_factory=utc_now_iso
    )

    latency_ms: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def info(self) -> dict[str, Any]:
        """
        Serialize routing decision.
        """

        return {
            "route_id": self.route_id,
            "selected_agent": self.selected_agent,
            "task": self.task,
            "strategy": self.strategy,
            "confidence": round(
                self.confidence,
                4,
            ),
            "reason": self.reason,
            "fallback_used": self.fallback_used,
            "created_at": self.created_at,
            "latency_ms": round(
                self.latency_ms,
                4,
            ),
            "candidate_count": len(
                self.candidates
            ),
            "candidates": [
                candidate.info()
                for candidate in self.candidates
            ],
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Alias for info().
        """

        return self.info()


# ============================================================================
# AGENT ROUTER
# ============================================================================


class AgentRouter:
    """
    Intelligent deterministic router untuk agent ZAI.

    Router ini sengaja tidak membutuhkan LLM.

    Tujuannya adalah membuat layer routing yang stabil terlebih dahulu.

    Nanti router dapat ditingkatkan menjadi:

        AgentRouter
            |
            +-- Rule Router
            |
            +-- Semantic Router
            |
            +-- LLM Router
            |
            +-- Hybrid Router

    Tanpa mengubah interface AgentRuntime secara drastis.
    """

    VERSION = ROUTER_VERSION

    name = "AgentRouter"

    def __init__(
        self,
        default_agent: str = DEFAULT_AGENT_NAME,
        min_score: float = DEFAULT_MIN_SCORE,
        history_limit: int = MAX_HISTORY_SIZE,
    ) -> None:
        """
        Initialize router.
        """

        self.default_agent = (
            str(default_agent).strip()
            or DEFAULT_AGENT_NAME
        )

        self.min_score = max(
            0.0,
            safe_float(
                min_score,
                DEFAULT_MIN_SCORE,
            ),
        )

        self.history_limit = max(
            1,
            safe_int(
                history_limit,
                MAX_HISTORY_SIZE,
            ),
        )

        self._rules: dict[str, RouteRule] = {}

        self._history: list[RouteDecision] = []

        self._route_count = 0

        self._fallback_count = 0

        self._successful_route_count = 0

        self._failed_route_count = 0

        self._agent_route_counts: dict[str, int] = {}

        self._strategy_counts: dict[str, int] = {}

        self._last_decision: RouteDecision | None = None

        self._started_at = utc_now_iso()

        self._register_builtin_rules()

    # ========================================================================
    # BUILT-IN RULES
    # ========================================================================

    def _register_builtin_rules(self) -> None:
        """
        Register standard ZAI routing rules.

        General agent is intentionally broad.

        Specialized agents can be registered later.
        """

        self.register_rule(
            RouteRule(
                agent_name="general_agent",
                keywords=(
                    "halo",
                    "hai",
                    "hello",
                    "bantu",
                    "tolong",
                    "jelaskan",
                    "apa",
                    "bagaimana",
                    "kenapa",
                    "mengapa",
                    "buat",
                    "bantu saya",
                    "zai",
                ),
                phrases=(
                    "halo zai",
                    "hai zai",
                    "hello zai",
                    "bantu saya",
                    "tolong bantu",
                ),
                priority=1,
                weight=1.0,
                description=(
                    "General-purpose routing rule."
                ),
            )
        )

    # ========================================================================
    # RULE MANAGEMENT
    # ========================================================================

    def register_rule(
        self,
        rule: RouteRule,
    ) -> None:
        """
        Register a routing rule.
        """

        if not isinstance(
            rule,
            RouteRule,
        ):
            raise TypeError(
                "rule harus merupakan instance RouteRule."
            )

        if not rule.agent_name:
            raise ValueError(
                "RouteRule agent_name tidak boleh kosong."
            )

        self._rules[
            rule.agent_name
        ] = rule

    def unregister_rule(
        self,
        agent_name: str,
    ) -> bool:
        """
        Remove a route rule.
        """

        name = str(
            agent_name
        ).strip()

        if name in self._rules:
            del self._rules[name]
            return True

        return False

    def has_rule(
        self,
        agent_name: str,
    ) -> bool:
        """
        Check whether a route rule exists.
        """

        return (
            str(agent_name).strip()
            in self._rules
        )

    def get_rule(
        self,
        agent_name: str,
    ) -> RouteRule:
        """
        Retrieve a routing rule.
        """

        name = str(
            agent_name
        ).strip()

        try:
            return self._rules[name]
        except KeyError as exc:
            raise KeyError(
                f"Route rule '{name}' tidak ditemukan."
            ) from exc

    def rules(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return all route rules.
        """

        return [
            rule.info()
            for rule in self._rules.values()
        ]

    # ========================================================================
    # AGENT DISCOVERY
    # ========================================================================

    def _extract_agent_info(
        self,
        agent: Any,
    ) -> dict[str, Any]:
        """
        Extract normalized information from an agent.

        Supports:

            BaseAgent.info()

        or plain objects exposing:

            name
            version
            capabilities
        """

        if agent is None:
            return {
                "name": "",
                "version": "",
                "capabilities": [],
                "status": "UNKNOWN",
            }

        if hasattr(agent, "info"):
            try:
                data = agent.info()

                if isinstance(
                    data,
                    Mapping,
                ):
                    return {
                        "name": str(
                            data.get(
                                "name",
                                getattr(
                                    agent,
                                    "name",
                                    "",
                                ),
                            )
                        ),
                        "version": str(
                            data.get(
                                "version",
                                getattr(
                                    agent,
                                    "version",
                                    "",
                                ),
                            )
                        ),
                        "capabilities": list(
                            data.get(
                                "capabilities",
                                getattr(
                                    agent,
                                    "capabilities",
                                    (),
                                ),
                            )
                            or []
                        ),
                        "status": str(
                            data.get(
                                "status",
                                "READY",
                            )
                        ),
                        **{
                            key: value
                            for key, value
                            in data.items()
                            if key
                            not in {
                                "name",
                                "version",
                                "capabilities",
                                "status",
                            }
                        },
                    }

            except Exception:
                pass

        return {
            "name": str(
                getattr(
                    agent,
                    "name",
                    "",
                )
            ),
            "version": str(
                getattr(
                    agent,
                    "version",
                    "",
                )
            ),
            "capabilities": list(
                getattr(
                    agent,
                    "capabilities",
                    (),
                )
                or []
            ),
            "status": "READY",
        }

    def _agent_name(
        self,
        agent: Any,
    ) -> str:
        """
        Extract agent name.
        """

        info = self._extract_agent_info(
            agent
        )

        return str(
            info.get(
                "name",
                "",
            )
        ).strip()

    def _agent_names(
        self,
        available_agents: Any,
    ) -> list[str]:
        """
        Extract agent names from several possible input formats.
        """

        if available_agents is None:
            return []

        names: list[str] = []

        if isinstance(
            available_agents,
            Mapping,
        ):
            for key, value in (
                available_agents.items()
            ):
                name = self._agent_name(
                    value
                )

                if not name:
                    name = str(
                        key
                    ).strip()

                if name:
                    names.append(name)

            return list(
                dict.fromkeys(names)
            )

        try:
            iterable = list(
                available_agents
            )
        except TypeError:
            return []

        for agent in iterable:
            if isinstance(
                agent,
                str,
            ):
                name = agent.strip()
            else:
                name = self._agent_name(
                    agent
                )

            if name:
                names.append(name)

        return list(
            dict.fromkeys(names)
        )

    def _find_agent(
        self,
        available_agents: Any,
        agent_name: str,
    ) -> Any:
        """
        Find agent object by name.
        """

        target = str(
            agent_name
        ).strip()

        if isinstance(
            available_agents,
            Mapping,
        ):
            if target in available_agents:
                return available_agents[target]

            for key, agent in (
                available_agents.items()
            ):
                if str(key).strip() == target:
                    return agent

                if (
                    self._agent_name(agent)
                    == target
                ):
                    return agent

            return None

        try:
            iterable = list(
                available_agents
            )
        except TypeError:
            return None

        for agent in iterable:
            if (
                self._agent_name(agent)
                == target
            ):
                return agent

        return None

    # ========================================================================
    # TASK CLASSIFICATION
    # ========================================================================

    def _detect_command(
        self,
        task: str,
    ) -> str:
        """
        Detect command-style task.

        Examples:

            /chat
            /code
            /search
            /memory
            /system
        """

        text = normalize_text(
            task
        )

        if not text.startswith("/"):
            return ""

        match = re.match(
            r"^/([a-zA-Z0-9_-]+)",
            text,
        )

        if not match:
            return ""

        return match.group(1)

    def _detect_task_type(
        self,
        task: str,
    ) -> str:
        """
        Basic deterministic task classification.

        This is intentionally conservative.

        It is not intended to replace an LLM classifier.
        """

        text = normalize_text(
            task
        )

        if not text:
            return "empty"

        if self._detect_command(text):
            return "command"

        code_words = (
            "kode",
            "coding",
            "program",
            "python",
            "flutter",
            "dart",
            "javascript",
            "typescript",
            "fastapi",
            "api",
            "debug",
            "bug",
            "error",
            "compile",
            "function",
            "class",
            "database",
            "sql",
        )

        if any(
            word in text
            for word in code_words
        ):
            return "coding"

        search_words = (
            "cari",
            "search",
            "riset",
            "research",
            "berita",
            "informasi terbaru",
            "terbaru",
            "harga hari ini",
        )

        if any(
            word in text
            for word in search_words
        ):
            return "research"

        memory_words = (
            "ingat",
            "ingatkan",
            "memory",
            "kenangan",
            "simpan ini",
            "lupakan",
            "forget",
        )

        if any(
            word in text
            for word in memory_words
        ):
            return "memory"

        system_words = (
            "sistem",
            "system",
            "komputer",
            "windows",
            "shutdown",
            "restart",
            "cpu",
            "ram",
            "disk",
        )

        if any(
            word in text
            for word in system_words
        ):
            return "system"

        planning_words = (
            "rencana",
            "planning",
            "roadmap",
            "strategi",
            "langkah",
            "buatkan strategi",
        )

        if any(
            word in text
            for word in planning_words
        ):
            return "planning"

        return "general"

    # ========================================================================
    # MATCHING
    # ========================================================================

    def _match_rule(
        self,
        rule: RouteRule,
        task: str,
        task_type: str,
        requested_capabilities: set[str],
    ) -> RouteCandidate:
        """
        Calculate candidate score for one rule.
        """

        normalized = normalize_text(
            task
        )

        tokens = set(
            tokenize(normalized)
        )

        candidate = RouteCandidate(
            agent_name=rule.agent_name,
            priority=rule.priority,
        )

        if not rule.enabled:
            candidate.add_reason(
                "routing_rule_disabled"
            )
            return candidate

        # --------------------------------------------------------------------
        # Keyword matching
        # --------------------------------------------------------------------

        for keyword in rule.keywords:
            normalized_keyword = normalize_text(
                keyword
            )

            if not normalized_keyword:
                continue

            if (
                normalized_keyword in tokens
                or normalized_keyword in normalized
            ):
                candidate.matched_keywords.append(
                    normalized_keyword
                )

        if candidate.matched_keywords:
            keyword_score = min(
                0.45,
                0.08
                * len(
                    candidate.matched_keywords
                ),
            )

            candidate.score += keyword_score

            candidate.add_reason(
                "keyword_match"
            )

        # --------------------------------------------------------------------
        # Phrase matching
        # --------------------------------------------------------------------

        for phrase in rule.phrases:
            normalized_phrase = normalize_text(
                phrase
            )

            if (
                normalized_phrase
                and normalized_phrase in normalized
            ):
                candidate.matched_phrases.append(
                    normalized_phrase
                )

        if candidate.matched_phrases:
            phrase_score = min(
                0.60,
                0.20
                * len(
                    candidate.matched_phrases
                ),
            )

            candidate.score += phrase_score

            candidate.add_reason(
                "phrase_match"
            )

        # --------------------------------------------------------------------
        # Command matching
        # --------------------------------------------------------------------

        command = self._detect_command(
            normalized
        )

        if command:
            for rule_command in rule.commands:
                if command == rule_command:
                    candidate.matched_commands.append(
                        rule_command
                    )

        if candidate.matched_commands:
            candidate.score += 0.80

            candidate.add_reason(
                "command_match"
            )

        # --------------------------------------------------------------------
        # Task type matching
        # --------------------------------------------------------------------

        normalized_task_type = normalize_text(
            task_type
        )

        if (
            normalized_task_type
            and normalized_task_type
            in rule.task_types
        ):
            candidate.matched_task_types.append(
                normalized_task_type
            )

            candidate.score += 0.50

            candidate.add_reason(
                "task_type_match"
            )

        # --------------------------------------------------------------------
        # Capability matching
        # --------------------------------------------------------------------

        for capability in rule.capabilities:
            normalized_capability = normalize_text(
                capability
            )

            if (
                normalized_capability
                in requested_capabilities
            ):
                candidate.matched_capabilities.append(
                    normalized_capability
                )

        if candidate.matched_capabilities:
            candidate.score += min(
                0.50,
                0.20
                * len(
                    candidate.matched_capabilities
                ),
            )

            candidate.add_reason(
                "capability_match"
            )

        # --------------------------------------------------------------------
        # Priority bonus
        # --------------------------------------------------------------------

        if rule.priority > 0 and candidate.score > 0:
            candidate.score += min(
                0.20,
                rule.priority * 0.02,
            )

            candidate.add_reason(
                "priority_bonus"
            )

        # --------------------------------------------------------------------
        # Rule weight
        # --------------------------------------------------------------------

        candidate.score *= max(
            0.0,
            rule.weight,
        )

        candidate.metadata.update(
            {
                "rule_priority": rule.priority,
                "rule_weight": rule.weight,
                "rule_enabled": rule.enabled,
            }
        )

        return candidate

    # ========================================================================
    # EXPLICIT ROUTING
    # ========================================================================

    def _explicit_agent(
        self,
        kwargs: Mapping[str, Any],
    ) -> str:
        """
        Extract explicit agent request from kwargs.

        Supported names:

            agent
            agent_name
            target_agent
            requested_agent
        """

        keys = (
            "agent",
            "agent_name",
            "target_agent",
            "requested_agent",
        )

        for key in keys:
            value = kwargs.get(
                key
            )

            if value:
                return str(
                    value
                ).strip()

        return ""

    # ========================================================================
    # ROUTING
    # ========================================================================

    def route(
        self,
        task: str,
        available_agents: Any = None,
        **kwargs: Any,
    ) -> RouteDecision:
        """
        Route a task to the best available agent.

        Compatible call examples:

            router.route(
                "Halo ZAI",
                available_agents
            )

        or:

            router.route(
                "Halo ZAI",
                agents=available_agents
            )

        or:

            router.route(
                "Halo ZAI",
                agent_name="general_agent"
            )
        """

        started = time.perf_counter()

        self._route_count += 1

        raw_task = (
            ""
            if task is None
            else str(task)
        )

        normalized_task = normalize_text(
            raw_task
        )

        if len(
            normalized_task
        ) > MAX_TASK_LENGTH:
            normalized_task = normalized_task[
                :MAX_TASK_LENGTH
            ]

        # --------------------------------------------------------------------
        # Resolve agents from alternative kwargs.
        # --------------------------------------------------------------------

        if available_agents is None:
            available_agents = kwargs.get(
                "agents"
            )

        if available_agents is None:
            available_agents = kwargs.get(
                "available_agents"
            )

        agent_names = self._agent_names(
            available_agents
        )

        # --------------------------------------------------------------------
        # Requested capabilities.
        # --------------------------------------------------------------------

        requested_capabilities_raw = kwargs.get(
            "capabilities",
            kwargs.get(
                "required_capabilities",
                (),
            ),
        )

        try:
            requested_capabilities = {
                normalize_text(item)
                for item in requested_capabilities_raw
                if normalize_text(item)
            }
        except TypeError:
            requested_capabilities = set()

        # --------------------------------------------------------------------
        # Explicit agent.
        # --------------------------------------------------------------------

        explicit_agent = self._explicit_agent(
            kwargs
        )

        if explicit_agent:
            if (
                not agent_names
                or explicit_agent in agent_names
            ):
                decision = RouteDecision(
                    selected_agent=explicit_agent,
                    task=raw_task,
                    strategy="explicit",
                    confidence=1.0,
                    reason=(
                        "Agent dipilih secara eksplisit "
                        "oleh caller."
                    ),
                    fallback_used=False,
                    metadata={
                        "explicit_agent": True,
                        "available_agents": agent_names,
                    },
                )

                self._finalize_decision(
                    decision,
                    started,
                )

                return decision

        # --------------------------------------------------------------------
        # Empty task.
        # --------------------------------------------------------------------

        if not normalized_task:
            selected = self._select_default(
                agent_names
            )

            decision = RouteDecision(
                selected_agent=selected,
                task=raw_task,
                strategy="default_empty_task",
                confidence=0.0,
                reason=(
                    "Task kosong sehingga "
                    "default agent digunakan."
                ),
                fallback_used=(
                    selected == self.default_agent
                ),
                metadata={
                    "task_type": "empty",
                    "available_agents": agent_names,
                },
            )

            self._finalize_decision(
                decision,
                started,
            )

            return decision

        # --------------------------------------------------------------------
        # Task classification.
        # --------------------------------------------------------------------

        task_type = self._detect_task_type(
            normalized_task
        )

        # --------------------------------------------------------------------
        # Generate candidates.
        # --------------------------------------------------------------------

        candidates: list[RouteCandidate] = []

        for rule in self._rules.values():
            if agent_names and (
                rule.agent_name
                not in agent_names
            ):
                continue

            candidate = self._match_rule(
                rule=rule,
                task=normalized_task,
                task_type=task_type,
                requested_capabilities=(
                    requested_capabilities
                ),
            )

            if candidate.score >= self.min_score:
                candidates.append(
                    candidate
                )

        # --------------------------------------------------------------------
        # Capability-only discovery.
        # --------------------------------------------------------------------

        if available_agents is not None:
            for agent in (
                available_agents.values()
                if isinstance(
                    available_agents,
                    Mapping,
                )
                else available_agents
            ):
                name = self._agent_name(
                    agent
                )

                if not name:
                    continue

                if (
                    name
                    not in agent_names
                ):
                    continue

                info = self._extract_agent_info(
                    agent
                )

                capabilities = {
                    normalize_text(item)
                    for item in (
                        info.get(
                            "capabilities",
                            [],
                        )
                        or []
                    )
                }

                matched = sorted(
                    capabilities
                    & requested_capabilities
                )

                if not matched:
                    continue

                existing = next(
                    (
                        item
                        for item in candidates
                        if item.agent_name == name
                    ),
                    None,
                )

                if existing is None:
                    existing = RouteCandidate(
                        agent_name=name
                    )
                    candidates.append(
                        existing
                    )

                for capability in matched:
                    if (
                        capability
                        not in existing.matched_capabilities
                    ):
                        existing.matched_capabilities.append(
                            capability
                        )

                existing.score += min(
                    0.50,
                    0.20 * len(matched),
                )

                existing.add_reason(
                    "agent_capability_match"
                )

        # --------------------------------------------------------------------
        # General agent fallback candidate.
        # --------------------------------------------------------------------

        if (
            self.default_agent in agent_names
        ):
            existing_general = next(
                (
                    item
                    for item in candidates
                    if item.agent_name
                    == self.default_agent
                ),
                None,
            )

            if existing_general is None:
                existing_general = RouteCandidate(
                    agent_name=self.default_agent,
                    score=0.01,
                    priority=1,
                )

                existing_general.add_reason(
                    "default_agent_candidate"
                )

                candidates.append(
                    existing_general
                )

        # --------------------------------------------------------------------
        # Sort candidates.
        # --------------------------------------------------------------------

        candidates.sort(
            key=lambda candidate: (
                candidate.score,
                candidate.priority,
            ),
            reverse=True,
        )

        # --------------------------------------------------------------------
        # Select winner.
        # --------------------------------------------------------------------

        if candidates:
            winner = candidates[0]

            if (
                winner.score >= self.min_score
                and (
                    not agent_names
                    or winner.agent_name
                    in agent_names
                )
            ):
                confidence = min(
                    1.0,
                    winner.score,
                )

                strategy = (
                    "rule"
                    if winner.reasons
                    else "candidate"
                )

                reason = (
                    "Routing dipilih berdasarkan "
                    + ", ".join(
                        winner.reasons
                    )
                    + "."
                )

                decision = RouteDecision(
                    selected_agent=winner.agent_name,
                    task=raw_task,
                    strategy=strategy,
                    confidence=confidence,
                    candidates=candidates,
                    reason=reason,
                    fallback_used=False,
                    metadata={
                        "task_type": task_type,
                        "available_agents": agent_names,
                        "normalized_task": normalized_task,
                    },
                )

                self._finalize_decision(
                    decision,
                    started,
                )

                return decision

        # --------------------------------------------------------------------
        # Default fallback.
        # --------------------------------------------------------------------

        selected = self._select_default(
            agent_names
        )

        fallback_used = True

        self._fallback_count += 1

        decision = RouteDecision(
            selected_agent=selected,
            task=raw_task,
            strategy="fallback",
            confidence=0.0,
            candidates=candidates,
            reason=(
                "Tidak ditemukan routing rule "
                "yang memiliki kecocokan memadai. "
                "Default agent digunakan."
            ),
            fallback_used=fallback_used,
            metadata={
                "task_type": task_type,
                "available_agents": agent_names,
                "normalized_task": normalized_task,
            },
        )

        self._finalize_decision(
            decision,
            started,
        )

        return decision

    # ========================================================================
    # DEFAULT SELECTION
    # ========================================================================

    def _select_default(
        self,
        agent_names: list[str],
    ) -> str:
        """
        Select default agent.

        Priority:

            1. configured default agent
            2. general_agent
            3. first available agent
            4. configured default even if unavailable
        """

        if (
            self.default_agent
            and self.default_agent
            in agent_names
        ):
            return self.default_agent

        if (
            DEFAULT_AGENT_NAME
            in agent_names
        ):
            return DEFAULT_AGENT_NAME

        if agent_names:
            return agent_names[0]

        return self.default_agent

    # ========================================================================
    # FINALIZATION
    # ========================================================================

    def _finalize_decision(
        self,
        decision: RouteDecision,
        started: float,
    ) -> None:
        """
        Finalize and persist routing decision.
        """

        decision.latency_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            4,
        )

        self._last_decision = decision

        self._history.append(
            decision
        )

        if len(
            self._history
        ) > self.history_limit:
            del self._history[
                : len(self._history)
                - self.history_limit
            ]

        selected = (
            decision.selected_agent
        )

        self._agent_route_counts[
            selected
        ] = (
            self._agent_route_counts.get(
                selected,
                0,
            )
            + 1
        )

        strategy = decision.strategy

        self._strategy_counts[
            strategy
        ] = (
            self._strategy_counts.get(
                strategy,
                0,
            )
            + 1
        )

        if decision.selected_agent:
            self._successful_route_count += 1
        else:
            self._failed_route_count += 1

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    def select(
        self,
        task: str,
        available_agents: Any = None,
        **kwargs: Any,
    ) -> str:
        """
        Return only selected agent name.
        """

        return self.route(
            task,
            available_agents,
            **kwargs,
        ).selected_agent

    def choose(
        self,
        task: str,
        available_agents: Any = None,
        **kwargs: Any,
    ) -> str:
        """
        Alias for select().
        """

        return self.select(
            task,
            available_agents,
            **kwargs,
        )

    def explain(
        self,
        task: str,
        available_agents: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Return complete routing explanation.
        """

        return self.route(
            task,
            available_agents,
            **kwargs,
        ).info()

    # ========================================================================
    # HISTORY
    # ========================================================================

    def history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return recent routing decisions.
        """

        if limit is None:
            selected = self._history
        else:
            safe_limit = max(
                0,
                safe_int(
                    limit,
                    0,
                ),
            )

            if safe_limit == 0:
                return []

            selected = self._history[
                -safe_limit:
            ]

        return [
            item.info()
            for item in selected
        ]

    def clear_history(self) -> None:
        """
        Clear routing history.
        """

        self._history.clear()

        self._last_decision = None

    # ========================================================================
    # STATISTICS
    # ========================================================================

    @property
    def route_count(self) -> int:
        """
        Total number of route calls.
        """

        return self._route_count

    @property
    def fallback_count(self) -> int:
        """
        Total fallback routes.
        """

        return self._fallback_count

    @property
    def success_count(self) -> int:
        """
        Total successful route decisions.
        """

        return self._successful_route_count

    @property
    def failure_count(self) -> int:
        """
        Total failed route decisions.
        """

        return self._failed_route_count

    @property
    def success_rate(self) -> float:
        """
        Routing success rate.
        """

        if self._route_count <= 0:
            return 0.0

        return round(
            (
                self._successful_route_count
                / self._route_count
            )
            * 100,
            2,
        )

    @property
    def fallback_rate(self) -> float:
        """
        Routing fallback rate.
        """

        if self._route_count <= 0:
            return 0.0

        return round(
            (
                self._fallback_count
                / self._route_count
            )
            * 100,
            2,
        )

    # ========================================================================
    # INFORMATION
    # ========================================================================

    def info(self) -> dict[str, Any]:
        """
        Return router information.

        This is used by AgentRuntime.info().
        """

        return {
            "router": self.name,
            "version": self.VERSION,
            "status": STATUS_READY,
            "default_agent": self.default_agent,
            "min_score": self.min_score,
            "route_count": self._route_count,
            "fallback_count": self._fallback_count,
            "success_count": self._successful_route_count,
            "failure_count": self._failed_route_count,
            "success_rate": self.success_rate,
            "fallback_rate": self.fallback_rate,
            "rule_count": len(
                self._rules
            ),
            "history_size": len(
                self._history
            ),
            "agent_route_counts": dict(
                self._agent_route_counts
            ),
            "strategy_counts": dict(
                self._strategy_counts
            ),
            "available_rules": [
                rule.agent_name
                for rule in self._rules.values()
                if rule.enabled
            ],
            "started_at": self._started_at,
        }

    def summary(self) -> dict[str, Any]:
        """
        Compact router summary.
        """

        return {
            "router": self.name,
            "version": self.VERSION,
            "status": STATUS_READY,
            "default_agent": self.default_agent,
            "route_count": self._route_count,
            "fallback_count": self._fallback_count,
            "success_rate": self.success_rate,
            "fallback_rate": self.fallback_rate,
            "rule_count": len(
                self._rules
            ),
            "history_size": len(
                self._history
            ),
        }

    def health(self) -> dict[str, Any]:
        """
        Health status for runtime diagnostics.
        """

        status = STATUS_HEALTHY

        if not self.default_agent:
            status = STATUS_DEGRADED

        return {
            "router": self.name,
            "version": self.VERSION,
            "status": status,
            "default_agent": self.default_agent,
            "route_count": self._route_count,
            "fallback_count": self._fallback_count,
            "success_rate": self.success_rate,
            "fallback_rate": self.fallback_rate,
            "rule_count": len(
                self._rules
            ),
            "history_size": len(
                self._history
            ),
        }

    # ========================================================================
    # RESET
    # ========================================================================

    def reset_statistics(self) -> None:
        """
        Reset runtime routing statistics.
        """

        self._route_count = 0

        self._fallback_count = 0

        self._successful_route_count = 0

        self._failed_route_count = 0

        self._agent_route_counts.clear()

        self._strategy_counts.clear()

    # ========================================================================
    # CUSTOM AGENT REGISTRATION
    # ========================================================================

    def register_agent_rule(
        self,
        agent_name: str,
        *,
        keywords: Iterable[str] = (),
        phrases: Iterable[str] = (),
        commands: Iterable[str] = (),
        capabilities: Iterable[str] = (),
        task_types: Iterable[str] = (),
        priority: int = 0,
        weight: float = 1.0,
        enabled: bool = True,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> RouteRule:
        """
        Convenience method for registering custom routing rules.

        Example:

            router.register_agent_rule(
                "coding_agent",
                keywords=(
                    "python",
                    "coding",
                    "debug",
                ),
                task_types=(
                    "coding",
                ),
                priority=10,
            )
        """

        rule = RouteRule(
            agent_name=agent_name,
            keywords=tuple(
                keywords
            ),
            phrases=tuple(
                phrases
            ),
            commands=tuple(
                commands
            ),
            capabilities=tuple(
                capabilities
            ),
            task_types=tuple(
                task_types
            ),
            priority=priority,
            weight=weight,
            enabled=enabled,
            description=description,
            metadata=dict(
                metadata
                or {}
            ),
        )

        self.register_rule(
            rule
        )

        return rule

    # ========================================================================
    # ROUTING PREVIEW
    # ========================================================================

    def preview(
        self,
        task: str,
        available_agents: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Preview routing without permanently modifying statistics.

        A temporary router is used with the same configuration.
        """

        preview_router = AgentRouter(
            default_agent=self.default_agent,
            min_score=self.min_score,
            history_limit=1,
        )

        preview_router._rules = {
            name: rule
            for name, rule
            in self._rules.items()
        }

        decision = preview_router.route(
            task,
            available_agents,
            **kwargs,
        )

        return decision.info()

    # ========================================================================
    # BATCH ROUTING
    # ========================================================================

    def route_many(
        self,
        tasks: Iterable[str],
        available_agents: Any = None,
        **kwargs: Any,
    ) -> list[RouteDecision]:
        """
        Route multiple tasks sequentially.

        This method is intentionally synchronous.
        """

        decisions: list[RouteDecision] = []

        for task in tasks:
            decisions.append(
                self.route(
                    task,
                    available_agents,
                    **kwargs,
                )
            )

        return decisions

    # ========================================================================
    # EXPORT
    # ========================================================================

    def export_state(self) -> dict[str, Any]:
        """
        Export router configuration and statistics.
        """

        return {
            "router": self.name,
            "version": self.VERSION,
            "default_agent": self.default_agent,
            "min_score": self.min_score,
            "history_limit": self.history_limit,
            "rules": self.rules(),
            "statistics": {
                "route_count": self._route_count,
                "fallback_count": self._fallback_count,
                "success_count": (
                    self._successful_route_count
                ),
                "failure_count": (
                    self._failed_route_count
                ),
                "success_rate": self.success_rate,
                "fallback_rate": self.fallback_rate,
                "agent_route_counts": dict(
                    self._agent_route_counts
                ),
                "strategy_counts": dict(
                    self._strategy_counts
                ),
            },
            "history": self.history(),
        }

    # ========================================================================
    # RESTORE
    # ========================================================================

    def import_rules(
        self,
        rules: Iterable[
            Mapping[str, Any]
        ],
        clear_existing: bool = False,
    ) -> int:
        """
        Import routing rules from dictionaries.
        """

        if clear_existing:
            self._rules.clear()

        imported = 0

        for data in rules:
            if not isinstance(
                data,
                Mapping,
            ):
                continue

            agent_name = str(
                data.get(
                    "agent_name",
                    "",
                )
            ).strip()

            if not agent_name:
                continue

            rule = RouteRule(
                agent_name=agent_name,
                keywords=tuple(
                    data.get(
                        "keywords",
                        (),
                    )
                    or ()
                ),
                phrases=tuple(
                    data.get(
                        "phrases",
                        (),
                    )
                    or ()
                ),
                commands=tuple(
                    data.get(
                        "commands",
                        (),
                    )
                    or ()
                ),
                capabilities=tuple(
                    data.get(
                        "capabilities",
                        (),
                    )
                    or ()
                ),
                task_types=tuple(
                    data.get(
                        "task_types",
                        (),
                    )
                    or ()
                ),
                priority=safe_int(
                    data.get(
                        "priority",
                        0,
                    )
                ),
                weight=safe_float(
                    data.get(
                        "weight",
                        1.0,
                    )
                ),
                enabled=bool(
                    data.get(
                        "enabled",
                        True,
                    )
                ),
                description=str(
                    data.get(
                        "description",
                        "",
                    )
                ),
                metadata=dict(
                    data.get(
                        "metadata",
                        {},
                    )
                    or {}
                ),
            )

            self.register_rule(
                rule
            )

            imported += 1

        return imported

    # ========================================================================
    # DEBUG
    # ========================================================================

    def debug_route(
        self,
        task: str,
        available_agents: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Verbose debugging helper.

        Useful while building new agents.
        """

        decision = self.route(
            task,
            available_agents,
            **kwargs,
        )

        return {
            "task": task,
            "decision": decision.info(),
            "router": self.info(),
        }

    # ========================================================================
    # STRING REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        """

        return (
            "AgentRouter("
            f"version={self.VERSION!r}, "
            f"default_agent={self.default_agent!r}, "
            f"routes={self._route_count}, "
            f"rules={len(self._rules)}"
            ")"
        )

    def __str__(self) -> str:
        """
        Human-readable representation.
        """

        return (
            f"{self.name} "
            f"v{self.VERSION} "
            f"[{STATUS_READY}]"
        )


# ============================================================================
# FACTORY
# ============================================================================


def create_agent_router(
    default_agent: str = DEFAULT_AGENT_NAME,
    **kwargs: Any,
) -> AgentRouter:
    """
    Factory function for AgentRouter.

    Keeps future dependency injection simple.
    """

    return AgentRouter(
        default_agent=default_agent,
        **kwargs,
    )


# ============================================================================
# DEFAULT ROUTER INSTANCE
# ============================================================================


default_router = AgentRouter()


# ============================================================================
# PUBLIC HELPERS
# ============================================================================


def route_task(
    task: str,
    available_agents: Any = None,
    **kwargs: Any,
) -> RouteDecision:
    """
    Route a task using the default router instance.
    """

    return default_router.route(
        task,
        available_agents,
        **kwargs,
    )


def select_agent(
    task: str,
    available_agents: Any = None,
    **kwargs: Any,
) -> str:
    """
    Select an agent using the default router.
    """

    return default_router.select(
        task,
        available_agents,
        **kwargs,
    )


def router_info() -> dict[str, Any]:
    """
    Return default router information.
    """

    return default_router.info()


def router_health() -> dict[str, Any]:
    """
    Return default router health.
    """

    return default_router.health()


# ============================================================================
# SELF TEST
# ============================================================================


def _self_test() -> dict[str, Any]:
    """
    Internal router self-test.

    This function does not require FastAPI,
    Ollama, database, or external services.
    """

    class TestAgent:
        def __init__(
            self,
            name: str,
            capabilities: tuple[str, ...],
        ) -> None:
            self.name = name
            self.version = "test"
            self.capabilities = capabilities

        def info(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "version": self.version,
                "capabilities": list(
                    self.capabilities
                ),
                "status": "READY",
            }

    general = TestAgent(
        "general_agent",
        (
            "general_task",
            "text_processing",
            "basic_reasoning",
        ),
    )

    coding = TestAgent(
        "coding_agent",
        (
            "coding",
            "python",
            "debugging",
        ),
    )

    research = TestAgent(
        "research_agent",
        (
            "research",
            "web_search",
        ),
    )

    router = AgentRouter(
        default_agent="general_agent"
    )

    router.register_agent_rule(
        "coding_agent",
        keywords=(
            "python",
            "coding",
            "debug",
            "program",
        ),
        phrases=(
            "buat kode",
            "perbaiki kode",
        ),
        commands=(
            "code",
            "coding",
        ),
        capabilities=(
            "coding",
            "python",
        ),
        task_types=(
            "coding",
        ),
        priority=10,
        weight=1.0,
        description=(
            "Routing untuk coding tasks."
        ),
    )

    router.register_agent_rule(
        "research_agent",
        keywords=(
            "cari",
            "research",
            "riset",
            "berita",
        ),
        phrases=(
            "cari informasi",
            "lakukan riset",
        ),
        commands=(
            "search",
            "research",
        ),
        capabilities=(
            "research",
            "web_search",
        ),
        task_types=(
            "research",
        ),
        priority=9,
        weight=1.0,
        description=(
            "Routing untuk research tasks."
        ),
    )

    agents = [
        general,
        coding,
        research,
    ]

    result_general = router.route(
        "Halo ZAI",
        agents,
    )

    result_coding = router.route(
        "Tolong perbaiki kode Python saya",
        agents,
    )

    result_research = router.route(
        "Tolong cari informasi terbaru",
        agents,
    )

    result_explicit = router.route(
        "Halo",
        agents,
        agent_name="coding_agent",
    )

    assertions = {
        "general_route": (
            result_general.selected_agent
            == "general_agent"
        ),
        "coding_route": (
            result_coding.selected_agent
            == "coding_agent"
        ),
        "research_route": (
            result_research.selected_agent
            == "research_agent"
        ),
        "explicit_route": (
            result_explicit.selected_agent
            == "coding_agent"
        ),
        "route_count": (
            router.route_count == 4
        ),
        "health": (
            router.health()["status"]
            == STATUS_HEALTHY
        ),
        "rules": (
            router.has_rule(
                "coding_agent"
            )
            and router.has_rule(
                "research_agent"
            )
        ),
    }

    passed = all(
        assertions.values()
    )

    return {
        "success": passed,
        "assertions": assertions,
        "router": router.info(),
        "results": {
            "general": result_general.info(),
            "coding": result_coding.info(),
            "research": result_research.info(),
            "explicit": result_explicit.info(),
        },
    }


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    test_result = _self_test()

    print("=" * 72)
    print("ZAI AGENT ROUTER SELF TEST")
    print("=" * 72)

    print(
        "success:",
        test_result["success"],
    )

    print(
        "router:",
        test_result["router"],
    )

    print(
        "assertions:",
        test_result["assertions"],
    )

    print("=" * 72)

    if test_result["success"]:
        print(
            "AGENT_ROUTER_SELF_TEST_OK"
        )
    else:
        print(
            "AGENT_ROUTER_SELF_TEST_FAILED"
        )

        raise SystemExit(1)
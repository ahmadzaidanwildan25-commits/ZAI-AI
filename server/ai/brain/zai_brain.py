from __future__ import annotations

"""
ZAI Brain
=========

Central intelligence coordinator for the ZAI platform.

Architecture:

    User Task
        |
        v
    ZAIBrain
        |
        +--> Intent Analysis
        |
        +--> Task Classification
        |
        +--> Agent Planning
        |
        +--> Tool Planning
        |
        +--> Agent Runtime
        |
        +--> Tool Manager / Tool Registry
        |
        +--> Result Validation
        |
        +--> Response Synthesis
        |
        v
    Final Response


Design goals
------------

1. Do not tightly couple the brain to a specific implementation of
   AgentRuntime, AgentManager, ToolManager, or ToolRegistry.

2. Preserve compatibility with the existing ZAI platform.

3. Fail gracefully when optional components are unavailable.

4. Keep execution observable.

5. Keep planning deterministic for the current local intelligence layer.

6. Provide a stable foundation for a future LLM/Ollama brain.

7. Never silently execute an unknown tool.

8. Never silently execute an unknown agent.

9. Maintain execution statistics.

10. Produce structured results.

This module intentionally does not directly call Ollama.
The LLM provider will be added in a later layer.

The current brain is therefore the deterministic orchestration
and intelligence foundation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from inspect import isawaitable
from time import perf_counter
from typing import Any, Callable, Iterable
from uuid import uuid4


ZAI_BRAIN_VERSION = "1.0.0"


# ============================================================================
# ENUMS
# ============================================================================


class BrainStatus(str, Enum):
    """
    Lifecycle state of a brain execution.
    """

    READY = "ready"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"


class BrainIntent(str, Enum):
    """
    High-level task intent.
    """

    GENERAL = "general"
    CODING = "coding"
    RESEARCH = "research"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"


# ============================================================================
# HELPERS
# ============================================================================


def utc_now() -> str:
    """
    Return an ISO-8601 UTC timestamp.
    """

    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    """
    Normalize arbitrary input to a clean string.
    """

    if value is None:
        return ""

    text = str(value)

    return " ".join(
        text.replace("\x00", " ")
        .split()
    ).strip()


def safe_lower(value: Any) -> str:
    """
    Lowercase helper that never raises.
    """

    return normalize_text(value).lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float safely.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to int safely.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def object_to_dict(value: Any) -> dict[str, Any]:
    """
    Convert a common ZAI object to a dictionary.

    Supports:
        - dictionaries
        - dataclasses
        - objects exposing to_dict()
        - regular objects
    """

    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        try:
            result = to_dict()

            if isinstance(result, dict):
                return result
        except Exception:
            pass

    try:
        from dataclasses import asdict

        if hasattr(value, "__dataclass_fields__"):
            result = asdict(value)

            if isinstance(result, dict):
                return result
    except Exception:
        pass

    result: dict[str, Any] = {}

    for key in (
        "success",
        "status",
        "response",
        "data",
        "error",
        "agent",
        "tool",
        "execution_id",
        "metadata",
    ):
        if hasattr(value, key):
            try:
                result[key] = getattr(value, key)
            except Exception:
                continue

    return result


# ============================================================================
# BRAIN INTENT
# ============================================================================


@dataclass(slots=True)
class BrainIntentResult:
    """
    Result of intent analysis.
    """

    intent: BrainIntent
    confidence: float
    reasons: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    requires_agent: bool = True
    requires_tool: bool = False
    task_type: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
            "keywords": list(self.keywords),
            "entities": list(self.entities),
            "requires_agent": self.requires_agent,
            "requires_tool": self.requires_tool,
            "task_type": self.task_type,
        }


# ============================================================================
# BRAIN PLAN
# ============================================================================


@dataclass(slots=True)
class BrainPlan:
    """
    Execution plan produced by the brain.
    """

    plan_id: str
    task: str
    intent: BrainIntent
    agent_name: str | None = None
    tool_name: str | None = None
    strategy: str = "direct_agent"
    confidence: float = 0.0
    steps: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task": self.task,
            "intent": self.intent.value,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 4),
            "steps": list(self.steps),
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


# ============================================================================
# BRAIN EXECUTION
# ============================================================================


@dataclass(slots=True)
class BrainExecution:
    """
    Complete brain execution record.
    """

    execution_id: str
    task: str
    status: BrainStatus
    response: Any = None
    error: str | None = None

    intent: BrainIntent = BrainIntent.UNKNOWN

    plan: BrainPlan | None = None
    intent_result: BrainIntentResult | None = None

    agent_result: Any = None
    tool_result: Any = None

    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None

    latency_ms: float = 0.0

    observations: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def observe(
        self,
        event: str,
        **data: Any,
    ) -> None:
        """
        Add an execution observation.
        """

        self.observations.append(
            {
                "event": normalize_text(event),
                "data": dict(data),
                "timestamp": utc_now(),
                "sequence": len(self.observations) + 1,
            }
        )

    def complete(
        self,
        response: Any,
        latency_ms: float,
    ) -> None:
        """
        Mark execution as completed.
        """

        self.response = response
        self.status = BrainStatus.COMPLETED
        self.completed_at = utc_now()
        self.latency_ms = round(latency_ms, 4)

    def fail(
        self,
        error: str,
        latency_ms: float,
    ) -> None:
        """
        Mark execution as failed.
        """

        self.error = normalize_text(error)
        self.status = BrainStatus.FAILED
        self.completed_at = utc_now()
        self.latency_ms = round(latency_ms, 4)

    @property
    def success(self) -> bool:
        return self.status == BrainStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task": self.task,
            "status": self.status.value,
            "success": self.success,
            "response": self.response,
            "error": self.error,
            "intent": self.intent.value,
            "plan": (
                self.plan.to_dict()
                if self.plan is not None
                else None
            ),
            "intent_result": (
                self.intent_result.to_dict()
                if self.intent_result is not None
                else None
            ),
            "agent_result": object_to_dict(
                self.agent_result
            ),
            "tool_result": object_to_dict(
                self.tool_result
            ),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "observations": list(self.observations),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# ZAI BRAIN
# ============================================================================


class ZAIBrain:
    """
    Central intelligence coordinator.

    The brain connects the existing ZAI platform components:

        AgentRuntime
        AgentManager
        AgentRegistry
        AgentRouter
        ToolManager
        ToolRegistry

    Components are optional at construction time so the class remains
    import-safe during incremental platform development.
    """

    name = "ZAIBrain"
    version = ZAI_BRAIN_VERSION

    GENERAL_KEYWORDS = {
        "halo",
        "hai",
        "hello",
        "bantu",
        "tolong",
        "jelaskan",
        "apa",
        "bagaimana",
        "kenapa",
        "buat",
        "bisa",
        "zai",
    }

    CODING_KEYWORDS = {
        "code",
        "coding",
        "program",
        "programming",
        "python",
        "dart",
        "flutter",
        "javascript",
        "typescript",
        "java",
        "c++",
        "c#",
        "php",
        "sql",
        "html",
        "css",
        "debug",
        "bug",
        "error",
        "compile",
        "compiler",
        "syntax",
        "function",
        "class",
        "api",
        "backend",
        "frontend",
        "repository",
        "git",
        "github",
    }

    RESEARCH_KEYWORDS = {
        "riset",
        "research",
        "penelitian",
        "teliti",
        "cari informasi",
        "cari sumber",
        "sumber",
        "referensi",
        "literatur",
        "analisis",
        "bandingkan",
        "compare",
        "studi",
        "data",
        "evidence",
        "citation",
        "citation",
        "paper",
        "jurnal",
    }

    SYSTEM_KEYWORDS = {
        "system",
        "sistem",
        "windows",
        "linux",
        "server",
        "komputer",
        "cpu",
        "ram",
        "disk",
        "storage",
        "process",
        "service",
        "port",
        "network",
        "wifi",
        "shutdown",
        "restart",
        "powershell",
        "terminal",
        "environment",
        "health",
        "status",
    }

    TOOL_KEYWORDS = {
        "gunakan tool",
        "pakai tool",
        "jalankan tool",
        "execute tool",
        "tool",
        "command",
        "jalankan",
        "eksekusi",
    }

    def __init__(
        self,
        *,
        runtime: Any = None,
        agent_manager: Any = None,
        agent_registry: Any = None,
        agent_router: Any = None,
        tool_manager: Any = None,
        tool_registry: Any = None,
        orchestrator: Any = None,
        history_limit: int = 100,
        minimum_confidence: float = 0.15,
    ) -> None:

        self.runtime = runtime
        self.agent_manager = agent_manager
        self.agent_registry = agent_registry
        self.agent_router = agent_router
        self.tool_manager = tool_manager
        self.tool_registry = tool_registry
        self.orchestrator = orchestrator

        self.history_limit = max(
            1,
            safe_int(history_limit, 100),
        )

        self.minimum_confidence = max(
            0.0,
            min(
                1.0,
                safe_float(
                    minimum_confidence,
                    0.15,
                ),
            ),
        )

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.denied_count = 0

        self.intent_count = 0
        self.plan_count = 0
        self.agent_execution_count = 0
        self.tool_execution_count = 0

        self.total_latency_ms = 0.0

        self._history: list[BrainExecution] = []

    # ========================================================================
    # INFORMATION
    # ========================================================================

    def info(self) -> dict[str, Any]:
        """
        Return brain information.
        """

        return {
            "brain": self.name,
            "version": self.version,
            "status": "READY",
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "denied_count": self.denied_count,
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency_ms,
            "intent_count": self.intent_count,
            "plan_count": self.plan_count,
            "agent_execution_count": self.agent_execution_count,
            "tool_execution_count": self.tool_execution_count,
            "history_size": len(self._history),
            "history_limit": self.history_limit,
            "minimum_confidence": self.minimum_confidence,
            "components": self.component_status(),
        }

    @property
    def success_rate(self) -> float:
        """
        Success percentage.
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

    @property
    def failure_rate(self) -> float:
        """
        Failure percentage.
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

    @property
    def average_latency_ms(self) -> float:
        """
        Average execution latency.
        """

        if self.execution_count <= 0:
            return 0.0

        return round(
            self.total_latency_ms
            / self.execution_count,
            4,
        )

    # ========================================================================
    # COMPONENT STATUS
    # ========================================================================

    def component_status(self) -> dict[str, Any]:
        """
        Report availability of connected ZAI components.
        """

        components = {
            "runtime": self.runtime,
            "agent_manager": self.agent_manager,
            "agent_registry": self.agent_registry,
            "agent_router": self.agent_router,
            "tool_manager": self.tool_manager,
            "tool_registry": self.tool_registry,
            "orchestrator": self.orchestrator,
        }

        result: dict[str, Any] = {}

        for name, component in components.items():
            result[name] = {
                "available": component is not None,
                "type": (
                    type(component).__name__
                    if component is not None
                    else None
                ),
            }

        return result

    def health(self) -> dict[str, Any]:
        """
        Health information for the brain.
        """

        problems: list[str] = []

        if self.execution_count > 0:
            if self.failure_rate >= 50.0:
                problems.append(
                    "failure_rate_high"
                )

        return {
            "brain": self.name,
            "version": self.version,
            "status": (
                "DEGRADED"
                if problems
                else "HEALTHY"
            ),
            "problems": problems,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "denied_count": self.denied_count,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "average_latency_ms": self.average_latency_ms,
            "components": self.component_status(),
        }

    # ========================================================================
    # INTENT ANALYSIS
    # ========================================================================

    def analyze_intent(
        self,
        task: str,
    ) -> BrainIntentResult:
        """
        Analyze a task using deterministic local signals.

        This is intentionally lightweight.

        A future LLM intent classifier can be inserted above or below
        this layer without changing the public Brain API.
        """

        normalized = safe_lower(task)

        if not normalized:
            return BrainIntentResult(
                intent=BrainIntent.UNKNOWN,
                confidence=0.0,
                reasons=["empty_task"],
                task_type="unknown",
                requires_agent=False,
            )

        self.intent_count += 1

        keywords = self._extract_keywords(
            normalized
        )

        coding_score = self._keyword_score(
            normalized,
            self.CODING_KEYWORDS,
        )

        research_score = self._keyword_score(
            normalized,
            self.RESEARCH_KEYWORDS,
        )

        system_score = self._keyword_score(
            normalized,
            self.SYSTEM_KEYWORDS,
        )

        tool_score = self._keyword_score(
            normalized,
            self.TOOL_KEYWORDS,
        )

        general_score = self._keyword_score(
            normalized,
            self.GENERAL_KEYWORDS,
        )

        scores = {
            BrainIntent.CODING: coding_score,
            BrainIntent.RESEARCH: research_score,
            BrainIntent.SYSTEM: system_score,
            BrainIntent.TOOL: tool_score,
            BrainIntent.GENERAL: general_score,
        }

        best_intent = max(
            scores,
            key=scores.get,
        )

        best_score = scores[best_intent]

        reasons: list[str] = []

        if coding_score > 0:
            reasons.append(
                "coding_signal_detected"
            )

        if research_score > 0:
            reasons.append(
                "research_signal_detected"
            )

        if system_score > 0:
            reasons.append(
                "system_signal_detected"
            )

        if tool_score > 0:
            reasons.append(
                "tool_signal_detected"
            )

        if general_score > 0:
            reasons.append(
                "general_signal_detected"
            )

        phrase_bonus = 0.0

        if (
            "cari informasi" in normalized
            or "cari sumber" in normalized
        ):
            phrase_bonus += 0.15

        if (
            "perbaiki kode" in normalized
            or "perbaiki code" in normalized
        ):
            phrase_bonus += 0.20

        if (
            "cek sistem" in normalized
            or "status sistem" in normalized
        ):
            phrase_bonus += 0.20

        if phrase_bonus > 0:
            best_score += phrase_bonus
            reasons.append(
                "phrase_match_bonus"
            )

        confidence = max(
            0.0,
            min(
                1.0,
                best_score,
            ),
        )

        if best_score <= 0:
            intent = BrainIntent.GENERAL
            confidence = 0.20
            reasons.append(
                "general_fallback"
            )
        else:
            intent = best_intent

        requires_tool = (
            intent == BrainIntent.TOOL
        )

        task_type = intent.value

        return BrainIntentResult(
            intent=intent,
            confidence=confidence,
            reasons=reasons,
            keywords=keywords,
            entities=self._extract_entities(
                normalized
            ),
            requires_agent=True,
            requires_tool=requires_tool,
            task_type=task_type,
        )

    # ========================================================================
    # KEYWORD ANALYSIS
    # ========================================================================

    @staticmethod
    def _keyword_score(
        task: str,
        vocabulary: Iterable[str],
    ) -> float:
        """
        Calculate a deterministic keyword score.
        """

        matches = 0

        for keyword in vocabulary:
            if keyword in task:
                matches += 1

        if matches <= 0:
            return 0.0

        return min(
            1.0,
            0.20
            + (
                matches
                * 0.15
            ),
        )

    @staticmethod
    def _extract_keywords(
        task: str,
    ) -> list[str]:
        """
        Extract useful words from task text.
        """

        raw_words = task.split()

        result: list[str] = []

        stop_words = {
            "yang",
            "dan",
            "atau",
            "untuk",
            "dengan",
            "dari",
            "ke",
            "di",
            "ini",
            "itu",
            "saya",
            "aku",
            "anda",
            "kamu",
            "tolong",
            "bisa",
            "bantu",
        }

        for word in raw_words:
            cleaned = (
                word.strip(
                    ".,!?;:\"'`()[]{}<>"
                )
            )

            if (
                len(cleaned) < 3
                or cleaned in stop_words
            ):
                continue

            if cleaned not in result:
                result.append(cleaned)

        return result[:50]

    @staticmethod
    def _extract_entities(
        task: str,
    ) -> list[str]:
        """
        Basic entity detection.

        This intentionally remains conservative.
        """

        entities: list[str] = []

        known_entities = {
            "zai",
            "ollama",
            "python",
            "flutter",
            "dart",
            "windows",
            "linux",
            "github",
            "fastapi",
            "openai",
        }

        for entity in known_entities:
            if entity in task:
                entities.append(entity)

        return entities

    # ========================================================================
    # AGENT DISCOVERY
    # ========================================================================

    def available_agents(self) -> list[str]:
        """
        Return names of currently available agents.

        Multiple registry implementations are supported.
        """

        candidates = [
            self.agent_registry,
            self.agent_manager,
            self.runtime,
            self.orchestrator,
        ]

        for component in candidates:
            if component is None:
                continue

            names_method = getattr(
                component,
                "names",
                None,
            )

            if callable(names_method):
                try:
                    names = names_method()

                    if names:
                        return [
                            str(name)
                            for name in names
                        ]
                except Exception:
                    pass

            info_method = getattr(
                component,
                "info",
                None,
            )

            if callable(info_method):
                try:
                    data = info_method()

                    registry = (
                        data.get("registry", {})
                        if isinstance(data, dict)
                        else {}
                    )

                    names = registry.get(
                        "agent_names",
                        [],
                    )

                    if names:
                        return [
                            str(name)
                            for name in names
                        ]
                except Exception:
                    pass

        return []

    # ========================================================================
    # AGENT SELECTION
    # ========================================================================

    def select_agent(
        self,
        task: str,
        intent_result: BrainIntentResult | None = None,
    ) -> tuple[str | None, float, list[str]]:
        """
        Select an agent using existing routing infrastructure when possible.

        Falls back to deterministic intent mapping.
        """

        if intent_result is None:
            intent_result = self.analyze_intent(
                task
            )

        reasons: list[str] = []

        available = self.available_agents()

        # --------------------------------------------------------------------
        # Try AgentRouter
        # --------------------------------------------------------------------

        if self.agent_router is not None:
            route_method = getattr(
                self.agent_router,
                "route",
                None,
            )

            if callable(route_method):
                try:
                    agents = self._resolve_agent_objects(
                        available
                    )

                    if agents:
                        route = route_method(
                            task,
                            agents,
                        )

                        route_data = object_to_dict(
                            route
                        )

                        selected = (
                            route_data.get(
                                "selected_agent"
                            )
                        )

                        confidence = safe_float(
                            route_data.get(
                                "confidence"
                            ),
                            0.0,
                        )

                        if selected:
                            reasons.append(
                                "agent_router"
                            )

                            return (
                                str(selected),
                                confidence,
                                reasons,
                            )
                except Exception as exc:
                    reasons.append(
                        "agent_router_fallback"
                    )

        # --------------------------------------------------------------------
        # Deterministic mapping
        # --------------------------------------------------------------------

        mapping = {
            BrainIntent.CODING:
                "coding_agent",

            BrainIntent.RESEARCH:
                "research_agent",

            BrainIntent.SYSTEM:
                "system_agent",

            BrainIntent.GENERAL:
                "general_agent",
        }

        preferred = mapping.get(
            intent_result.intent
        )

        if (
            preferred
            and preferred in available
        ):
            reasons.append(
                "intent_mapping"
            )

            return (
                preferred,
                max(
                    0.25,
                    intent_result.confidence,
                ),
                reasons,
            )

        # --------------------------------------------------------------------
        # Fallback general agent
        # --------------------------------------------------------------------

        if "general_agent" in available:
            reasons.append(
                "general_agent_fallback"
            )

            return (
                "general_agent",
                0.20,
                reasons,
            )

        if available:
            reasons.append(
                "first_available_agent"
            )

            return (
                available[0],
                0.10,
                reasons,
            )

        return (
            None,
            0.0,
            reasons,
        )

    # ========================================================================
    # AGENT OBJECT RESOLUTION
    # ========================================================================

    def _resolve_agent_objects(
        self,
        names: list[str],
    ) -> list[Any]:
        """
        Resolve agent names to agent instances.
        """

        result: list[Any] = []

        registry = self.agent_registry

        if registry is not None:
            get_method = getattr(
                registry,
                "get",
                None,
            )

            if callable(get_method):
                for name in names:
                    try:
                        result.append(
                            get_method(name)
                        )
                    except Exception:
                        continue

                if result:
                    return result

        manager = self.agent_manager

        if manager is not None:
            get_method = getattr(
                manager,
                "get",
                None,
            )

            if callable(get_method):
                for name in names:
                    try:
                        result.append(
                            get_method(name)
                        )
                    except Exception:
                        continue

        return result

    # ========================================================================
    # PLAN CREATION
    # ========================================================================

    def create_plan(
        self,
        task: str,
        intent_result: BrainIntentResult | None = None,
    ) -> BrainPlan:
        """
        Build an execution plan.
        """

        if intent_result is None:
            intent_result = self.analyze_intent(
                task
            )

        self.plan_count += 1

        agent_name, confidence, reasons = (
            self.select_agent(
                task,
                intent_result,
            )
        )

        steps: list[str] = [
            "analyze_intent",
            "classify_task",
            "select_agent",
        ]

        strategy = "direct_agent"

        if intent_result.requires_tool:
            strategy = "tool_or_agent"
            steps.append(
                "evaluate_tool_requirement"
            )

        steps.extend(
            [
                "execute",
                "validate_result",
                "synthesize_response",
            ]
        )

        return BrainPlan(
            plan_id=str(uuid4()),
            task=task,
            intent=intent_result.intent,
            agent_name=agent_name,
            strategy=strategy,
            confidence=max(
                confidence,
                intent_result.confidence,
            ),
            steps=steps,
            reasons=reasons,
            metadata={
                "task_type":
                    intent_result.task_type,

                "intent_confidence":
                    intent_result.confidence,

                "available_agents":
                    self.available_agents(),
            },
        )

    # ========================================================================
    # TOOL DISCOVERY
    # ========================================================================

    def available_tools(self) -> list[str]:
        """
        Return currently available tools.
        """

        candidates = [
            self.tool_registry,
            self.tool_manager,
        ]

        for component in candidates:
            if component is None:
                continue

            names_method = getattr(
                component,
                "names",
                None,
            )

            if callable(names_method):
                try:
                    names = names_method()

                    if names:
                        return [
                            str(name)
                            for name in names
                        ]
                except Exception:
                    pass

            list_method = getattr(
                component,
                "list_tools",
                None,
            )

            if callable(list_method):
                try:
                    tools = list_method()

                    result: list[str] = []

                    for tool in tools:
                        data = object_to_dict(
                            tool
                        )

                        name = data.get(
                            "name"
                        )

                        if name:
                            result.append(
                                str(name)
                            )

                    if result:
                        return result
                except Exception:
                    pass

        return []

    # ========================================================================
    # TOOL SELECTION
    # ========================================================================

    def select_tool(
        self,
        task: str,
        intent_result: BrainIntentResult | None = None,
    ) -> str | None:
        """
        Select an explicit tool if the task strongly indicates one.

        Current version intentionally avoids guessing arbitrary tools.

        This is important for safety and deterministic behavior.
        """

        if intent_result is None:
            intent_result = self.analyze_intent(
                task
            )

        if not intent_result.requires_tool:
            return None

        tools = self.available_tools()

        if not tools:
            return None

        normalized = safe_lower(task)

        # Exact tool mention.
        for tool_name in tools:
            if safe_lower(tool_name) in normalized:
                return tool_name

        # Common deterministic aliases.
        aliases = {
            "calculator": [
                "hitung",
                "kalkulator",
                "calculate",
            ],
            "search": [
                "cari",
                "search",
                "pencarian",
            ],
            "system": [
                "system",
                "sistem",
            ],
        }

        for tool_name in tools:
            lower_name = safe_lower(
                tool_name
            )

            aliases_for_tool = aliases.get(
                lower_name,
                [],
            )

            for alias in aliases_for_tool:
                if alias in normalized:
                    return tool_name

        return None

    # ========================================================================
    # AGENT EXECUTION
    # ========================================================================

    async def _execute_agent(
        self,
        agent_name: str,
        task: str,
        execution: BrainExecution,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an agent through the best available runtime interface.
        """

        self.agent_execution_count += 1

        execution.observe(
            "agent_execution_started",
            agent=agent_name,
        )

        # --------------------------------------------------------------------
        # AgentRuntime
        # --------------------------------------------------------------------

        if self.runtime is not None:
            execute_method = getattr(
                self.runtime,
                "execute",
                None,
            )

            if callable(execute_method):
                try:
                    result = execute_method(
                        agent_name,
                        task,
                        **kwargs,
                    )

                    if isawaitable(result):
                        result = await result

                    execution.observe(
                        "agent_execution_completed",
                        agent=agent_name,
                    )

                    return result

                except TypeError:
                    # Compatibility fallback for runtimes
                    # that use keyword parameters.
                    try:
                        result = execute_method(
                            agent_name=agent_name,
                            task=task,
                            **kwargs,
                        )

                        if isawaitable(result):
                            result = await result

                        execution.observe(
                            "agent_execution_completed",
                            agent=agent_name,
                        )

                        return result

                    except Exception:
                        raise

        # --------------------------------------------------------------------
        # AgentManager
        # --------------------------------------------------------------------

        if self.agent_manager is not None:
            execute_method = getattr(
                self.agent_manager,
                "execute",
                None,
            )

            if callable(execute_method):
                try:
                    result = execute_method(
                        agent_name,
                        task,
                        **kwargs,
                    )

                    if isawaitable(result):
                        result = await result

                    execution.observe(
                        "agent_manager_execution_completed",
                        agent=agent_name,
                    )

                    return result

                except TypeError:
                    result = execute_method(
                        agent_name=agent_name,
                        task=task,
                        **kwargs,
                    )

                    if isawaitable(result):
                        result = await result

                    return result

        # --------------------------------------------------------------------
        # Registry direct execution
        # --------------------------------------------------------------------

        agent = None

        if self.agent_registry is not None:
            get_method = getattr(
                self.agent_registry,
                "get",
                None,
            )

            if callable(get_method):
                agent = get_method(
                    agent_name
                )

        if agent is None:
            raise RuntimeError(
                f"Agent '{agent_name}' tidak tersedia."
            )

        execute_method = getattr(
            agent,
            "execute",
            None,
        )

        if not callable(execute_method):
            raise RuntimeError(
                f"Agent '{agent_name}' tidak memiliki execute()."
            )

        result = execute_method(
            task,
            **kwargs,
        )

        if isawaitable(result):
            result = await result

        return result

    # ========================================================================
    # TOOL EXECUTION
    # ========================================================================

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        execution: BrainExecution,
    ) -> Any:
        """
        Execute a tool through ToolManager or ToolRegistry.
        """

        self.tool_execution_count += 1

        execution.observe(
            "tool_execution_started",
            tool=tool_name,
        )

        candidates = [
            self.tool_manager,
            self.tool_registry,
        ]

        for component in candidates:
            if component is None:
                continue

            execute_method = getattr(
                component,
                "execute",
                None,
            )

            if not callable(execute_method):
                continue

            try:
                result = execute_method(
                    tool_name,
                    arguments=arguments,
                )

                if isawaitable(result):
                    result = await result

                execution.observe(
                    "tool_execution_completed",
                    tool=tool_name,
                )

                return result

            except TypeError:
                try:
                    result = execute_method(
                        tool_name,
                        **arguments,
                    )

                    if isawaitable(result):
                        result = await result

                    execution.observe(
                        "tool_execution_completed",
                        tool=tool_name,
                    )

                    return result

                except Exception:
                    raise

        raise RuntimeError(
            f"Tool '{tool_name}' tidak tersedia."
        )

    # ========================================================================
    # RESULT VALIDATION
    # ========================================================================

    def validate_result(
        self,
        result: Any,
    ) -> tuple[bool, str | None]:
        """
        Validate an agent/tool result.

        Supports the existing ZAI result structures.
        """

        if result is None:
            return (
                False,
                "Execution menghasilkan result kosong.",
            )

        data = object_to_dict(
            result
        )

        if not data:
            return (
                True,
                None,
            )

        status = safe_lower(
            data.get(
                "status",
                "",
            )
        )

        success = data.get(
            "success",
            None,
        )

        if success is False:
            return (
                False,
                normalize_text(
                    data.get(
                        "error",
                        "Execution gagal.",
                    )
                ),
            )

        if status in {
            "failed",
            "failure",
            "error",
            "denied",
            "blocked",
        }:
            return (
                False,
                normalize_text(
                    data.get(
                        "error",
                        "Execution gagal.",
                    )
                ),
            )

        return (
            True,
            None,
        )

    # ========================================================================
    # RESPONSE SYNTHESIS
    # ========================================================================

    def synthesize_response(
        self,
        execution: BrainExecution,
    ) -> Any:
        """
        Convert raw execution result into a clean response.

        Existing AgentResult/ToolResult response fields are preserved.
        """

        if execution.agent_result is not None:
            data = object_to_dict(
                execution.agent_result
            )

            if "response" in data:
                return data["response"]

            if "data" in data:
                return data["data"]

        if execution.tool_result is not None:
            data = object_to_dict(
                execution.tool_result
            )

            if "response" in data:
                return data["response"]

            if "data" in data:
                return data["data"]

        if execution.response is not None:
            return execution.response

        return (
            "ZAI berhasil menjalankan task, "
            "tetapi tidak menerima response."
        )

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: str,
        *,
        tool_arguments: dict[str, Any] | None = None,
        force_agent: str | None = None,
        force_tool: str | None = None,
        **kwargs: Any,
    ) -> BrainExecution:
        """
        Execute a task through the ZAI Brain.

        Parameters
        ----------
        task:
            User task.

        tool_arguments:
            Explicit arguments for a tool.

        force_agent:
            Force a specific agent.

        force_tool:
            Force a specific tool.

        kwargs:
            Additional runtime/agent arguments.
        """

        started_counter = perf_counter()

        self.execution_count += 1

        normalized_task = normalize_text(
            task
        )

        execution = BrainExecution(
            execution_id=str(uuid4()),
            task=normalized_task,
            status=BrainStatus.ANALYZING,
        )

        execution.metadata.update(
            {
                "brain": self.name,
                "brain_version": self.version,
                "task_length": len(
                    normalized_task
                ),
            }
        )

        self._remember(
            execution
        )

        execution.observe(
            "brain_execution_started",
            task_length=len(
                normalized_task
            ),
        )

        try:
            # ---------------------------------------------------------------
            # Empty task
            # ---------------------------------------------------------------

            if not normalized_task:
                self.denied_count += 1

                execution.status = (
                    BrainStatus.DENIED
                )

                execution.error = (
                    "Task tidak boleh kosong."
                )

                execution.completed_at = utc_now()

                execution.latency_ms = round(
                    (
                        perf_counter()
                        - started_counter
                    )
                    * 1000,
                    4,
                )

                self.total_latency_ms += (
                    execution.latency_ms
                )

                return execution

            # ---------------------------------------------------------------
            # Intent
            # ---------------------------------------------------------------

            intent_result = (
                self.analyze_intent(
                    normalized_task
                )
            )

            execution.intent_result = (
                intent_result
            )

            execution.intent = (
                intent_result.intent
            )

            execution.observe(
                "intent_analyzed",
                intent=intent_result.intent.value,
                confidence=(
                    intent_result.confidence
                ),
            )

            # ---------------------------------------------------------------
            # Plan
            # ---------------------------------------------------------------

            execution.status = (
                BrainStatus.PLANNING
            )

            plan = self.create_plan(
                normalized_task,
                intent_result,
            )

            execution.plan = plan

            execution.observe(
                "plan_created",
                plan_id=plan.plan_id,
                strategy=plan.strategy,
                agent=plan.agent_name,
            )

            # ---------------------------------------------------------------
            # Forced tool
            # ---------------------------------------------------------------

            selected_tool = (
                force_tool
                or self.select_tool(
                    normalized_task,
                    intent_result,
                )
            )

            if selected_tool:
                execution.status = (
                    BrainStatus.EXECUTING
                )

                execution.observe(
                    "tool_selected",
                    tool=selected_tool,
                )

                arguments = (
                    dict(
                        tool_arguments
                        or {}
                    )
                )

                tool_result = (
                    await self._execute_tool(
                        selected_tool,
                        arguments,
                        execution,
                    )
                )

                execution.tool_result = (
                    tool_result
                )

                valid, error = (
                    self.validate_result(
                        tool_result
                    )
                )

                execution.observe(
                    "tool_result_validated",
                    valid=valid,
                )

                if not valid:
                    raise RuntimeError(
                        error
                        or "Tool execution gagal."
                    )

                execution.response = (
                    self.synthesize_response(
                        execution
                    )
                )

            else:
                # -----------------------------------------------------------
                # Agent
                # -----------------------------------------------------------

                selected_agent = (
                    force_agent
                    or plan.agent_name
                )

                if not selected_agent:
                    raise RuntimeError(
                        "Tidak ada agent yang dapat dipilih."
                    )

                execution.status = (
                    BrainStatus.EXECUTING
                )

                execution.observe(
                    "agent_selected",
                    agent=selected_agent,
                )

                agent_result = (
                    await self._execute_agent(
                        selected_agent,
                        normalized_task,
                        execution,
                        **kwargs,
                    )
                )

                execution.agent_result = (
                    agent_result
                )

                valid, error = (
                    self.validate_result(
                        agent_result
                    )
                )

                execution.status = (
                    BrainStatus.VALIDATING
                )

                execution.observe(
                    "agent_result_validated",
                    valid=valid,
                )

                if not valid:
                    raise RuntimeError(
                        error
                        or "Agent execution gagal."
                    )

                execution.response = (
                    self.synthesize_response(
                        execution
                    )
                )

            # ---------------------------------------------------------------
            # Complete
            # ---------------------------------------------------------------

            execution.status = (
                BrainStatus.COMPLETED
            )

            execution.completed_at = (
                utc_now()
            )

            execution.latency_ms = round(
                (
                    perf_counter()
                    - started_counter
                )
                * 1000,
                4,
            )

            self.total_latency_ms += (
                execution.latency_ms
            )

            self.success_count += 1

            execution.metadata.update(
                {
                    "success": True,
                    "success_rate":
                        self.success_rate,
                    "runtime_latency_ms":
                        execution.latency_ms,
                    "brain_execution_count":
                        self.execution_count,
                    "brain_success_count":
                        self.success_count,
                    "brain_failure_count":
                        self.failure_count,
                }
            )

            execution.observe(
                "brain_execution_completed",
                success=True,
                latency_ms=(
                    execution.latency_ms
                ),
            )

            return execution

        except Exception as exc:
            # ---------------------------------------------------------------
            # Failure
            # ---------------------------------------------------------------

            self.failure_count += 1

            execution.fail(
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                (
                    (
                        perf_counter()
                        - started_counter
                    )
                    * 1000
                ),
            )

            self.total_latency_ms += (
                execution.latency_ms
            )

            execution.metadata.update(
                {
                    "success": False,
                    "success_rate":
                        self.success_rate,
                    "brain_execution_count":
                        self.execution_count,
                    "brain_success_count":
                        self.success_count,
                    "brain_failure_count":
                        self.failure_count,
                }
            )

            execution.observe(
                "brain_execution_failed",
                error=execution.error,
                latency_ms=(
                    execution.latency_ms
                ),
            )

            return execution

    # ========================================================================
    # SYNCHRONOUS API
    # ========================================================================

    def execute_sync(
        self,
        task: str,
        **kwargs: Any,
    ) -> BrainExecution:
        """
        Synchronous helper.

        Intended for CLI/tests where no event loop is active.
        """

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "execute_sync() tidak dapat "
                "dipanggil dari running event loop. "
                "Gunakan await execute()."
            )

        return asyncio.run(
            self.execute(
                task,
                **kwargs,
            )
        )

    # ========================================================================
    # HISTORY
    # ========================================================================

    def _remember(
        self,
        execution: BrainExecution,
    ) -> None:
        """
        Store execution history.
        """

        self._history.append(
            execution
        )

        overflow = (
            len(self._history)
            - self.history_limit
        )

        if overflow > 0:
            del self._history[
                :overflow
            ]

    def history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return execution history.
        """

        if limit is None:
            limit = self.history_limit

        limit = max(
            0,
            safe_int(
                limit,
                self.history_limit,
            ),
        )

        if limit == 0:
            return []

        items = self._history[
            -limit:
        ]

        return [
            item.to_dict()
            for item in items
        ]

    def clear_history(self) -> None:
        """
        Clear brain execution history.
        """

        self._history.clear()

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Return detailed statistics.
        """

        intent_distribution: dict[
            str,
            int,
        ] = {}

        for execution in self._history:
            key = execution.intent.value

            intent_distribution[key] = (
                intent_distribution.get(
                    key,
                    0,
                )
                + 1
            )

        return {
            "brain": self.name,
            "version": self.version,
            "execution_count":
                self.execution_count,
            "success_count":
                self.success_count,
            "failure_count":
                self.failure_count,
            "denied_count":
                self.denied_count,
            "success_rate":
                self.success_rate,
            "failure_rate":
                self.failure_rate,
            "average_latency_ms":
                self.average_latency_ms,
            "total_latency_ms":
                round(
                    self.total_latency_ms,
                    4,
                ),
            "intent_count":
                self.intent_count,
            "plan_count":
                self.plan_count,
            "agent_execution_count":
                self.agent_execution_count,
            "tool_execution_count":
                self.tool_execution_count,
            "history_size":
                len(self._history),
            "intent_distribution":
                intent_distribution,
        }

    # ========================================================================
    # RESET
    # ========================================================================

    def reset_statistics(
        self,
        *,
        clear_history: bool = False,
    ) -> None:
        """
        Reset runtime counters.
        """

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.denied_count = 0

        self.intent_count = 0
        self.plan_count = 0
        self.agent_execution_count = 0
        self.tool_execution_count = 0

        self.total_latency_ms = 0.0

        if clear_history:
            self.clear_history()

    # ========================================================================
    # DIRECT REGISTRATION HELPERS
    # ========================================================================

    def attach_runtime(
        self,
        runtime: Any,
    ) -> "ZAIBrain":
        """
        Attach AgentRuntime.
        """

        self.runtime = runtime
        return self

    def attach_agent_manager(
        self,
        manager: Any,
    ) -> "ZAIBrain":
        """
        Attach AgentManager.
        """

        self.agent_manager = manager
        return self

    def attach_agent_registry(
        self,
        registry: Any,
    ) -> "ZAIBrain":
        """
        Attach AgentRegistry.
        """

        self.agent_registry = registry
        return self

    def attach_agent_router(
        self,
        router: Any,
    ) -> "ZAIBrain":
        """
        Attach AgentRouter.
        """

        self.agent_router = router
        return self

    def attach_tool_manager(
        self,
        manager: Any,
    ) -> "ZAIBrain":
        """
        Attach ToolManager.
        """

        self.tool_manager = manager
        return self

    def attach_tool_registry(
        self,
        registry: Any,
    ) -> "ZAIBrain":
        """
        Attach ToolRegistry.
        """

        self.tool_registry = registry
        return self

    def attach_orchestrator(
        self,
        orchestrator: Any,
    ) -> "ZAIBrain":
        """
        Attach AgentOrchestrator.
        """

        self.orchestrator = orchestrator
        return self

    # ========================================================================
    # BRAIN RESET / SHUTDOWN
    # ========================================================================

    def shutdown(self) -> dict[str, Any]:
        """
        Lightweight shutdown.

        The brain itself does not own the lifecycle of external
        components, therefore it only reports shutdown state.
        """

        return {
            "brain": self.name,
            "version": self.version,
            "status": "SHUTDOWN",
            "execution_count":
                self.execution_count,
            "history_size":
                len(self._history),
        }


# ============================================================================
# DEFAULT FACTORY
# ============================================================================


def create_brain(
    *,
    runtime: Any = None,
    agent_manager: Any = None,
    agent_registry: Any = None,
    agent_router: Any = None,
    tool_manager: Any = None,
    tool_registry: Any = None,
    orchestrator: Any = None,
) -> ZAIBrain:
    """
    Create a configured ZAI Brain.

    This factory makes future dependency injection easier.
    """

    return ZAIBrain(
        runtime=runtime,
        agent_manager=agent_manager,
        agent_registry=agent_registry,
        agent_router=agent_router,
        tool_manager=tool_manager,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
    )


__all__ = [
    "ZAI_BRAIN_VERSION",
    "BrainStatus",
    "BrainIntent",
    "BrainIntentResult",
    "BrainPlan",
    "BrainExecution",
    "ZAIBrain",
    "create_brain",
]
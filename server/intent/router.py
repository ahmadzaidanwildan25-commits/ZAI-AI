"""
Super ZAI Intent Router.

Routes classified intents to the appropriate subsystem.

The router does not contain the actual business logic of tools.
It produces a deterministic RouteDecision that can be consumed
by CognitiveOrchestrator / AIBrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RouteDecision:
    """
    Structured routing decision.
    """

    route: str
    intent: str
    confidence: float

    tool: Optional[str] = None

    arguments: Dict[str, Any] = field(
        default_factory=dict
    )

    handled_locally: bool = False

    requires_memory: bool = False
    requires_llm: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "route": self.route,
            "intent": self.intent,
            "confidence": self.confidence,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "handled_locally": self.handled_locally,
            "requires_memory": self.requires_memory,
            "requires_llm": self.requires_llm,
            "metadata": dict(self.metadata),
        }


class IntentRouter:
    """
    Deterministic intent routing layer.
    """

    VERSION = "0.11.0"

    TOOL_ROUTES = {
        "calculation": "calculator",
        "weather": "weather",
        "search": "search",
        "fetch": "fetch",
    }

    MEMORY_INTENTS = {
        "memory_save",
        "memory_recall",
        "memory_count",
        "memory_forget",
    }

    LOCAL_INTENTS = {
        "greeting",
        "identity",
        "help",
        "status",
        "conversation",
    }

    LLM_INTENTS = {
        "coding",
        "general",
    }

    def __init__(
        self,
        memory_manager: Any = None,
        response_engine: Any = None,
    ) -> None:

        self.memory_manager = memory_manager
        self.response_engine = response_engine

        self._route_count = 0

    # ============================================================
    # PUBLIC
    # ============================================================

    def route(
        self,
        intent_result: Any,
    ) -> RouteDecision:

        self._route_count += 1

        data = self._normalize_result(
            intent_result
        )

        intent = data["intent"]
        confidence = data["confidence"]
        entities = data["entities"]
        text = data["text"]

        # --------------------------------------------------------
        # TOOL ROUTES
        # --------------------------------------------------------

        if intent in self.TOOL_ROUTES:

            tool = self.TOOL_ROUTES[intent]

            arguments = self._tool_arguments(
                tool=tool,
                text=text,
                entities=entities,
            )

            return RouteDecision(
                route=tool,
                intent=intent,
                confidence=confidence,
                tool=tool,
                arguments=arguments,
                handled_locally=False,
                requires_memory=False,
                requires_llm=False,
                metadata={
                    "router": self.VERSION,
                    "tool_execution": True,
                },
            )

        # --------------------------------------------------------
        # MEMORY
        # --------------------------------------------------------

        if intent in self.MEMORY_INTENTS:

            return RouteDecision(
                route="memory",
                intent=intent,
                confidence=confidence,
                arguments=entities,
                handled_locally=False,
                requires_memory=True,
                requires_llm=False,
                metadata={
                    "router": self.VERSION,
                    "memory_operation": intent,
                },
            )

        # --------------------------------------------------------
        # LOCAL
        # --------------------------------------------------------

        if intent in self.LOCAL_INTENTS:

            return RouteDecision(
                route="local",
                intent=intent,
                confidence=confidence,
                arguments=entities,
                handled_locally=True,
                requires_memory=False,
                requires_llm=False,
                metadata={
                    "router": self.VERSION,
                    "fast_response": True,
                },
            )

        # --------------------------------------------------------
        # LLM
        # --------------------------------------------------------

        if intent in self.LLM_INTENTS:

            return RouteDecision(
                route="llm",
                intent=intent,
                confidence=confidence,
                arguments=entities,
                handled_locally=False,
                requires_memory=False,
                requires_llm=True,
                metadata={
                    "router": self.VERSION,
                    "llm_required": True,
                },
            )

        # --------------------------------------------------------
        # GENERAL FALLBACK
        # --------------------------------------------------------

        return RouteDecision(
            route="llm",
            intent=intent,
            confidence=confidence,
            arguments=entities,
            handled_locally=False,
            requires_memory=False,
            requires_llm=True,
            metadata={
                "router": self.VERSION,
                "fallback": True,
            },
        )

    def decide(
        self,
        intent_result: Any,
    ) -> Dict[str, Any]:

        return self.route(
            intent_result
        ).to_dict()

    # ============================================================
    # ARGUMENT BUILDING
    # ============================================================

    def _tool_arguments(
        self,
        tool: str,
        text: str,
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:

        if tool == "calculator":

            return {
                "query": entities.get(
                    "expression",
                    text,
                ),
                "expression": entities.get(
                    "expression"
                ),
            }

        if tool == "weather":

            return {
                "query": entities.get(
                    "city",
                    text,
                ),
                "city": entities.get(
                    "city"
                ),
            }

        if tool == "search":

            return {
                "query": entities.get(
                    "query",
                    text,
                ),
            }

        if tool == "fetch":

            return {
                "url": entities.get(
                    "url"
                ),
            }

        return {
            "query": text,
        }

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_result(
        result: Any,
    ) -> Dict[str, Any]:

        if hasattr(
            result,
            "to_dict",
        ):
            result = result.to_dict()

        if not isinstance(
            result,
            dict,
        ):
            result = {}

        return {
            "intent": str(
                result.get(
                    "intent",
                    "general",
                )
            ),
            "confidence": float(
                result.get(
                    "confidence",
                    0.0,
                )
            ),
            "text": str(
                result.get(
                    "text",
                    "",
                )
            ),
            "entities": dict(
                result.get(
                    "entities",
                    {},
                )
                or {}
            ),
        }

    # ============================================================
    # STATS
    # ============================================================

    def stats(self) -> Dict[str, Any]:

        return {
            "engine": "IntentRouter",
            "version": self.VERSION,
            "tool_routes": dict(
                self.TOOL_ROUTES
            ),
            "memory_intents": sorted(
                self.MEMORY_INTENTS
            ),
            "local_intents": sorted(
                self.LOCAL_INTENTS
            ),
            "llm_intents": sorted(
                self.LLM_INTENTS
            ),
            "routes_executed": self._route_count,
            "memory_manager": (
                self.memory_manager is not None
            ),
            "response_engine": (
                self.response_engine is not None
            ),
            "status": "READY",
        }


__all__ = [
    "IntentRouter",
    "RouteDecision",
]

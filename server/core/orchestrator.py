from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from intent.engine import get_intent_engine
from intent.router import IntentRouter
from memory.manager import MemoryManager
from response.engine import ResponseEngine
from core.tool_engine import get_tool_engine


@dataclass
class OrchestratorResult:

    handled: bool
    response: Optional[str]
    route: str
    intent: str
    confidence: float = 0.0
    metadata: dict = field(
        default_factory=dict
    )
    error: Optional[str] = None

    def to_dict(self) -> dict:

        return {
            "handled": self.handled,
            "response": self.response,
            "route": self.route,
            "intent": self.intent,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "error": self.error,
        }


class CognitiveOrchestrator:
    """
    Super ZAI Cognitive Pipeline.

    v1.0.0

    Pipeline:

        intent
          ↓
        routing
          ↓
        local / memory / tool / llm
          ↓
        response
          ↓
        fallback
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        intent_engine=None,
        router=None,
        response_engine=None,
        tool_engine=None,
    ):

        self.intent_engine = (
            intent_engine
            or get_intent_engine()
        )

        self.memory_manager = (
            MemoryManager()
        )

        self.router = (
            router
            or IntentRouter(
                self.memory_manager
            )
        )

        self.response_engine = (
            response_engine
            or ResponseEngine(
                self.memory_manager
            )
        )

        self.tool_engine = (
            tool_engine
            or get_tool_engine()
        )

    # ==========================================================
    # STATS
    # ==========================================================

    def stats(self) -> dict:

        return {
            "engine": (
                "CognitiveOrchestrator"
            ),
            "version": self.VERSION,
            "intent_engine": (
                type(
                    self.intent_engine
                ).__name__
            ),
            "router": (
                type(
                    self.router
                ).__name__
            ),
            "response_engine": (
                type(
                    self.response_engine
                ).__name__
            ),
            "tool_engine": (
                type(
                    self.tool_engine
                ).__name__
            ),
            "pipeline": [
                "intent",
                "routing",
                "tool_execution",
                "response",
                "fallback",
            ],
            "status": "READY",
        }

    # ==========================================================
    # HANDLE
    # ==========================================================

    def handle(
        self,
        message: str,
        location: Optional[dict] = None,
    ) -> OrchestratorResult:

        text = str(
            message or ""
        ).strip()

        if not text:

            return OrchestratorResult(
                handled=True,
                response=(
                    "Pesan kosong."
                ),
                route="local",
                intent="empty",
                confidence=1.0,
            )

        try:

            # --------------------------------------------------
            # INTENT
            # --------------------------------------------------

            intent_result = (
                self.intent_engine.analyze(
                    text
                )
            )

            intent = str(
                intent_result.get(
                    "intent",
                    "general",
                )
            )

            confidence = float(
                intent_result.get(
                    "confidence",
                    0.5,
                )
            )

            normalized_text = (
                intent_result.get(
                    "normalized_text",
                    text,
                )
            )

            # --------------------------------------------------
            # ROUTING
            # --------------------------------------------------

            try:

                routing = self.router.route(
                    text,
                    intent_result,
                )

            except TypeError:

                routing = self.router.route(
                    intent_result
                )

            route = str(
                routing.get(
                    "route",
                    "llm",
                )
            )

            metadata = dict(
                routing.get(
                    "metadata",
                    {},
                )
            )

            metadata.update(
                {
                    "orchestrator": (
                        self.VERSION
                    ),
                    "normalized_text": (
                        normalized_text
                    ),
                    "intent_confidence": (
                        confidence
                    ),
                    "high_confidence": (
                        confidence >= 0.90
                    ),
                }
            )

            # --------------------------------------------------
            # LOCAL / MEMORY RESPONSE
            # --------------------------------------------------

            if route in {
                "local",
                "memory",
            }:

                result = (
                    self.response_engine.handle(
                        intent,
                        text,
                    )
                )

                response = (
                    self._response_text(
                        result
                    )
                )

                if response:

                    metadata.update(
                        self._response_metadata(
                            result
                        )
                    )

                    return OrchestratorResult(
                        handled=True,
                        response=response,
                        route=route,
                        intent=intent,
                        confidence=confidence,
                        metadata=metadata,
                    )

            # --------------------------------------------------
            # TOOL EXECUTION
            # --------------------------------------------------

            if route in {
                "calculator",
                "weather",
                "search",
            }:

                tool_metadata = dict(
                    metadata
                )

                if location:
                    tool_metadata[
                        "location"
                    ] = location

                tool_result = (
                    self.tool_engine.execute(
                        route,
                        text,
                        tool_metadata,
                    )
                )

                metadata[
                    "tool"
                ] = tool_result.tool

                metadata[
                    "tool_success"
                ] = tool_result.success

                metadata[
                    "tool_data"
                ] = tool_result.data

                if tool_result.success:

                    return OrchestratorResult(
                        handled=True,
                        response=(
                            tool_result.response
                        ),
                        route=route,
                        intent=intent,
                        confidence=confidence,
                        metadata=metadata,
                    )

                # Tool gagal.
                # Jangan crash.
                # Kembalikan error terkontrol.

                return OrchestratorResult(
                    handled=True,
                    response=self._tool_error_message(
                        route,
                        tool_result.error,
                    ),
                    route=route,
                    intent=intent,
                    confidence=confidence,
                    metadata=metadata,
                    error=tool_result.error,
                )

            # --------------------------------------------------
            # LLM
            # --------------------------------------------------

            if route == "llm":

                return OrchestratorResult(
                    handled=False,
                    response=None,
                    route="llm",
                    intent=intent,
                    confidence=confidence,
                    metadata=metadata,
                )

            # --------------------------------------------------
            # FALLBACK
            # --------------------------------------------------

            return OrchestratorResult(
                handled=False,
                response=None,
                route="llm",
                intent=intent,
                confidence=confidence,
                metadata={
                    **metadata,
                    "fallback": True,
                },
            )

        except Exception as error:

            return OrchestratorResult(
                handled=False,
                response=None,
                route="error",
                intent="error",
                confidence=0.0,
                metadata={
                    "orchestrator": (
                        self.VERSION
                    ),
                },
                error=str(error),
            )

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _response_text(
        result: Any,
    ) -> Optional[str]:

        if result is None:
            return None

        if isinstance(
            result,
            str,
        ):
            return result

        if hasattr(
            result,
            "response",
        ):
            return result.response

        if isinstance(
            result,
            dict,
        ):
            return result.get(
                "response"
            )

        return None

    @staticmethod
    def _response_metadata(
        result: Any,
    ) -> dict:

        if result is None:
            return {}

        if isinstance(
            result,
            dict,
        ):
            metadata = result.get(
                "metadata",
                {},
            )

            return (
                metadata
                if isinstance(
                    metadata,
                    dict,
                )
                else {}
            )

        if hasattr(
            result,
            "metadata",
        ):

            metadata = result.metadata

            return (
                metadata
                if isinstance(
                    metadata,
                    dict,
                )
                else {}
            )

        return {}

    @staticmethod
    def _tool_error_message(
        route: str,
        error: Optional[str],
    ) -> str:

        if route == "weather":

            return (
                "Saya belum bisa mendapatkan "
                "data cuaca saat ini. "
                + (
                    error or ""
                )
            ).strip()

        if route == "search":

            return (
                "Saya belum bisa mendapatkan "
                "hasil pencarian saat ini. "
                + (
                    error or ""
                )
            ).strip()

        if route == "calculator":

            return (
                "Perhitungannya gagal. "
                + (
                    error or ""
                )
            ).strip()

        return (
            "Tool gagal dijalankan. "
            + (
                error or ""
            )
        ).strip()
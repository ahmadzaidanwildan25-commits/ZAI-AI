from __future__ import annotations

from typing import Any, Optional


# ============================================================
# SUPER ZAI
# COGNITIVE ORCHESTRATOR
# VERSION 0.9.2
#
# Pipeline:
#
# USER
#   ↓
# INTENT
#   ↓
# ROUTING
#   ↓
# LOCAL RESPONSE / TOOL EXECUTION / LLM
#   ↓
# RESPONSE
#
# Tool routes:
#   calculator
#   weather
#   search
# ============================================================


class CognitiveOrchestrator:
    """
    Main cognitive pipeline for Super ZAI.

    Responsibilities:
    1. Analyze user intent.
    2. Route the request.
    3. Execute local response when possible.
    4. Execute tools when required.
    5. Return structured result.
    6. Fall back safely when no handler exists.
    """

    VERSION = "0.9.2"

    def __init__(
        self,
        intent_engine,
        router,
        response_engine,
        tool_engine=None,
    ):
        self.intent_engine = intent_engine
        self.router = router
        self.response_engine = response_engine

        # ToolEngine is optional for backward compatibility.
        # If not injected, we lazily load the singleton.
        self.tool_engine = tool_engine

    # ========================================================
    # TOOL ENGINE
    # ========================================================

    def _get_tool_engine(self):
        """
        Lazily resolve ToolEngine.

        This keeps older code compatible:
            CognitiveOrchestrator(i, r, e)

        while allowing:
            CognitiveOrchestrator(i, r, e, t)
        """

        if self.tool_engine is not None:
            return self.tool_engine

        try:
            from core.tool_engine import get_tool_engine

            self.tool_engine = get_tool_engine()
            return self.tool_engine

        except Exception:
            return None

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> dict[str, Any]:
        return {
            "engine": "CognitiveOrchestrator",
            "version": self.VERSION,
            "intent_engine": type(self.intent_engine).__name__,
            "router": type(self.router).__name__,
            "response_engine": type(self.response_engine).__name__,
            "tool_engine": (
                type(self.tool_engine).__name__
                if self.tool_engine is not None
                else "lazy"
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

    # ========================================================
    # RESULT BUILDER
    # ========================================================

    def _result(
        self,
        *,
        handled: bool,
        response: Optional[str],
        route: str,
        intent: str,
        confidence: float,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> dict[str, Any]:

        return {
            "handled": handled,
            "response": response,
            "route": route,
            "intent": intent,
            "confidence": confidence,
            "metadata": metadata or {},
            "error": error,
        }

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    def _execute_tool(
        self,
        route: str,
        message: str,
        intent_result: dict[str, Any],
        route_result: dict[str, Any],
    ) -> dict[str, Any]:

        tool_engine = self._get_tool_engine()

        if tool_engine is None:
            return self._result(
                handled=False,
                response=None,
                route=route,
                intent=intent_result.get("intent", "general"),
                confidence=float(
                    intent_result.get("confidence", 0.5)
                ),
                metadata={
                    **route_result.get("metadata", {}),
                    "tool_execution": False,
                    "tool_engine": "unavailable",
                },
                error="ToolEngine is unavailable.",
            )

        try:
            tool_result = tool_engine.execute(
                route,
                message,
            )

            # ToolResult is expected to expose to_dict().
            if hasattr(tool_result, "to_dict"):
                data = tool_result.to_dict()
            elif isinstance(tool_result, dict):
                data = tool_result
            else:
                data = {
                    "success": True,
                    "response": str(tool_result),
                    "data": None,
                    "error": None,
                }

            success = bool(data.get("success", False))
            response = data.get("response")

            metadata = {
                **route_result.get("metadata", {}),
                "tool_execution": True,
                "tool": route,
                "tool_success": success,
                "tool_data": data.get("data"),
            }

            return self._result(
                handled=success,
                response=response,
                route=route,
                intent=intent_result.get("intent", "general"),
                confidence=float(
                    intent_result.get("confidence", 0.5)
                ),
                metadata=metadata,
                error=data.get("error"),
            )

        except Exception as exc:
            return self._result(
                handled=False,
                response=None,
                route=route,
                intent=intent_result.get("intent", "general"),
                confidence=float(
                    intent_result.get("confidence", 0.5)
                ),
                metadata={
                    **route_result.get("metadata", {}),
                    "tool_execution": True,
                    "tool": route,
                },
                error=f"{type(exc).__name__}: {exc}",
            )

    # ========================================================
    # MAIN HANDLE
    # ========================================================

    def handle(self, message: str) -> dict[str, Any]:
        """
        Execute the complete cognitive pipeline.
        """

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------

        if message is None:
            message = ""

        if not isinstance(message, str):
            message = str(message)

        message = message.strip()

        if not message:
            return self._result(
                handled=True,
                response="Silakan masukkan pesan.",
                route="local",
                intent="general",
                confidence=1.0,
                metadata={
                    "fast_response": True,
                    "empty_message": True,
                    "orchestrator": self.VERSION,
                },
            )

        # ----------------------------------------------------
        # 1. INTENT
        # ----------------------------------------------------

        try:
            intent_result = self.intent_engine.analyze(message)

        except Exception as exc:
            return self._result(
                handled=False,
                response=None,
                route="error",
                intent="general",
                confidence=0.0,
                metadata={
                    "orchestrator": self.VERSION,
                },
                error=(
                    f"IntentEngine error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        intent = str(
            intent_result.get("intent", "general")
        )

        confidence = float(
            intent_result.get("confidence", 0.5)
        )

        normalized_text = intent_result.get(
            "normalized_text",
            message,
        )

        high_confidence = bool(
            intent_result.get(
                "high_confidence",
                confidence >= 0.90,
            )
        )

        # ----------------------------------------------------
        # 2. ROUTING
        # ----------------------------------------------------

        try:
            route_result = self.router.route(
                message,
                intent_result,
            )

        except TypeError:
            # Compatibility with older router API.
            try:
                route_result = self.router.route(
                    intent_result
                )
            except Exception as exc:
                return self._result(
                    handled=False,
                    response=None,
                    route="error",
                    intent=intent,
                    confidence=confidence,
                    metadata={
                        "orchestrator": self.VERSION,
                        "normalized_text": normalized_text,
                    },
                    error=(
                        f"Router error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

        except Exception as exc:
            return self._result(
                handled=False,
                response=None,
                route="error",
                intent=intent,
                confidence=confidence,
                metadata={
                    "orchestrator": self.VERSION,
                    "normalized_text": normalized_text,
                },
                error=(
                    f"Router error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        route = str(
            route_result.get("route", "llm")
        )

        route_metadata = dict(
            route_result.get("metadata", {})
        )

        # ----------------------------------------------------
        # COMMON METADATA
        # ----------------------------------------------------

        base_metadata = {
            **route_metadata,
            "orchestrator": self.VERSION,
            "normalized_text": normalized_text,
            "intent_confidence": confidence,
            "high_confidence": high_confidence,
        }

        # ----------------------------------------------------
        # 3. LOCAL RESPONSE
        # ----------------------------------------------------

        if route in {
            "local",
            "memory",
        }:
            try:
                response_result = self.response_engine.handle(
                    intent,
                    message,
                )

                if hasattr(response_result, "to_dict"):
                    data = response_result.to_dict()
                elif isinstance(response_result, dict):
                    data = response_result
                else:
                    data = {
                        "handled": True,
                        "response": str(response_result),
                    }

                metadata = {
                    **base_metadata,
                    **data.get("metadata", {}),
                }

                return self._result(
                    handled=bool(
                        data.get("handled", True)
                    ),
                    response=data.get("response"),
                    route=data.get(
                        "route",
                        route,
                    ),
                    intent=data.get(
                        "intent",
                        intent,
                    ),
                    confidence=confidence,
                    metadata=metadata,
                    error=data.get("error"),
                )

            except Exception as exc:
                return self._result(
                    handled=False,
                    response=None,
                    route=route,
                    intent=intent,
                    confidence=confidence,
                    metadata=base_metadata,
                    error=(
                        f"ResponseEngine error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

        # ----------------------------------------------------
        # 4. TOOL EXECUTION
        # ----------------------------------------------------

        if route in {
            "calculator",
            "weather",
            "search",
        }:
            result = self._execute_tool(
                route=route,
                message=message,
                intent_result=intent_result,
                route_result={
                    **route_result,
                    "metadata": base_metadata,
                },
            )

            return result

        # ----------------------------------------------------
        # 5. LLM FALLBACK
        # ----------------------------------------------------

        if route == "llm":
            return self._result(
                handled=False,
                response=None,
                route="llm",
                intent=intent,
                confidence=confidence,
                metadata={
                    **base_metadata,
                    "requires_llm": True,
                    "message": message,
                },
            )

        # ----------------------------------------------------
        # 6. UNKNOWN ROUTE → SAFE LLM FALLBACK
        # ----------------------------------------------------

        return self._result(
            handled=False,
            response=None,
            route="llm",
            intent=intent,
            confidence=confidence,
            metadata={
                **base_metadata,
                "fallback": True,
                "requires_llm": True,
                "message": message,
            },
        )


# ============================================================
# FACTORY
# ============================================================

def create_cognitive_orchestrator(
    intent_engine,
    router,
    response_engine,
    tool_engine=None,
) -> CognitiveOrchestrator:

    return CognitiveOrchestrator(
        intent_engine=intent_engine,
        router=router,
        response_engine=response_engine,
        tool_engine=tool_engine,
    )

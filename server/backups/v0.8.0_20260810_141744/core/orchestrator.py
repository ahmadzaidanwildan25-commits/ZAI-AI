from __future__ import annotations

from typing import Any, Dict


class CognitiveOrchestrator:
    """
    Super ZAI Cognitive Pipeline

    Pipeline:
        intent
        ↓
        routing
        ↓
        response
        ↓
        fallback
    """

    VERSION = "0.8.0"

    def __init__(
        self,
        intent_engine,
        router,
        response_engine,
    ):
        self.intent_engine = intent_engine
        self.router = router
        self.response_engine = response_engine

    def stats(self) -> Dict[str, Any]:
        return {
            "engine": "CognitiveOrchestrator",
            "version": self.VERSION,
            "intent_engine": type(
                self.intent_engine
            ).__name__,
            "router": type(
                self.router
            ).__name__,
            "response_engine": type(
                self.response_engine
            ).__name__,
            "pipeline": [
                "intent",
                "routing",
                "response",
                "fallback",
            ],
            "status": "READY",
        }

    def handle(self, message: str) -> Dict[str, Any]:

        text = str(message or "").strip()

        if not text:
            return {
                "handled": True,
                "response": "Pesan kosong.",
                "route": "local",
                "intent": "general",
                "confidence": 0.0,
                "metadata": {},
                "error": None,
            }

        try:
            # ----------------------------------------------
            # 1. INTENT
            # ----------------------------------------------

            intent_result = self.intent_engine.analyze(
                text
            )

            intent = intent_result.get(
                "intent",
                "general",
            )

            confidence = intent_result.get(
                "confidence",
                0.50,
            )

            # ----------------------------------------------
            # 2. ROUTING
            # ----------------------------------------------

            try:
                routing = self.router.route(
                    text,
                    intent_result,
                )
            except TypeError:
                routing = self.router.route(
                    intent_result
                )

            route = routing.get(
                "route",
                "llm",
            )

            metadata = dict(
                routing.get("metadata") or {}
            )

            metadata.update({
                "orchestrator": self.VERSION,
                "normalized_text":
                    intent_result.get(
                        "normalized_text",
                        text.lower(),
                    ),
                "intent_confidence": confidence,
                "high_confidence":
                    intent_result.get(
                        "high_confidence",
                        False,
                    ),
            })

            # ----------------------------------------------
            # 3. RESPONSE ENGINE
            # ----------------------------------------------

            result = self.response_engine.handle(
                intent,
                text,
            )

            if result is not None:

                if hasattr(result, "to_dict"):
                    output = result.to_dict()
                elif isinstance(result, dict):
                    output = dict(result)
                else:
                    output = {
                        "handled": True,
                        "response": str(result),
                        "route": route,
                        "intent": intent,
                        "metadata": {},
                    }

                output["confidence"] = confidence

                output_metadata = dict(
                    output.get("metadata") or {}
                )

                output_metadata.update(metadata)

                output["metadata"] = output_metadata

                output.setdefault(
                    "error",
                    None,
                )

                return output

            # ----------------------------------------------
            # 4. FALLBACK
            # ----------------------------------------------

            return {
                "handled": False,
                "response": None,
                "route": route,
                "intent": intent,
                "confidence": confidence,
                "metadata": metadata,
                "error": None,
            }

        except Exception as error:

            return {
                "handled": False,
                "response": None,
                "route": "error",
                "intent": "general",
                "confidence": 0.0,
                "metadata": {
                    "orchestrator": self.VERSION,
                },
                "error": str(error),
            }

from __future__ import annotations

from typing import Any, Dict, Optional

from memory.manager import MemoryManager


class IntentRouter:
    """
    ZAI Intent Router 0.8.0

    Mendukung:
        route(intent_result)
        route(message, intent_result)
        route(message, intent_name)
    """

    VERSION = "0.8.0"

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager

    def route(
        self,
        first: Any,
        second: Any = None,
    ) -> Dict[str, Any]:

        # --------------------------------------------------
        # FORMAT 1:
        # route({"intent": "...", ...})
        # --------------------------------------------------

        if isinstance(first, dict):
            result = dict(first)
            message = str(
                result.get("normalized_text")
                or result.get("message")
                or ""
            )
            intent = str(
                result.get("intent")
                or "general"
            )

        # --------------------------------------------------
        # FORMAT 2:
        # route("pesan", {"intent": "weather"})
        # --------------------------------------------------

        elif isinstance(second, dict):
            message = str(first or "")
            result = dict(second)
            intent = str(
                result.get("intent")
                or "general"
            )

        # --------------------------------------------------
        # FORMAT 3:
        # route("pesan", "weather")
        # --------------------------------------------------

        else:
            message = str(first or "")
            intent = str(second or "general")

            result = {
                "intent": intent,
                "normalized_text": message.strip().lower(),
            }

        intent = intent.strip().lower()

        # --------------------------------------------------
        # ROUTING
        # --------------------------------------------------

        if intent in {
            "greeting",
            "identity",
            "status",
            "help",
        }:
            route = "local"

        elif intent.startswith("memory_"):
            route = "memory"

        elif intent == "calculation":
            route = "calculator"

        elif intent == "weather":
            route = "weather"

        elif intent == "search":
            route = "search"

        elif intent == "coding":
            route = "llm"

        else:
            route = "llm"

        metadata = dict(result.get("metadata") or {})

        if route == "weather":
            metadata.update({
                "requires_weather": True,
                "message": message,
            })

        elif route == "calculator":
            metadata.update({
                "requires_calculator": True,
                "message": message,
            })

        elif route == "search":
            metadata.update({
                "requires_search": True,
                "message": message,
            })

        elif route == "llm":
            metadata.update({
                "requires_llm": True,
                "message": message,
            })

            if intent == "coding":
                metadata["mode"] = "deep"
            else:
                metadata["mode"] = "normal"

        return {
            "route": route,
            "intent": intent,
            "confidence": result.get(
                "confidence",
                0.50,
            ),
            "metadata": metadata,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "engine": "IntentRouter",
            "version": self.VERSION,
            "status": "READY",
            "weather": True,
            "calculator": True,
            "search": True,
            "llm": True,
            "memory": True,
        }

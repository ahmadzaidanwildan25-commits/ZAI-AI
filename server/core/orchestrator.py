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
    metadata: dict = field(default_factory=dict)
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

    def __getitem__(self, key: str):
        """
        Compatibility layer.

        Beberapa test lama atau komponen lama masih
        memperlakukan hasil orchestrator seperti dict.

        Dengan ini kedua pola tetap didukung:

            result.response

        dan:

            result["response"]
        """
        return self.to_dict()[key]

    def get(self, key: str, default=None):
        return self.to_dict().get(key, default)


class CognitiveOrchestrator:
    """
    SUPER ZAI COGNITIVE ORCHESTRATOR

    Version:
        1.1.0

    Pipeline:

        MESSAGE
           ↓
        INTENT
           ↓
        ROUTING
           ↓
        LOCAL / MEMORY / TOOL / LLM
           ↓
        RESPONSE
           ↓
        FALLBACK

    Tool routes:

        calculator
        weather
        search

    Local routes:

        local
        memory

    Unknown/general:

        llm
    """

    VERSION = "1.1.0"

    def __init__(
        self,
        intent_engine=None,
        router=None,
        response_engine=None,
        tool_engine=None,
        memory_manager=None,
    ):

        self.intent_engine = (
            intent_engine
            or get_intent_engine()
        )

        self.memory_manager = (
            memory_manager
            or MemoryManager()
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
            "tool_engine": type(
                self.tool_engine
            ).__name__,
            "pipeline": [
                "intent",
                "routing",
                "tool_execution",
                "response",
                "fallback",
            ],
            "compatibility": {
                "dict_access": True,
                "attribute_access": True,
            },
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
                response="Pesan kosong.",
                route="local",
                intent="empty",
                confidence=1.0,
                metadata={
                    "orchestrator": self.VERSION,
                    "fast_response": True,
                },
            )

        try:

            # ==================================================
            # INTENT
            # ==================================================

            intent_result = (
                self.intent_engine.analyze(text)
            )

            intent = self._read_value(
                intent_result,
                "intent",
                "general",
            )

            confidence = self._read_float(
                intent_result,
                "confidence",
                0.5,
            )

            normalized_text = self._read_value(
                intent_result,
                "normalized_text",
                text,
            )

            # ==================================================
            # ROUTING
            # ==================================================

            routing = self._route(
                text,
                intent_result,
            )

            route = self._read_value(
                routing,
                "route",
                "llm",
            )

            metadata = self._read_dict(
                routing,
                "metadata",
            )

            metadata.update(
                {
                    "orchestrator": self.VERSION,
                    "normalized_text": normalized_text,
                    "intent_confidence": confidence,
                    "high_confidence": confidence >= 0.90,
                    "message": text,
                }
            )

            # ==================================================
            # LOCAL / MEMORY
            # ==================================================

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

                response = self._response_text(
                    result
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

            # ==================================================
            # TOOLS
            # ==================================================

            if route in {
                "calculator",
                "weather",
                "search",
            }:

                tool_metadata = dict(
                    metadata
                )

                resolved_location = (
                    self._resolve_location(
                        text=text,
                        supplied_location=location,
                        metadata=tool_metadata,
                    )
                )

                if resolved_location:

                    tool_metadata[
                        "location"
                    ] = resolved_location

                    metadata[
                        "resolved_location"
                    ] = resolved_location

                # --------------------------------------------------
                # WEATHER WITHOUT LOCATION
                # --------------------------------------------------

                if (
                    route == "weather"
                    and not resolved_location
                ):

                    return OrchestratorResult(
                        handled=True,
                        response=(
                            "Saya perlu mengetahui "
                            "lokasi untuk memberikan "
                            "cuaca. Contoh: "
                            "cuaca Jakarta."
                        ),
                        route="weather",
                        intent=intent,
                        confidence=confidence,
                        metadata={
                            **metadata,
                            "tool": "weather",
                            "tool_success": False,
                            "requires_location": True,
                        },
                    )

                # --------------------------------------------------
                # EXECUTE TOOL
                # --------------------------------------------------

                tool_result = (
                    self.tool_engine.execute(
                        route,
                        text,
                        tool_metadata,
                    )
                )

                metadata["tool"] = (
                    tool_result.tool
                )

                metadata["tool_success"] = (
                    tool_result.success
                )

                metadata["tool_data"] = (
                    tool_result.data
                )

                if tool_result.success:

                    return OrchestratorResult(
                        handled=True,
                        response=tool_result.response,
                        route=route,
                        intent=intent,
                        confidence=confidence,
                        metadata=metadata,
                    )

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

            # ==================================================
            # LLM
            # ==================================================

            if route == "llm":

                return OrchestratorResult(
                    handled=False,
                    response=None,
                    route="llm",
                    intent=intent,
                    confidence=confidence,
                    metadata=metadata,
                )

            # ==================================================
            # FALLBACK
            # ==================================================

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
                    "orchestrator": self.VERSION,
                    "message": text,
                },
                error=str(error),
            )

    # ==========================================================
    # ROUTER COMPATIBILITY
    # ==========================================================

    def _route(
        self,
        text: str,
        intent_result: Any,
    ):

        try:

            return self.router.route(
                text,
                intent_result,
            )

        except TypeError:

            return self.router.route(
                intent_result
            )

    # ==========================================================
    # LOCATION RESOLUTION
    # ==========================================================

    @staticmethod
    def _resolve_location(
        text: str,
        supplied_location: Optional[dict],
        metadata: dict,
    ) -> Optional[dict]:

        # ------------------------------------------------------
        # 1. Explicit location argument
        # ------------------------------------------------------

        if isinstance(
            supplied_location,
            dict,
        ):

            if (
                supplied_location.get(
                    "latitude"
                )
                is not None
                and
                supplied_location.get(
                    "longitude"
                )
                is not None
            ):

                return supplied_location

        # ------------------------------------------------------
        # 2. Location inside metadata
        # ------------------------------------------------------

        metadata_location = metadata.get(
            "location"
        )

        if isinstance(
            metadata_location,
            dict,
        ):

            if (
                metadata_location.get(
                    "latitude"
                )
                is not None
                and
                metadata_location.get(
                    "longitude"
                )
                is not None
            ):

                return metadata_location

        # ------------------------------------------------------
        # 3. City name extraction
        #
        # ToolEngine juga memiliki geocoding sendiri.
        # Di sini kita hanya perlu memberikan sinyal bahwa
        # user menyebutkan lokasi.
        # ------------------------------------------------------

        location_name = (
            CognitiveOrchestrator
            ._extract_location_name(text)
        )

        if location_name:

            return {
                "city": location_name,
            }

        return None

    @staticmethod
    def _extract_location_name(
        text: str,
    ) -> Optional[str]:

        import re

        value = str(
            text or ""
        ).strip()

        patterns = [

            # cuaca Jakarta
            r"^\s*cuaca\s+(.+?)\s*$",

            # cuaca di Jakarta
            r"^\s*cuaca\s+di\s+(.+?)\s*$",

            # bagaimana cuaca Jakarta
            r"^\s*(?:bagaimana|gimana)\s+cuaca\s+(.+?)\s*$",

            # bagaimana cuaca di Jakarta
            r"^\s*(?:bagaimana|gimana)\s+cuaca\s+di\s+(.+?)\s*$",

            # di Jakarta
            r"^\s*cuaca.*?\bdi\s+(.+?)\s*$",
        ]

        for pattern in patterns:

            match = re.match(
                pattern,
                value,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            location = (
                match.group(1)
                .strip()
                .strip("?.!,")
            )

            if not location:
                continue

            # Jangan menganggap "hari ini" sebagai kota.
            if location.lower() in {
                "hari ini",
                "sekarang",
                "saat ini",
                "besok",
                "nanti",
            }:
                return None

            return location

        return None

    # ==========================================================
    # RESPONSE HELPERS
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

    # ==========================================================
    # GENERIC RESULT HELPERS
    # ==========================================================

    @staticmethod
    def _read_value(
        result: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            result,
            dict,
        ):

            return result.get(
                key,
                default,
            )

        if hasattr(
            result,
            key,
        ):

            return getattr(
                result,
                key,
            )

        if hasattr(
            result,
            "get",
        ):

            try:

                return result.get(
                    key,
                    default,
                )

            except Exception:
                pass

        return default

    @staticmethod
    def _read_float(
        result: Any,
        key: str,
        default: float,
    ) -> float:

        value = (
            CognitiveOrchestrator
            ._read_value(
                result,
                key,
                default,
            )
        )

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    @staticmethod
    def _read_dict(
        result: Any,
        key: str,
    ) -> dict:

        value = (
            CognitiveOrchestrator
            ._read_value(
                result,
                key,
                {},
            )
        )

        if isinstance(
            value,
            dict,
        ):

            return dict(value)

        return {}

    # ==========================================================
    # TOOL ERROR
    # ==========================================================

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
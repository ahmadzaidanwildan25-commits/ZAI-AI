from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# ============================================================
# ZAI RESPONSE ENGINE
# SUPER ZAI
# VERSION 0.1.0
# ============================================================


@dataclass
class ResponseResult:
    """
    Hasil keputusan Response Engine.
    """

    handled: bool
    response: Optional[str] = None
    route: str = "llm"
    intent: str = "general"
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "handled": self.handled,
            "response": self.response,
            "route": self.route,
            "intent": self.intent,
            "metadata": self.metadata or {},
        }


class ResponseEngine:
    """
    Response Engine untuk Super ZAI.

    Tugas utama:

    - Menangani intent sederhana tanpa LLM.
    - Menghasilkan respons cepat.
    - Menentukan apakah request harus diteruskan
      ke LLM.
    - Menjadi lapisan antara Intent Router dan Ollama.
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        memory_manager: Any = None,
    ) -> None:

        self.memory_manager = memory_manager

    # ========================================================
    # PUBLIC API
    # ========================================================

    def handle(
        self,
        intent: str,
        message: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> ResponseResult:
        """
        Memproses intent dan menentukan respons.

        Returns:
            ResponseResult
        """

        intent = self._normalize_intent(intent)
        message = self._normalize_message(message)

        context = context or {}

        # ----------------------------------------------------
        # GREETING
        # ----------------------------------------------------

        if intent == "greeting":
            return self._greeting()

        # ----------------------------------------------------
        # IDENTITY
        # ----------------------------------------------------

        if intent == "identity":
            return self._identity()

        # ----------------------------------------------------
        # MEMORY COUNT
        # ----------------------------------------------------

        if intent == "memory_count":
            return self._memory_count()

        # ----------------------------------------------------
        # MEMORY QUERY
        # ----------------------------------------------------

        if intent == "memory_query":
            return self._memory_query()

        # ----------------------------------------------------
        # MEMORY SAVE
        # ----------------------------------------------------

        if intent == "memory_save":
            return ResponseResult(
                handled=False,
                response=None,
                route="memory",
                intent=intent,
                metadata={
                    "requires_memory_action": True,
                },
            )

        # ----------------------------------------------------
        # MEMORY DELETE
        # ----------------------------------------------------

        if intent == "memory_delete":
            return ResponseResult(
                handled=False,
                response=None,
                route="memory",
                intent=intent,
                metadata={
                    "requires_memory_action": True,
                },
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if intent == "status":
            return self._status(context)

        # ----------------------------------------------------
        # HELP
        # ----------------------------------------------------

        if intent == "help":
            return self._help()

        # ----------------------------------------------------
        # CALCULATION
        # ----------------------------------------------------

        if intent == "calculation":
            return ResponseResult(
                handled=False,
                response=None,
                route="calculator",
                intent=intent,
                metadata={
                    "requires_calculator": True,
                    "message": message,
                },
            )

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        if intent == "weather":
            return ResponseResult(
                handled=False,
                response=None,
                route="weather",
                intent=intent,
                metadata={
                    "requires_weather": True,
                    "message": message,
                },
            )

        # ----------------------------------------------------
        # CODING
        # ----------------------------------------------------

        if intent == "coding":
            return self._llm(
                intent=intent,
                message=message,
                mode="deep",
            )

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        if intent == "search":
            return ResponseResult(
                handled=False,
                response=None,
                route="search",
                intent=intent,
                metadata={
                    "requires_search": True,
                    "message": message,
                },
            )

        # ----------------------------------------------------
        # GENERAL
        # ----------------------------------------------------

        return self._llm(
            intent=intent,
            message=message,
            mode="normal",
        )

    # ========================================================
    # GREETING
    # ========================================================

    def _greeting(self) -> ResponseResult:

        return ResponseResult(
            handled=True,
            response="Halo! Saya ZAI. Ada yang bisa saya bantu?",
            route="local",
            intent="greeting",
            metadata={
                "fast_response": True,
            },
        )

    # ========================================================
    # IDENTITY
    # ========================================================

    def _identity(self) -> ResponseResult:

        return ResponseResult(
            handled=True,
            response=(
                "Saya ZAI, intelligence core dari Super ZAI. "
                "Saya dirancang sebagai asisten AI pribadi "
                "yang dapat berkembang dengan memory, tools, "
                "automation, dan berbagai kemampuan lainnya."
            ),
            route="local",
            intent="identity",
            metadata={
                "fast_response": True,
            },
        )

    # ========================================================
    # MEMORY COUNT
    # ========================================================

    def _memory_count(self) -> ResponseResult:

        count = 0

        try:

            if self.memory_manager is not None:

                if hasattr(
                    self.memory_manager,
                    "count",
                ):
                    count = int(
                        self.memory_manager.count()
                    )

                elif hasattr(
                    self.memory_manager,
                    "count_memories",
                ):
                    count = int(
                        self.memory_manager.count_memories()
                    )

        except Exception:

            count = 0

        return ResponseResult(
            handled=True,
            response=(
                f"Saat ini saya memiliki "
                f"{count} memory tersimpan."
            ),
            route="memory",
            intent="memory_count",
            metadata={
                "count": count,
                "fast_response": True,
            },
        )

    def _memory_query(self) -> ResponseResult:

        memories = []

        try:

            if self.memory_manager is not None:

                if hasattr(
                    self.memory_manager,
                    "get_important_memories",
                ):
                    memories = (
                        self.memory_manager
                        .get_important_memories(
                            limit=10
                        )
                    )

                elif hasattr(
                    self.memory_manager,
                    "important",
                ):
                    memories = (
                        self.memory_manager
                        .important(
                            limit=10
                        )
                    )

        except Exception:

            memories = []

        if not memories:

            return ResponseResult(
                handled=True,
                response=(
                    "Saat ini belum ada memory penting "
                    "yang bisa saya tampilkan."
                ),
                route="memory",
                intent="memory_query",
                metadata={
                    "count": 0,
                    "fast_response": True,
                },
            )

        lines = [
            "Berikut beberapa hal yang saya ingat:"
        ]

        for item in memories:

            if not isinstance(
                item,
                dict,
            ):
                continue

            key = str(
                item.get(
                    "key",
                    "",
                )
            ).strip()

            value = str(
                item.get(
                    "value",
                    "",
                )
            ).strip()

            if key and value:

                if key.lower() == value.lower():

                    lines.append(
                        f"- {value}"
                    )

                else:

                    lines.append(
                        f"- {key}: {value}"
                    )

        if len(lines) == 1:

            return ResponseResult(
                handled=True,
                response=(
                    "Saya belum menemukan memory "
                    "yang bisa ditampilkan."
                ),
                route="memory",
                intent="memory_query",
            )

        return ResponseResult(
            handled=True,
            response="\n".join(lines),
            route="memory",
            intent="memory_query",
            metadata={
                "count": len(lines) - 1,
                "fast_response": True,
            },
        )

    # ========================================================
    # STATUS
    # ========================================================

    def _status(
        self,
        context: dict[str, Any],
    ) -> ResponseResult:

        model = context.get(
            "model",
            "qwen3:8b",
        )

        memory_enabled = context.get(
            "memory",
            True,
        )

        streaming = context.get(
            "streaming",
            True,
        )

        ollama = context.get(
            "ollama",
            True,
        )

        status = context.get(
            "status",
            "ONLINE",
        )

        response = (
            "ZAI sedang online.\n"
            f"Status: {status}\n"
            f"Model: {model}\n"
            f"Ollama: "
            f"{'ONLINE' if ollama else 'OFFLINE'}\n"
            f"Memory: "
            f"{'ENABLED' if memory_enabled else 'DISABLED'}\n"
            f"Streaming: "
            f"{'ENABLED' if streaming else 'DISABLED'}"
        )

        return ResponseResult(
            handled=True,
            response=response,
            route="local",
            intent="status",
            metadata={
                "fast_response": True,
                "status": status,
                "model": model,
            },
        )

    # ========================================================
    # HELP
    # ========================================================

    def _help(self) -> ResponseResult:

        response = (
            "Saya bisa membantu dengan:\n\n"
            "- Percakapan dan pertanyaan umum\n"
            "- Memory jangka panjang\n"
            "- Perhitungan\n"
            "- Cuaca\n"
            "- Coding\n"
            "- Pencarian informasi\n"
            "- Analisis dan penjelasan\n"
            "- Status sistem ZAI\n\n"
            "Contoh:\n"
            "\"ingat saya sedang membangun Super ZAI\"\n"
            "\"berapa memory\"\n"
            "\"status zai\"\n"
            "\"hitung 20 + 30\""
        )

        return ResponseResult(
            handled=True,
            response=response,
            route="local",
            intent="help",
            metadata={
                "fast_response": True,
            },
        )

    # ========================================================
    # LLM ROUTE
    # ========================================================

    def _llm(
        self,
        intent: str,
        message: str,
        mode: str,
    ) -> ResponseResult:

        return ResponseResult(
            handled=False,
            response=None,
            route="llm",
            intent=intent,
            metadata={
                "mode": mode,
                "message": message,
                "requires_llm": True,
            },
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_intent(
        intent: str,
    ) -> str:

        if not intent:
            return "general"

        return (
            str(intent)
            .strip()
            .lower()
        )

    @staticmethod
    def _normalize_message(
        message: str,
    ) -> str:

        if not message:
            return ""

        return " ".join(
            str(message)
            .strip()
            .split()
        )

    # ========================================================
    # VERSION
    # ========================================================

    @classmethod
    def version(cls) -> str:

        return cls.VERSION


# ============================================================
# SINGLETON
# ============================================================

_response_engine: Optional[ResponseEngine] = None


def get_response_engine(
    memory_manager: Any = None,
) -> ResponseEngine:

    global _response_engine

    if _response_engine is None:

        _response_engine = ResponseEngine(
            memory_manager=memory_manager
        )

    elif (
        memory_manager is not None
        and _response_engine.memory_manager is None
    ):

        _response_engine.memory_manager = (
            memory_manager
        )

    return _response_engine


# ============================================================
# RESET
# ============================================================

def reset_response_engine() -> None:

    global _response_engine

    _response_engine = None
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from memory.manager import MemoryManager


# ==========================================================
# RESPONSE RESULT
# ==========================================================

@dataclass
class ResponseResult:
    """
    Standard result object dari ResponseEngine.

    Mendukung:
        result.to_dict()
        result["response"]
        result["handled"]
        result["route"]
        result["intent"]
        result["metadata"]
    """

    handled: bool
    response: Optional[str]
    route: str
    intent: str
    metadata: dict
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "handled": self.handled,
            "response": self.response,
            "route": self.route,
            "intent": self.intent,
            "metadata": self.metadata,
            "error": self.error,
        }

    def __getitem__(self, key: str) -> Any:
        """
        Compatibility layer agar object dapat digunakan
        seperti dictionary oleh master test / orchestrator.
        """
        return self.to_dict()[key]

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


# ==========================================================
# RESPONSE ENGINE
# ==========================================================

class ResponseEngine:
    """
    Super ZAI Response Engine.

    Bertugas menangani response lokal berlatensi rendah
    sebelum request diteruskan ke LLM atau tool.

    Version:
        1.1.0
    """

    VERSION = "1.1.0"

    def __init__(
        self,
        memory_manager: MemoryManager,
    ):
        self.memory = memory_manager

    # ======================================================
    # MAIN
    # ======================================================

    def handle(
        self,
        intent: str,
        message: str,
    ) -> ResponseResult:

        intent = str(
            intent or "general"
        ).strip().lower()

        text = str(
            message or ""
        ).strip()

        # --------------------------------------------------
        # GREETING
        # --------------------------------------------------

        if intent == "greeting":

            return ResponseResult(
                handled=True,
                response=(
                    "Halo! Saya ZAI. "
                    "Ada yang bisa saya bantu?"
                ),
                route="local",
                intent=intent,
                metadata={
                    "fast_response": True,
                },
            )

        # --------------------------------------------------
        # IDENTITY
        # --------------------------------------------------

        if intent == "identity":

            return ResponseResult(
                handled=True,
                response=(
                    "Saya ZAI, intelligence core "
                    "dari Super ZAI. Saya dirancang "
                    "sebagai asisten AI pribadi yang "
                    "dapat berkembang dengan memory, "
                    "tools, automation, dan berbagai "
                    "kemampuan lainnya."
                ),
                route="local",
                intent=intent,
                metadata={
                    "fast_response": True,
                },
            )

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        if intent == "status":

            return ResponseResult(
                handled=True,
                response=(
                    "ZAI sedang online.\n"
                    "Status: ONLINE\n"
                    "Model: qwen3:8b\n"
                    "Ollama: ONLINE\n"
                    "Memory: ENABLED\n"
                    "Streaming: ENABLED\n"
                    "Tool Engine: ENABLED"
                ),
                route="local",
                intent=intent,
                metadata={
                    "fast_response": True,
                    "status": "ONLINE",
                    "model": "qwen3:8b",
                },
            )

        # --------------------------------------------------
        # HELP
        # --------------------------------------------------

        if intent == "help":

            return ResponseResult(
                handled=True,
                response=(
                    "Saya bisa membantu dengan:\n\n"
                    "- Percakapan dan pertanyaan umum\n"
                    "- Memory jangka panjang\n"
                    "- Perhitungan\n"
                    "- Cuaca\n"
                    "- Pencarian informasi\n"
                    "- Coding\n"
                    "- Analisis\n"
                    "- Status sistem ZAI\n"
                    "- Tool execution\n\n"
                    "Contoh:\n"
                    '"ingat saya sedang membangun Super ZAI"\n'
                    '"berapa memory"\n'
                    '"status zai"\n'
                    '"hitung 20 + 30"\n'
                    '"cuaca Jakarta"\n'
                    '"cari berita terbaru tentang AI"'
                ),
                route="local",
                intent=intent,
                metadata={
                    "fast_response": True,
                },
            )

        # --------------------------------------------------
        # MEMORY COUNT
        # --------------------------------------------------

        if intent == "memory_count":

            try:
                count = self.memory.count()

            except Exception as error:

                return ResponseResult(
                    handled=False,
                    response=None,
                    route="memory",
                    intent=intent,
                    metadata={
                        "fast_response": False,
                    },
                    error=(
                        f"Memory count gagal: {error}"
                    ),
                )

            return ResponseResult(
                handled=True,
                response=(
                    f"Saat ini saya memiliki "
                    f"{count} memory tersimpan."
                ),
                route="memory",
                intent=intent,
                metadata={
                    "count": count,
                    "fast_response": True,
                },
            )

        # --------------------------------------------------
        # MEMORY QUERY
        # --------------------------------------------------

        if intent == "memory_query":

            try:
                memories = self.memory.query(
                    text
                )

            except Exception as error:

                return ResponseResult(
                    handled=False,
                    response=None,
                    route="memory",
                    intent=intent,
                    metadata={
                        "fast_response": False,
                    },
                    error=(
                        f"Memory query gagal: {error}"
                    ),
                )

            if not memories:

                return ResponseResult(
                    handled=True,
                    response=(
                        "Saat ini belum ada "
                        "memory penting yang bisa "
                        "saya tampilkan."
                    ),
                    route="memory",
                    intent=intent,
                    metadata={
                        "count": 0,
                        "fast_response": True,
                    },
                )

            lines = [
                "Berikut beberapa hal "
                "yang saya ingat:"
            ]

            for item in memories[:10]:

                if isinstance(item, dict):

                    value = item.get(
                        "value",
                        item.get(
                            "key",
                            "",
                        ),
                    )

                else:

                    value = str(item)

                if value:
                    lines.append(
                        f"- {value}"
                    )

            return ResponseResult(
                handled=True,
                response="\n".join(
                    lines
                ),
                route="memory",
                intent=intent,
                metadata={
                    "count": len(memories),
                    "fast_response": True,
                },
            )

        # --------------------------------------------------
        # GENERAL / UNKNOWN
        # --------------------------------------------------

        return ResponseResult(
            handled=False,
            response=None,
            route="llm",
            intent=intent,
            metadata={
                "requires_llm": True,
                "message": text,
            },
        )

    # ======================================================
    # STATUS
    # ======================================================

    def stats(self) -> dict:

        return {
            "engine": "ResponseEngine",
            "version": self.VERSION,
            "memory": self.memory is not None,
            "local_responses": True,
            "status": "READY",
        }


# ==========================================================
# SINGLETON
# ==========================================================

_response_engine: Optional[ResponseEngine] = None


def get_response_engine(
    memory_manager: Optional[MemoryManager] = None,
) -> ResponseEngine:
    """
    Global ResponseEngine factory.

    Bisa dipanggil dengan:

        get_response_engine()

    atau:

        get_response_engine(memory_manager)
    """

    global _response_engine

    if _response_engine is None:

        if memory_manager is None:
            memory_manager = MemoryManager()

        _response_engine = ResponseEngine(
            memory_manager
        )

    return _response_engine


# ==========================================================
# RESET
# ==========================================================

def reset_response_engine() -> None:
    """
    Reset singleton ResponseEngine.

    Berguna untuk testing.
    """

    global _response_engine

    _response_engine = None
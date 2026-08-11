"""
============================================================
SUPER ZAI - AI BRAIN
============================================================

Central cognitive engine of ZAI.

Responsibilities:
- Context management
- Intent processing
- Memory integration
- Planning
- Reasoning
- Agent loop
- Tool execution
- Final response generation

Version: 0.11.0
============================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AIBrain:
    """
    Core cognitive brain of Super ZAI.
    """

    VERSION = "0.11.0"

    def __init__(
        self,
        tool_engine: Any = None,
        intent_engine: Any = None,
        intent_router: Any = None,
        memory_manager: Any = None,
        response_engine: Any = None,
        llm_client: Any = None,
    ) -> None:

        self.tool_engine = tool_engine
        self.intent_engine = intent_engine
        self.intent_router = intent_router
        self.memory_manager = memory_manager
        self.response_engine = response_engine
        self.llm_client = llm_client

        self._runtime = {
            "thoughts": 0,
            "tool_calls": 0,
            "memory_reads": 0,
            "memory_writes": 0,
            "memory_hits": 0,
            "memory_misses": 0,
            "errors": 0,
        }

    # ========================================================
    # TIME
    # ========================================================

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> Dict[str, Any]:

        memory_ready = False

        if self.memory_manager is not None:
            memory_ready = True

        return {
            "engine": "AIBrain",
            "version": self.VERSION,
            "context_manager": True,
            "planner": True,
            "reasoning_engine": True,
            "agent_loop": True,
            "tool_engine": self.tool_engine is not None,
            "intent_engine": self.intent_engine is not None,
            "intent_router": self.intent_router is not None,
            "memory_manager": memory_ready,
            "response_engine": self.response_engine is not None,
            "llm_client": self.llm_client is not None,
            "runtime": dict(self._runtime),
            "status": "READY",
        }

    # ========================================================
    # MEMORY READ
    # ========================================================

    def _memory_search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:

        if self.memory_manager is None:
            return []

        try:

            self._runtime["memory_reads"] += 1

            results = self.memory_manager.search(
                query,
                limit=limit,
            )

            if results:
                self._runtime["memory_hits"] += 1
            else:
                self._runtime["memory_misses"] += 1

            return results or []

        except Exception:
            self._runtime["errors"] += 1
            return []

    # ========================================================
    # IMPORTANT MEMORY
    # ========================================================

    def _important_memories(
        self,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:

        if self.memory_manager is None:
            return []

        try:

            self._runtime["memory_reads"] += 1

            return self.memory_manager.important(
                limit=limit,
            ) or []

        except Exception:
            self._runtime["errors"] += 1
            return []

    # ========================================================
    # MEMORY SAVE
    # ========================================================

    def _save_memory(
        self,
        key: str,
        value: str,
        category: str = "general",
        importance: int = 5,
    ) -> bool:

        if self.memory_manager is None:
            return False

        try:

            result = self.memory_manager.save(
                key=key,
                value=value,
                category=category,
                importance=importance,
            )

            if result:
                self._runtime["memory_writes"] += 1

            return bool(result)

        except Exception:
            self._runtime["errors"] += 1
            return False

    # ========================================================
    # CONTEXT
    # ========================================================

    def _build_context(
        self,
        message: str,
        conversation: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:

        conversation = conversation or []

        important = self._important_memories(limit=5)

        relevant = self._memory_search(
            message,
            limit=10,
        )

        return {
            "user_message": message,
            "conversation": conversation,
            "important_memories": important,
            "relevant_memories": relevant,
        }

    # ========================================================
    # INTENT
    # ========================================================

    def _detect_intent(
        self,
        message: str,
    ) -> Dict[str, Any]:

        if self.intent_engine is None:

            return {
                "intent": "general",
                "confidence": 0.0,
                "normalized_text": message,
                "entities": {},
                "tool": None,
                "arguments": {},
                "reasoning": "Intent engine belum tersedia.",
            }

        try:

            if hasattr(
                self.intent_engine,
                "analyze",
            ):

                result = self.intent_engine.analyze(
                    message
                )

            elif hasattr(
                self.intent_engine,
                "process",
            ):

                result = self.intent_engine.process(
                    message
                )

            elif hasattr(
                self.intent_engine,
                "detect",
            ):

                result = self.intent_engine.detect(
                    message
                )

            else:

                return {
                    "intent": "general",
                    "confidence": 0.0,
                    "normalized_text": message,
                    "entities": {},
                    "tool": None,
                    "arguments": {},
                    "reasoning": "API intent engine tidak ditemukan.",
                }

            if hasattr(result, "to_dict"):
                return result.to_dict()

            if isinstance(result, dict):
                return result

            return {
                "intent": "general",
                "confidence": 0.0,
                "normalized_text": message,
                "entities": {},
                "tool": None,
                "arguments": {},
                "reasoning": "Format intent tidak dikenali.",
            }

        except Exception as exc:

            self._runtime["errors"] += 1

            return {
                "intent": "general",
                "confidence": 0.0,
                "normalized_text": message,
                "entities": {},
                "tool": None,
                "arguments": {},
                "reasoning": str(exc),
            }

    # ========================================================
    # MEMORY INTENT
    # ========================================================

    def _handle_memory_intent(
        self,
        intent_result: Dict[str, Any],
    ) -> Optional[str]:

        intent = str(
            intent_result.get(
                "intent",
                "",
            )
        ).lower()

        entities = (
            intent_result.get(
                "entities",
                {},
            )
            or {}
        )

        # ----------------------------------------------------
        # MEMORY SAVE
        # ----------------------------------------------------

        if intent == "memory_save":

            key = str(
                entities.get(
                    "memory_key",
                    "",
                )
            ).strip()

            value = str(
                entities.get(
                    "memory_value",
                    "",
                )
            ).strip()

            if not key or not value:
                return (
                    "Saya belum mendapatkan data yang cukup "
                    "untuk disimpan ke memory."
                )

            saved = self._save_memory(
                key=key,
                value=value,
                category="identity",
                importance=10,
            )

            if saved:

                return (
                    f"Baik. Saya sudah menyimpan bahwa "
                    f"{key} Anda adalah {value}."
                )

            return (
                "Saya memahami informasinya, tetapi "
                "memory belum berhasil menyimpannya."
            )

        # ----------------------------------------------------
        # MEMORY GET
        # ----------------------------------------------------

        if intent in {
            "memory_get",
            "memory_recall",
            "memory_query",
        }:

            key = str(
                entities.get(
                    "memory_key",
                    "",
                )
            ).strip()

            if key and self.memory_manager is not None:

                try:

                    self._runtime[
                        "memory_reads"
                    ] += 1

                    value = self.memory_manager.get(
                        key
                    )

                    if value:

                        self._runtime[
                            "memory_hits"
                        ] += 1

                        return str(value)

                    self._runtime[
                        "memory_misses"
                    ] += 1

                except Exception:

                    self._runtime[
                        "errors"
                    ] += 1

            return (
                "Saya belum menemukan informasi "
                "tersebut di memory ZAI."
            )

        return None

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    def _execute_tool(
        self,
        tool: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        if self.tool_engine is None:

            return {
                "success": False,
                "tool": tool,
                "response": "Tool engine belum tersedia.",
                "data": {},
                "error": "tool_engine_unavailable",
            }

        try:

            self._runtime["tool_calls"] += 1

            query = arguments.get(
                "query",
                "",
            )

            result = self.tool_engine.execute(
                tool,
                query,
            )

            if hasattr(result, "to_dict"):
                return result.to_dict()

            if isinstance(result, dict):
                return result

            return {
                "success": False,
                "tool": tool,
                "response": str(result),
                "data": {},
                "error": None,
            }

        except Exception as exc:

            self._runtime["errors"] += 1

            return {
                "success": False,
                "tool": tool,
                "response": "Tool gagal dijalankan.",
                "data": {},
                "error": str(exc),
            }

    # ========================================================
    # SIMPLE TOOL ROUTING
    # ========================================================

    @staticmethod
    def _detect_tool_from_message(
        message: str,
    ) -> Optional[str]:

        text = message.lower().strip()

        # calculator

        calculator_words = [
            "hitung",
            "berapa",
            "kali",
            "dibagi",
            "ditambah",
            "dikurangi",
            "persen",
            "perhitungan",
        ]

        if any(
            word in text
            for word in calculator_words
        ):
            return "calculator"

        # weather

        weather_words = [
            "cuaca",
            "hujan",
            "suhu",
            "temperatur",
            "kelembapan",
            "angin",
        ]

        if any(
            word in text
            for word in weather_words
        ):
            return "weather"

        # search

        search_words = [
            "cari",
            "search",
            "berita",
            "terbaru",
            "informasi",
        ]

        if any(
            word in text
            for word in search_words
        ):
            return "search"

        # fetch

        if (
            "http://" in text
            or "https://" in text
            or text.startswith("www.")
        ):
            return "fetch"

        return None

    # ========================================================
    # RESPONSE
    # ========================================================

    def _generate_response(
        self,
        message: str,
        observations: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> str:

        if not observations:

            relevant = context.get(
                "relevant_memories",
                [],
            )

            if relevant:

                first = relevant[0]

                value = first.get(
                    "value"
                )

                if value:
                    return str(value)

            return (
                "Saya memahami permintaan Anda. "
                "Namun saya belum mendapatkan hasil "
                "yang cukup untuk memberikan jawaban."
            )

        successful = [
            item
            for item in observations
            if item.get("success") is True
        ]

        if not successful:

            error = observations[-1].get(
                "error"
            )

            if error:
                return (
                    "Tool tidak berhasil memproses "
                    "permintaan tersebut."
                )

            return (
                "Tool tidak menghasilkan hasil "
                "yang dapat digunakan."
            )

        result = successful[-1]

        response = result.get(
            "response"
        )

        if response:
            return str(response)

        data = result.get(
            "data",
            {},
        )

        if data:
            return str(data)

        return (
            "Permintaan berhasil diproses, "
            "tetapi tidak ada hasil yang dapat ditampilkan."
        )

    # ========================================================
    # MAIN THINK
    # ========================================================

    def think(
        self,
        user_message: str,
        conversation: Optional[
            List[Dict[str, str]]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:

        self._runtime["thoughts"] += 1

        message = str(
            user_message or ""
        ).strip()

        conversation = list(
            conversation or []
        )

        metadata = dict(
            metadata or {}
        )

        context = self._build_context(
            message=message,
            conversation=conversation,
        )

        intent_result = self._detect_intent(
            message
        )

        intent = str(
            intent_result.get(
                "intent",
                "general",
            )
        )

        # ----------------------------------------------------
        # MEMORY INTENT
        # ----------------------------------------------------

        memory_response = self._handle_memory_intent(
            intent_result
        )

        if memory_response is not None:

            return {
                "user_message": message,
                "conversation": conversation,
                "intent": intent,
                "entities": intent_result.get(
                    "entities",
                    {},
                ),
                "plan": [
                    {
                        "step_id": 1,
                        "action": "memory",
                        "tool": None,
                        "arguments": {},
                        "description": (
                            "Baca atau simpan informasi "
                            "menggunakan MemoryManager."
                        ),
                        "required": True,
                    }
                ],
                "observations": [],
                "metadata": {
                    **metadata,
                    "agent_steps_executed": 0,
                    "memory": True,
                },
                "final_response": memory_response,
                "created_at": self._timestamp(),
                "intent_result": intent_result,
            }

        # ----------------------------------------------------
        # TOOL
        # ----------------------------------------------------

        tool = intent_result.get(
            "tool"
        )

        if not tool:

            tool = self._detect_tool_from_message(
                message
            )

        observations: List[
            Dict[str, Any]
        ] = []

        plan: List[
            Dict[str, Any]
        ] = []

        if tool:

            plan.append(
                {
                    "step_id": 1,
                    "action": "execute_tool",
                    "tool": tool,
                    "arguments": {
                        "query": message,
                    },
                    "description": (
                        f"Gunakan tool {tool} "
                        "untuk memenuhi permintaan pengguna."
                    ),
                    "required": True,
                }
            )

            tool_result = self._execute_tool(
                tool=tool,
                arguments={
                    "query": message,
                },
            )

            observations.append(
                {
                    **tool_result,
                    "timestamp": self._timestamp(),
                }
            )

            plan.append(
                {
                    "step_id": 2,
                    "action": "analyze_result",
                    "tool": None,
                    "arguments": {},
                    "description": (
                        "Analisis hasil tool sebelum "
                        "memberikan jawaban."
                    ),
                    "required": True,
                }
            )

            plan.append(
                {
                    "step_id": 3,
                    "action": "generate_response",
                    "tool": None,
                    "arguments": {},
                    "description": (
                        "Susun jawaban akhir berdasarkan "
                        "hasil dan konteks percakapan."
                    ),
                    "required": True,
                }
            )

        else:

            plan.extend(
                [
                    {
                        "step_id": 1,
                        "action": "reason",
                        "tool": None,
                        "arguments": {},
                        "description": (
                            "Analisis permintaan menggunakan AI Brain."
                        ),
                        "required": True,
                    },
                    {
                        "step_id": 2,
                        "action": "generate_response",
                        "tool": None,
                        "arguments": {},
                        "description": (
                            "Buat jawaban yang relevan dan jelas."
                        ),
                        "required": True,
                    },
                ]
            )

        final_response = self._generate_response(
            message=message,
            observations=observations,
            context=context,
        )

        return {
            "user_message": message,
            "conversation": conversation,
            "intent": intent,
            "entities": intent_result.get(
                "entities",
                {},
            ),
            "plan": plan,
            "observations": observations,
            "metadata": {
                **metadata,
                "agent_steps_executed": len(
                    observations
                ),
                "runtime": dict(
                    self._runtime
                ),
            },
            "final_response": final_response,
            "created_at": self._timestamp(),
            "intent_result": intent_result,
        }
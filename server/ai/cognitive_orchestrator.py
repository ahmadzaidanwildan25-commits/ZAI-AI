"""
Super ZAI Cognitive Orchestrator.

Central orchestration layer between:
- AIBrain
- ToolEngine
- IntentEngine
- IntentRouter
- MemoryManager
- ResponseEngine
- LLMClient

ZAI uses this layer as the public entry point
for cognitive processing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai.brain import AIBrain


class CognitiveOrchestrator:
    """
    Main cognitive orchestration layer for ZAI.

    Responsibilities:
    - receive user messages
    - delegate reasoning to AIBrain
    - connect future AI subsystems
    - expose process(), chat(), and stats()
    """

    VERSION = "0.10.1"

    def __init__(
        self,
        tool_engine: Any,
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

        self.brain = AIBrain(
            tool_engine=tool_engine,
            intent_engine=intent_engine,
            intent_router=intent_router,
            memory_manager=memory_manager,
            response_engine=response_engine,
            llm_client=llm_client,
        )

    def process(
        self,
        message: str,
        conversation: Optional[
            List[Dict[str, str]]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Process one user request through ZAI's cognitive brain.
        """

        if not isinstance(message, str):
            raise TypeError("message harus berupa string.")

        message = message.strip()

        if not message:
            return {
                "user_message": "",
                "conversation": conversation or [],
                "intent": "empty",
                "entities": {},
                "plan": [],
                "observations": [],
                "metadata": {
                    "agent_steps_executed": 0,
                },
                "final_response": (
                    "Silakan berikan perintah atau pertanyaan "
                    "yang ingin diproses oleh ZAI."
                ),
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        return self.brain.think(
            user_message=message,
            conversation=conversation,
            metadata=metadata,
        )

    def chat(
        self,
        message: str,
        conversation: Optional[
            List[Dict[str, str]]
        ] = None,
    ) -> str:
        """
        Simple text interface for ZAI.
        """

        result = self.process(
            message=message,
            conversation=conversation,
        )

        response = result.get(
            "final_response",
            "",
        )

        if response is None:
            return ""

        return str(response)

    def stats(self) -> Dict[str, Any]:
        """
        Return complete orchestrator status.
        """

        brain_stats: Dict[str, Any] = {}

        if hasattr(self.brain, "stats"):
            try:
                brain_stats = self.brain.stats()
            except Exception as exc:
                brain_stats = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        tool_stats: Dict[str, Any] = {}

        if hasattr(self.tool_engine, "stats"):
            try:
                tool_stats = self.tool_engine.stats()
            except Exception as exc:
                tool_stats = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        return {
            "engine": "CognitiveOrchestrator",
            "version": self.VERSION,
            "brain": brain_stats,
            "tool_engine": tool_stats,
            "components": {
                "intent_engine": self.intent_engine is not None,
                "intent_router": self.intent_router is not None,
                "memory_manager": self.memory_manager is not None,
                "response_engine": self.response_engine is not None,
                "llm_client": self.llm_client is not None,
            },
            "status": "READY",
        }


__all__ = [
    "CognitiveOrchestrator",
]

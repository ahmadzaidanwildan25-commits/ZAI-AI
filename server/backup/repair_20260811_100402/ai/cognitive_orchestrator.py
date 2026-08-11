"""
Super ZAI Cognitive Orchestrator.

Version 0.11.0

Pipeline:

USER
 ↓
IntentEngine
 ↓
AIBrain
 ↓
Planner / Reasoning
 ↓
ToolEngine
 ↓
Response
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai.brain import AIBrain
from intent.engine import IntentEngine
from intent.engine import get_intent_engine


class CognitiveOrchestrator:

    VERSION = "0.11.0"

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

        self.intent_engine = (
            intent_engine
            if intent_engine is not None
            else get_intent_engine()
        )

        self.intent_router = intent_router
        self.memory_manager = memory_manager
        self.response_engine = response_engine
        self.llm_client = llm_client

        self.brain = AIBrain(
            tool_engine=tool_engine,
            intent_engine=self.intent_engine,
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

        if not isinstance(
            message,
            str,
        ):
            raise TypeError(
                "message harus berupa string."
            )

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
                    "Silakan berikan "
                    "pertanyaan atau perintah."
                ),
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        intent_result = self.intent_engine.analyze(
            message
        )

        merged_metadata = dict(
            metadata or {}
        )

        merged_metadata[
            "intent_engine"
        ] = self.intent_engine.VERSION

        merged_metadata[
            "intent_result"
        ] = intent_result

        result = self.brain.think(
            user_message=message,
            conversation=conversation,
            metadata=merged_metadata,
        )

        if isinstance(
            result,
            dict,
        ):
            result.setdefault(
                "intent",
                intent_result.get(
                    "intent",
                    "general",
                ),
            )

            result.setdefault(
                "entities",
                intent_result.get(
                    "entities",
                    {},
                ),
            )

            result.setdefault(
                "intent_result",
                intent_result,
            )

        return result

    def chat(
        self,
        message: str,
        conversation: Optional[
            List[Dict[str, str]]
        ] = None,
    ) -> str:

        result = self.process(
            message=message,
            conversation=conversation,
        )

        return str(
            result.get(
                "final_response",
                "",
            )
        )

    def stats(self) -> Dict[str, Any]:

        brain_stats: Dict[str, Any] = {}

        if hasattr(
            self.brain,
            "stats",
        ):
            try:
                brain_stats = (
                    self.brain.stats()
                )
            except Exception as exc:
                brain_stats = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        tool_stats: Dict[str, Any] = {}

        if hasattr(
            self.tool_engine,
            "stats",
        ):
            try:
                tool_stats = (
                    self.tool_engine.stats()
                )
            except Exception as exc:
                tool_stats = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        intent_stats: Dict[str, Any] = {}

        if hasattr(
            self.intent_engine,
            "stats",
        ):
            try:
                intent_stats = (
                    self.intent_engine.stats()
                )
            except Exception as exc:
                intent_stats = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        return {
            "engine": "CognitiveOrchestrator",
            "version": self.VERSION,
            "intent_engine": intent_stats,
            "brain": brain_stats,
            "tool_engine": tool_stats,
            "components": {
                "intent_engine": (
                    self.intent_engine
                    is not None
                ),
                "intent_router": (
                    self.intent_router
                    is not None
                ),
                "memory_manager": (
                    self.memory_manager
                    is not None
                ),
                "response_engine": (
                    self.response_engine
                    is not None
                ),
                "llm_client": (
                    self.llm_client
                    is not None
                ),
            },
            "pipeline": [
                "intent_detection",
                "entity_extraction",
                "planning",
                "reasoning",
                "tool_execution",
                "response_generation",
            ],
            "status": "READY",
        }


__all__ = [
    "CognitiveOrchestrator",
]

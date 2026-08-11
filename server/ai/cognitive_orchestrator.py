"""
============================================================
SUPER ZAI - COGNITIVE ORCHESTRATOR
============================================================

Main orchestration layer for Super ZAI.

Version: 0.11.0
============================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.brain import AIBrain
from core.tool_engine import get_tool_engine
from memory.memory_manager import MemoryManager


class CognitiveOrchestrator:

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

        # ----------------------------------------------------
        # TOOL ENGINE
        # ----------------------------------------------------

        self.tool_engine = (
            tool_engine
            if tool_engine is not None
            else get_tool_engine()
        )

        # ----------------------------------------------------
        # OPTIONAL COMPONENTS
        # ----------------------------------------------------

        self.intent_engine = intent_engine
        self.intent_router = intent_router

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        self.memory_manager = (
            memory_manager
            if memory_manager is not None
            else MemoryManager()
        )

        # ----------------------------------------------------
        # RESPONSE / LLM
        # ----------------------------------------------------

        self.response_engine = response_engine
        self.llm_client = llm_client

        # ----------------------------------------------------
        # BRAIN
        # ----------------------------------------------------

        self.brain = AIBrain(
            tool_engine=self.tool_engine,
            intent_engine=self.intent_engine,
            intent_router=self.intent_router,
            memory_manager=self.memory_manager,
            response_engine=self.response_engine,
            llm_client=self.llm_client,
        )

    # ========================================================
    # PROCESS
    # ========================================================

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

        return self.brain.think(
            user_message=message,
            conversation=conversation,
            metadata=metadata,
        )

    # ========================================================
    # CHAT
    # ========================================================

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

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> Dict[str, Any]:

        return {
            "engine": "CognitiveOrchestrator",
            "version": self.VERSION,

            "brain": (
                self.brain.stats()
                if hasattr(self.brain, "stats")
                else {}
            ),

            "tool_engine": (
                self.tool_engine.stats()
                if hasattr(
                    self.tool_engine,
                    "stats",
                )
                else {}
            ),

            "memory_manager": (
                self.memory_manager.stats()
                if hasattr(
                    self.memory_manager,
                    "stats",
                )
                else {}
            ),

            "status": "READY",
        }

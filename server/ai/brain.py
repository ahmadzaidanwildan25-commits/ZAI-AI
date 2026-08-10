"""
Super ZAI AI Brain.

High-level cognitive layer connecting:
Context -> Intent -> Planning -> Agent -> Reasoning -> Response
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent_loop import AgentLoop
from .context import ContextManager, BrainContext
from .planner import Planner
from .reasoning import ReasoningEngine


class AIBrain:

    VERSION = "0.10.0"

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

        self.context_manager = ContextManager(
            max_messages=20
        )

        self.reasoning_engine = ReasoningEngine(
            llm_client=llm_client
        )

        self.planner = Planner(
            max_steps=8
        )

        self.agent_loop = AgentLoop(
            tool_engine=tool_engine,
            reasoning_engine=self.reasoning_engine,
            max_steps=8,
        )

    # ---------------------------------------------------------
    # INTENT
    # ---------------------------------------------------------

    def detect_intent(
        self,
        context: BrainContext,
    ) -> BrainContext:

        if self.intent_engine is None:
            context.intent = "general"
            return context

        try:

            engine = self.intent_engine

            if hasattr(engine, "detect"):
                result = engine.detect(
                    context.user_message
                )

            elif hasattr(engine, "analyze"):
                result = engine.analyze(
                    context.user_message
                )

            else:
                result = None

            if isinstance(result, dict):

                context.intent = (
                    result.get("intent")
                    or result.get("name")
                    or "general"
                )

                context.entities = (
                    result.get("entities")
                    or {}
                )

            elif result is not None:
                context.intent = str(result)

            else:
                context.intent = "general"

        except Exception as exc:

            context.metadata[
                "intent_error"
            ] = str(exc)

            context.intent = "general"

        return context

    # ---------------------------------------------------------
    # MEMORY
    # ---------------------------------------------------------

    def load_memory(
        self,
        context: BrainContext,
    ) -> BrainContext:

        if self.memory_manager is None:
            return context

        try:

            manager = self.memory_manager

            memory = None

            if hasattr(manager, "search"):
                memory = manager.search(
                    context.user_message
                )

            elif hasattr(manager, "retrieve"):
                memory = manager.retrieve(
                    context.user_message
                )

            if memory is not None:

                context.metadata[
                    "memory"
                ] = memory

        except Exception as exc:

            context.metadata[
                "memory_error"
            ] = str(exc)

        return context

    # ---------------------------------------------------------
    # PLANNING
    # ---------------------------------------------------------

    def create_plan(
        self,
        context: BrainContext,
    ) -> BrainContext:

        context.plan = self.planner.create_plan(
            user_message=context.user_message,
            intent=context.intent,
            entities=context.entities,
        )

        return context

    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------

    def execute(
        self,
        context: BrainContext,
    ) -> BrainContext:

        return self.agent_loop.execute(
            context
        )

    # ---------------------------------------------------------
    # RESPONSE
    # ---------------------------------------------------------

    def generate_response(
        self,
        context: BrainContext,
    ) -> BrainContext:

        reasoning_summary = (
            self.reasoning_engine
            .build_reasoning_summary(
                context
            )
        )

        # Existing ResponseEngine takes priority.
        if self.response_engine is not None:

            try:

                engine = self.response_engine

                if hasattr(engine, "generate"):

                    response = engine.generate(
                        context.user_message,
                        reasoning_summary,
                    )

                    if response:
                        context.final_response = str(
                            response
                        )
                        return context

                if hasattr(engine, "respond"):

                    response = engine.respond(
                        context.user_message,
                        reasoning_summary,
                    )

                    if response:
                        context.final_response = str(
                            response
                        )
                        return context

            except Exception as exc:

                context.metadata[
                    "response_engine_error"
                ] = str(exc)

        # Fallback.
        if reasoning_summary.strip():

            context.final_response = (
                reasoning_summary.strip()
            )

        else:

            context.final_response = (
                "Saya memahami permintaan Anda. "
                "Namun saya belum mendapatkan hasil "
                "yang cukup untuk memberikan jawaban."
            )

        return context

    # ---------------------------------------------------------
    # MAIN THINK CYCLE
    # ---------------------------------------------------------

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

        context = self.context_manager.create(
            user_message=user_message,
            conversation=conversation,
            metadata=metadata,
        )

        # 1. Memory
        context = self.load_memory(
            context
        )

        # 2. Intent
        context = self.detect_intent(
            context
        )

        # 3. Plan
        context = self.create_plan(
            context
        )

        # 4. Execute
        context = self.execute(
            context
        )

        # 5. Generate response
        context = self.generate_response(
            context
        )

        # 6. Trim context
        context = self.context_manager.trim(
            context
        )

        return context.to_dict()

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def stats(self) -> Dict[str, Any]:

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
            "memory_manager": self.memory_manager is not None,
            "response_engine": self.response_engine is not None,
            "llm_client": self.llm_client is not None,
            "status": "READY",
        }
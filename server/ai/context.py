"""
Super ZAI Context Manager.

Responsible for preparing the context used by the AI Brain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ConversationMessage:
    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class BrainContext:
    user_message: str
    conversation: List[ConversationMessage] = field(default_factory=list)

    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)

    plan: List[Dict[str, Any]] = field(default_factory=list)

    observations: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    final_response: Optional[str] = None

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        self.conversation.append(
            ConversationMessage(
                role=role,
                content=content,
            )
        )

    def add_observation(
        self,
        tool: str,
        success: bool,
        response: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.observations.append(
            {
                "tool": tool,
                "success": success,
                "response": response,
                "data": data or {},
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_message": self.user_message,
            "conversation": [
                {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp,
                }
                for message in self.conversation
            ],
            "intent": self.intent,
            "entities": self.entities,
            "plan": self.plan,
            "observations": self.observations,
            "metadata": self.metadata,
            "final_response": self.final_response,
            "created_at": self.created_at,
        }


class ContextManager:
    """
    Creates and manages BrainContext objects.
    """

    def __init__(
        self,
        max_messages: int = 20,
    ) -> None:
        self.max_messages = max(1, max_messages)

    def create(
        self,
        user_message: str,
        conversation: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainContext:

        context = BrainContext(
            user_message=user_message.strip(),
            metadata=metadata or {},
        )

        if conversation:
            for item in conversation[-self.max_messages:]:
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))

                if not content.strip():
                    continue

                context.add_message(
                    role=role,
                    content=content,
                )

        context.add_message(
            role="user",
            content=user_message,
        )

        return context

    def trim(
        self,
        context: BrainContext,
    ) -> BrainContext:

        if len(context.conversation) > self.max_messages:
            context.conversation = context.conversation[
                -self.max_messages:
            ]

        return context
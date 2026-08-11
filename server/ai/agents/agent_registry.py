from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class AgentRegistry:
    """
    Registry pusat seluruh agent ZAI.
    """

    VERSION = "2.1.0"

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    # ============================================================
    # REGISTER
    # ============================================================

    def register(
        self,
        agent: BaseAgent,
    ) -> None:

        if not isinstance(agent, BaseAgent):
            raise TypeError(
                "Agent harus merupakan turunan BaseAgent."
            )

        if not agent.name:
            raise ValueError(
                "Agent harus memiliki nama."
            )

        self._agents[agent.name] = agent

    # ============================================================
    # UNREGISTER
    # ============================================================

    def unregister(
        self,
        name: str,
    ) -> bool:

        if name not in self._agents:
            return False

        del self._agents[name]

        return True

    # ============================================================
    # GET
    # ============================================================

    def get(
        self,
        name: str,
    ) -> BaseAgent:

        try:
            return self._agents[name]

        except KeyError as exc:
            raise KeyError(
                f"Agent '{name}' tidak terdaftar."
            ) from exc

    # ============================================================
    # HAS
    # ============================================================

    def has(
        self,
        name: str,
    ) -> bool:
        return name in self._agents

    # ============================================================
    # ACTIVE
    # ============================================================

    def active(self) -> list[dict[str, Any]]:
        return [
            agent.info()
            for agent in self._agents.values()
        ]

    # ============================================================
    # NAMES
    # ============================================================

    def names(self) -> list[str]:
        return list(
            self._agents.keys()
        )

    # ============================================================
    # COUNT
    # ============================================================

    @property
    def count(self) -> int:
        return len(self._agents)

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self) -> None:
        self._agents.clear()

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(self) -> dict[str, Any]:
        return {
            "registry_version": self.VERSION,
            "total_agents": len(self._agents),
            "active_agents": len(self._agents),
            "agent_names": self.names(),
            "agents": self.active(),
            "status": "READY",
        }

    # ============================================================
    # ITERATION
    # ============================================================

    def __iter__(self):
        return iter(
            self._agents.values()
        )

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(
        self,
        name: str,
    ) -> bool:
        return name in self._agents
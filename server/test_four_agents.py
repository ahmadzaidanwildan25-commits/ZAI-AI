from __future__ import annotations

import asyncio

from ai.agents.runtime import AgentRuntime
from ai.agents.general_agent import GeneralAgent
from ai.agents.coding_agent import CodingAgent
from ai.agents.research_agent import ResearchAgent
from ai.agents.system_agent import SystemAgent


async def main() -> None:
    runtime = AgentRuntime()

    runtime.register_agent(
        GeneralAgent()
    )

    runtime.register_agent(
        CodingAgent()
    )

    runtime.register_agent(
        ResearchAgent()
    )

    runtime.register_agent(
        SystemAgent()
    )

    print("=== RUNTIME INFO ===")
    print(
        runtime.info()
    )

    print()
    print("=== GENERAL ===")

    general = await runtime.execute(
        "general_agent",
        "Halo ZAI",
    )

    print(
        general.to_dict()
    )

    print()
    print("=== CODING ===")

    coding = await runtime.execute(
        "coding_agent",
        'Analisis kode Python ```python\nprint("Halo ZAI")\n```',
    )

    print(
        coding.to_dict()
    )

    print()
    print("=== RESEARCH ===")

    research = await runtime.execute(
        "research_agent",
        "Riset teknologi AI untuk membangun ZAI",
    )

    print(
        research.to_dict()
    )

    print()
    print("=== SYSTEM ===")

    system = await runtime.execute(
        "system_agent",
        "Tampilkan informasi sistem komputer saya",
    )

    print(
        system.to_dict()
    )

    print()
    print(
        "FOUR_AGENT_RUNTIME_OK"
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
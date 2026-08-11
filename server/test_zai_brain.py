from __future__ import annotations

import asyncio

from ai.agents.coding_agent import CodingAgent
from ai.agents.general_agent import GeneralAgent
from ai.agents.research_agent import ResearchAgent
from ai.agents.runtime import AgentRuntime
from ai.agents.system_agent import SystemAgent
from ai.brain import ZAIBrain


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

    brain = ZAIBrain(
        runtime=runtime
    )

    print("=" * 80)
    print("ZAI BRAIN TEST")
    print("=" * 80)

    print()
    print("BRAIN INFO")
    print(brain.info())

    tasks = [
        "Halo ZAI",
        "Analisis kode Python dan cari bug",
        "Riset teknologi AI untuk membangun ZAI",
        "Cek status sistem Windows",
    ]

    results = []

    for task in tasks:
        print()
        print("-" * 80)
        print("TASK")
        print(task)

        result = await brain.execute(
            task
        )

        results.append(result)

        print()
        print("INTENT")
        print(result.intent.value)

        print()
        print("PLAN")
        print(
            result.plan.to_dict()
            if result.plan
            else None
        )

        print()
        print("STATUS")
        print(result.status.value)

        print()
        print("SUCCESS")
        print(result.success)

        print()
        print("RESPONSE")
        print(result.response)

        print()
        print("AGENT RESULT")
        print(
            result.agent_result.to_dict()
            if hasattr(
                result.agent_result,
                "to_dict",
            )
            else result.agent_result
        )

        print()
        print("OBSERVATIONS")

        for observation in (
            result.observations
        ):
            print(
                observation
            )

    print()
    print("=" * 80)
    print("BRAIN STATISTICS")
    print("=" * 80)

    print(
        brain.statistics()
    )

    assert len(results) == 4

    assert all(
        result.success
        for result in results
    )

    assert results[0].intent.value == (
        "general"
    )

    assert results[1].intent.value == (
        "coding"
    )

    assert results[2].intent.value == (
        "research"
    )

    assert results[3].intent.value == (
        "system"
    )

    assert brain.execution_count == 4

    assert brain.success_count == 4

    assert brain.failure_count == 0

    assert (
        brain.success_rate
        == 100.0
    )

    print()
    print(
        "BRAIN_MULTI_AGENT_TEST_OK"
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
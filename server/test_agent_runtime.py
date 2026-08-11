import asyncio
import json

from ai.agents import AgentHub


async def main():
    print("=" * 70)
    print("              ZAI AGENT RUNTIME TEST")
    print("=" * 70)

    hub = AgentHub()

    print("\n[1] HUB STATUS")
    print(json.dumps(
        hub.status(),
        indent=2,
        ensure_ascii=False,
    ))

    print("\n[2] REGISTERED AGENTS")

    for agent in hub.registry.active():
        print(
            f"[READY] {agent['name']} "
            f"v{agent['version']}"
        )

    tests = [
        (
            "coding_agent",
            "Buat fungsi Python sederhana untuk menjumlahkan dua angka.",
        ),
        (
            "developer_agent",
            "Rancang struktur aplikasi Python sederhana.",
        ),
        (
            "debugger_agent",
            "Analisis error Python ModuleNotFoundError.",
        ),
    ]

    for index, (agent_name, task) in enumerate(tests, start=3):
        print(f"\n[{index}] {agent_name.upper()} EXECUTION")

        try:
            result = await hub.execute(
                agent=agent_name,
                task=task,
            )

            print(json.dumps(
                result.to_dict(),
                indent=2,
                ensure_ascii=False,
            ))

        except Exception as exc:
            print(json.dumps(
                {
                    "success": False,
                    "agent": agent_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                indent=2,
                ensure_ascii=False,
            ))

    print("\n[6] RUNTIME STATISTICS")

    print(json.dumps(
        hub.runtime.stats(),
        indent=2,
        ensure_ascii=False,
    ))

    print("\n" + "=" * 70)
    print("              AGENT RUNTIME TEST SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
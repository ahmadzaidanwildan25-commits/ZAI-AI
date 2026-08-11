import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[2]

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from capabilities import CapabilityRegistry


def main():

    registry = CapabilityRegistry()

    print(registry.report())

    summary = registry.summary()

    assert summary["total"] > 0
    assert summary["active"] > 0

    brain = registry.get("brain")

    assert brain is not None
    assert brain.status == "ACTIVE"

    memory = registry.get("memory")

    assert memory is not None
    assert memory.status == "ACTIVE"

    tools = registry.get("tool_engine")

    assert tools is not None
    assert tools.status == "ACTIVE"

    output = SERVER_DIR / "capabilities" / "capabilities.json"

    registry.export(str(output))

    assert output.exists()

    print("")
    print("=" * 70)
    print("CAPABILITY REGISTRY TEST: PASS")
    print("=" * 70)
    print("")
    print(f"Exported: {output}")


if __name__ == "__main__":
    main()

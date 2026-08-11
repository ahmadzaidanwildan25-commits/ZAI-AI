import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from capabilities import CapabilityRegistry


def main():

    registry = CapabilityRegistry()

    print(registry.report())

    export_path = SERVER_DIR / "capabilities" / "capabilities.json"

    registry.export(str(export_path))

    print("")
    print(f"JSON REPORT: {export_path}")


if __name__ == "__main__":
    main()

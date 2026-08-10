import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"       {type(e).__name__}: {e}")
        return False


def foundation():
    modules = [
        "intent.engine",
        "intent.router",
        "memory.database",
        "memory.manager",
        "response.engine",
        "core.orchestrator",
        "core.tool_engine",
    ]

    for module in modules:
        importlib.import_module(module)


def intent_test():
    from intent.engine import get_intent_engine

    engine = get_intent_engine()

    tests = {
        "halo zai": "greeting",
        "siapa kamu": "identity",
        "berapa memory": "memory_count",
        "status zai": "status",
        "help": "help",
        "hitung 20 + 30": "calculation",
        "cuaca hari ini": "weather",
        "buatkan kode python": "coding",
        "cari berita terbaru": "search",
    }

    for message, expected in tests.items():
        result = engine.analyze(message)

        if result["intent"] != expected:
            raise AssertionError(
                f"{message!r}: expected {expected}, "
                f"got {result['intent']}"
            )


def memory_test():
    from memory.manager import MemoryManager

    memory = MemoryManager()

    assert memory.count() >= 0

    stats = memory.stats()

    assert stats["enabled"] is True
    assert stats["engine"] == "MemoryManager"


def router_test():
    from intent.engine import get_intent_engine
    from intent.router import IntentRouter
    from memory.manager import MemoryManager

    engine = get_intent_engine()
    router = IntentRouter(MemoryManager())

    tests = {
        "hitung 20 + 30": "calculator",
        "cuaca hari ini": "weather",
        "buatkan kode python": "llm",
        "cari berita terbaru": "search",
    }

    for message, expected_route in tests.items():
        intent = engine.analyze(message)
        result = router.route(message, intent)

        if result["route"] != expected_route:
            raise AssertionError(
                f"{message!r}: expected route "
                f"{expected_route}, got {result['route']}"
            )


def response_test():
    from memory.manager import MemoryManager
    from response.engine import ResponseEngine

    engine = ResponseEngine(MemoryManager())

    result = engine.handle(
        "greeting",
        "halo zai"
    )

    if hasattr(result, "to_dict"):
        result = result.to_dict()

    assert result["handled"] is True
    assert result["response"]


def tool_test():
    from core.tool_engine import get_tool_engine

    tools = get_tool_engine()

    assert tools.has_tool("calculator")
    assert tools.has_tool("weather")
    assert tools.has_tool("search")

    result = tools.execute(
        "calculator",
        "hitung 20 + 30"
    )

    data = result.to_dict()

    assert data["success"] is True
    assert data["data"]["result"] == 50


def security_test():
    from core.tool_engine import get_tool_engine

    tools = get_tool_engine()

    dangerous = [
        'hitung __import__("os").system("whoami")',
        'hitung open("x.txt")',
        "hitung 2 ** 999999",
    ]

    for message in dangerous:
        result = tools.execute(
            "calculator",
            message
        )

        if result.success:
            raise AssertionError(
                f"Security violation: {message}"
            )


def orchestrator_test():
    from intent.engine import get_intent_engine
    from intent.router import IntentRouter
    from memory.manager import MemoryManager
    from response.engine import ResponseEngine
    from core.orchestrator import CognitiveOrchestrator

    memory = MemoryManager()
    intent = get_intent_engine()
    router = IntentRouter(memory)
    response = ResponseEngine(memory)

    orchestrator = CognitiveOrchestrator(
        intent,
        router,
        response
    )

    tests = [
        ("halo zai", "local"),
        ("siapa kamu", "local"),
        ("berapa memory", "memory"),
        ("hitung 20 + 30", "calculator"),
        ("cuaca hari ini", "weather"),
        ("buatkan kode python", "llm"),
        ("cari berita terbaru", "search"),
    ]

    for message, expected_route in tests:
        result = orchestrator.handle(message)

        if result["route"] != expected_route:
            raise AssertionError(
                f"{message!r}: expected {expected_route}, "
                f"got {result['route']}"
            )

        if result.get("error"):
            raise AssertionError(
                f"Orchestrator error: {result['error']}"
            )


def fastapi_test():
    import main

    assert hasattr(main, "app")


def ollama_test():
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=15
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Ollama tidak tersedia."
        )

    if "qwen3:8b" not in result.stdout:
        raise RuntimeError(
            "Model qwen3:8b tidak ditemukan."
        )


def compile_test():
    files = [
        "intent/engine.py",
        "intent/router.py",
        "memory/database.py",
        "memory/manager.py",
        "response/engine.py",
        "core/orchestrator.py",
        "core/tool_engine.py",
        "main.py",
    ]

    result = subprocess.run(
        [sys.executable, "-m", "py_compile"] + files,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Python compile gagal."
        )


def main():
    print("=" * 72)
    print("                 ZAI MASTER SYSTEM CHECK")
    print("                 SUPER ZAI BACKEND")
    print("=" * 72)

    tests = [
        ("Foundation / Import seluruh engine", foundation),
        ("Intent Engine / seluruh intent utama", intent_test),
        ("Memory Engine / database + manager", memory_test),
        ("Intent Router / seluruh route utama", router_test),
        ("Response Engine / local responses", response_test),
        ("Tool Engine / calculator + weather + search", tool_test),
        ("Security / safe calculator", security_test),
        ("Cognitive Orchestrator / full pipeline", orchestrator_test),
        ("FastAPI / application import", fastapi_test),
        ("Ollama / qwen3:8b availability", ollama_test),
        ("Python Compile / seluruh core source", compile_test),
    ]

    passed = 0

    for name, fn in tests:
        if check(name, fn):
            passed += 1

    total = len(tests)

    print()
    print("=" * 72)
    print("                         HASIL MASTER CHECK")
    print("=" * 72)
    print(f"TOTAL CHECK : {total}")
    print(f"PASS        : {passed}")
    print(f"FAIL        : {total - passed}")
    print("=" * 72)

    if passed == total:
        print()
        print("SUPER ZAI MASTER CHECK: PASS")
        print("STATUS: FOUNDATION READY")
        print()
        sys.exit(0)

    print()
    print("SUPER ZAI MASTER CHECK: FAIL")
    print("STATUS: PERLU PERBAIKAN")
    print()

    sys.exit(1)


if __name__ == "__main__":
    main()

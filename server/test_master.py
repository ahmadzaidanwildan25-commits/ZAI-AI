from __future__ import annotations

import sys
import traceback


PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL

    try:
        result = fn()

        if result is False:
            raise RuntimeError("Test mengembalikan False")

        PASS += 1
        print(f"[PASS] {name}")
        return True

    except Exception as exc:
        FAIL += 1
        print(f"[FAIL] {name}")
        print(f"       {type(exc).__name__}: {exc}")
        return False


def import_core():
    from intent.engine import get_intent_engine
    from intent.router import IntentRouter
    from memory.manager import MemoryManager
    from response.engine import ResponseEngine
    from core.orchestrator import CognitiveOrchestrator
    from core.tool_engine import get_tool_engine

    return (
        get_intent_engine,
        IntentRouter,
        MemoryManager,
        ResponseEngine,
        CognitiveOrchestrator,
        get_tool_engine,
    )


def main():
    print()
    print("=" * 72)
    print("                    ZAI MASTER SYSTEM TEST")
    print("                    SUPER ZAI BACKEND")
    print("=" * 72)

    # ------------------------------------------------------------
    # 1. IMPORT / FOUNDATION
    # ------------------------------------------------------------

    holder = {}

    def foundation():
        imports = import_core()
        holder["imports"] = imports

        (
            get_intent_engine,
            IntentRouter,
            MemoryManager,
            ResponseEngine,
            CognitiveOrchestrator,
            get_tool_engine,
        ) = imports

        holder["intent"] = get_intent_engine()
        holder["memory"] = MemoryManager()
        holder["router"] = IntentRouter(holder["memory"])
        holder["response"] = ResponseEngine(holder["memory"])
        holder["orchestrator"] = CognitiveOrchestrator(
            holder["intent"],
            holder["router"],
            holder["response"],
        )
        holder["tools"] = get_tool_engine()

        return True

    test("Foundation / Import seluruh engine", foundation)

    # ------------------------------------------------------------
    # 2. INTENT ENGINE
    # ------------------------------------------------------------

    def intent_test():
        engine = holder["intent"]

        expected = {
            "halo zai": "greeting",
            "siapa kamu": "identity",
            "berapa memory": "memory_count",
            "status zai": "status",
            "help": "help",
            "hitung 20 + 30": "calculation",
            "cuaca hari ini": "weather",
            "cuaca sekarang": "weather",
            "buatkan kode python": "coding",
            "cari berita terbaru": "search",
        }

        for message, expected_intent in expected.items():
            result = engine.analyze(message)

            if result["intent"] != expected_intent:
                raise AssertionError(
                    f"{message!r}: expected={expected_intent!r}, "
                    f"got={result['intent']!r}"
                )

        return True

    test("Intent Engine / seluruh intent utama", intent_test)

    # ------------------------------------------------------------
    # 3. MEMORY ENGINE
    # ------------------------------------------------------------

    def memory_test():
        memory = holder["memory"]

        required = [
            "count",
            "stats",
            "save",
            "delete",
            "build_context",
        ]

        for name in required:
            if not hasattr(memory, name):
                raise AssertionError(f"MemoryManager tidak memiliki {name}()")

        stats = memory.stats()

        if not isinstance(stats, dict):
            raise AssertionError("stats() harus mengembalikan dict")

        if not stats.get("enabled", False):
            raise AssertionError("Memory harus ENABLED")

        if memory.count() < 0:
            raise AssertionError("Memory count invalid")

        return True

    test("Memory Engine / database + manager", memory_test)

    # ------------------------------------------------------------
    # 4. ROUTER
    # ------------------------------------------------------------

    def router_test():
        router = holder["router"]
        engine = holder["intent"]

        tests = {
            "cuaca hari ini": "weather",
            "hitung 20 + 30": "calculator",
            "buatkan kode python": "llm",
            "cari berita terbaru": "search",
        }

        for message, expected_route in tests.items():
            intent = engine.analyze(message)
            result = router.route(message, intent)

            if result["route"] != expected_route:
                raise AssertionError(
                    f"{message!r}: expected route={expected_route!r}, "
                    f"got={result.get('route')!r}"
                )

        return True

    test("Intent Router / seluruh route utama", router_test)

    # ------------------------------------------------------------
    # 5. RESPONSE ENGINE
    # ------------------------------------------------------------

    def response_test():
        response = holder["response"]

        tests = {
            "greeting": "halo zai",
            "identity": "siapa kamu",
            "memory_count": "berapa memory",
            "status": "status zai",
            "help": "help",
        }

        for intent_name, message in tests.items():
            result = response.handle(intent_name, message)

            if not result:
                raise AssertionError(f"Response kosong untuk {intent_name}")

            if not result.to_dict()["handled"]:
                raise AssertionError(
                    f"Response {intent_name} seharusnya handled=True"
                )

        return True

    test("Response Engine / local responses", response_test)

    # ------------------------------------------------------------
    # 6. TOOL ENGINE
    # ------------------------------------------------------------

    def tool_test():
        tools = holder["tools"]

        stats = tools.stats()

        if not stats.get("calculator"):
            raise AssertionError("Calculator tool tidak tersedia")

        if not stats.get("weather"):
            raise AssertionError("Weather tool tidak tersedia")

        if not stats.get("search"):
            raise AssertionError("Search tool tidak tersedia")

        calculations = {
            "hitung 20 + 30": 50,
            "hitung 100 * 5": 500,
            "hitung (20 + 30) * 2": 100,
            "hitung 100 / 4": 25,
            "hitung 2 ^ 10": 1024,
        }

        for message, expected in calculations.items():
            result = tools.execute("calculator", message).to_dict()

            if not result["success"]:
                raise AssertionError(
                    f"Calculator gagal: {message}"
                )

            actual = result["data"]["result"]

            if actual != expected:
                raise AssertionError(
                    f"{message}: expected={expected}, got={actual}"
                )

        return True

    test("Tool Engine / calculator + weather + search", tool_test)

    # ------------------------------------------------------------
    # 7. COGNITIVE ORCHESTRATOR
    # ------------------------------------------------------------

    def orchestrator_test():
        orchestrator = holder["orchestrator"]

        stats = orchestrator.stats()

        if stats.get("status") != "READY":
            raise AssertionError(
                f"Orchestrator status={stats.get('status')}"
            )

        tests = [
            "halo zai",
            "siapa kamu",
            "berapa memory",
            "status zai",
            "help",
            "hitung 20 + 30",
            "cuaca hari ini",
            "buatkan kode python",
            "cari berita terbaru",
        ]

        for message in tests:
            result = orchestrator.handle(message)

            if not isinstance(result, dict):
                raise AssertionError(
                    f"Result bukan dict untuk {message!r}"
                )

            if "intent" not in result:
                raise AssertionError(
                    f"intent tidak ada untuk {message!r}"
                )

            if "route" not in result:
                raise AssertionError(
                    f"route tidak ada untuk {message!r}"
                )

            if "error" not in result:
                raise AssertionError(
                    f"error tidak ada untuk {message!r}"
                )

        return True

    test("Cognitive Orchestrator / full pipeline", orchestrator_test)

    # ------------------------------------------------------------
    # 8. SECURITY / SAFE CALCULATOR
    # ------------------------------------------------------------

    def security_test():
        tools = holder["tools"]

        dangerous = [
            'hitung __import__("os").system("whoami")',
            'hitung open("secret.txt")',
            'hitung eval("2+2")',
            "hitung 2 ** 999999",
        ]

        for message in dangerous:
            result = tools.execute("calculator", message).to_dict()

            if result.get("success") is True:
                raise AssertionError(
                    f"Input berbahaya diterima: {message}"
                )

        return True

    test("Security / safe calculator", security_test)

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    total = PASS + FAIL

    print()
    print("=" * 72)
    print("                         HASIL MASTER TEST")
    print("=" * 72)
    print(f"TOTAL TEST : {total}")
    print(f"PASS       : {PASS}")
    print(f"FAIL       : {FAIL}")
    print("=" * 72)

    if FAIL == 0:
        print()
        print("SUPER ZAI MASTER TEST: PASS")
        print("STATUS: FOUNDATION READY")
        print()
        return 0

    print()
    print("SUPER ZAI MASTER TEST: FAILED")
    print("STATUS: NEED FIX")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
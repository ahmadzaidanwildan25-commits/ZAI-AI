from __future__ import annotations

import asyncio

from ai.tools import (
    AsyncToolCallable,
    ToolExecutionStatus,
    ToolRegistry,
)


def test_imports() -> None:
    assert ToolRegistry is not None
    assert ToolExecutionStatus is not None
    assert AsyncToolCallable is not None

    print("IMPORTS_OK")


def test_zero_argument() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "hello",
        lambda: "Halo ZAI",
        description="Tool greeting.",
    )

    registry.whitelist("hello")

    result = registry.execute(
        "hello"
    )

    print(result.to_dict())

    assert result.success is True
    assert result.response == "Halo ZAI"
    assert (
        result.status
        == ToolExecutionStatus.COMPLETED
    )

    print("ZERO_ARGUMENT_TOOL_OK")


def test_arguments() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "add",
        lambda a, b: a + b,
        description="Menjumlahkan dua angka.",
        permissions=["read"],
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": [
                "a",
                "b",
            ],
            "additionalProperties": False,
        },
    )

    registry.whitelist("add")

    result = registry.execute(
        "add",
        arguments={
            "a": 10,
            "b": 20,
        },
        permissions=["read"],
    )

    print(result.to_dict())

    assert result.success is True
    assert result.response == 30

    print("TOOL_ARGUMENT_VALIDATION_OK")


def test_invalid_arguments() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "add",
        lambda a, b: a + b,
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": [
                "a",
                "b",
            ],
            "additionalProperties": False,
        },
    )

    result = registry.execute(
        "add",
        arguments={
            "a": 10
        },
    )

    print(result.to_dict())

    assert result.success is False
    assert result.error is not None

    print("INVALID_ARGUMENT_OK")


def test_wrong_type() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "add",
        lambda a, b: a + b,
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": [
                "a",
                "b",
            ],
        },
    )

    result = registry.execute(
        "add",
        arguments={
            "a": "10",
            "b": 20,
        },
    )

    print(result.to_dict())

    assert result.success is False

    print("TYPE_VALIDATION_OK")


def test_permission() -> None:
    registry = ToolRegistry(
        strict_permissions=True
    )

    registry.register_function(
        "admin_tool",
        lambda: "ADMIN",
        permissions=["admin"],
    )

    denied = registry.execute(
        "admin_tool"
    )

    print(denied.to_dict())

    assert denied.success is False
    assert (
        denied.status
        == ToolExecutionStatus.DENIED
    )

    allowed = registry.execute(
        "admin_tool",
        permissions=["admin"],
    )

    print(allowed.to_dict())

    assert allowed.success is True
    assert allowed.response == "ADMIN"

    print("PERMISSION_SYSTEM_OK")


def test_whitelist() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "safe_tool",
        lambda: "SAFE",
        whitelist_required=True,
    )

    denied = registry.execute(
        "safe_tool"
    )

    assert denied.success is False
    assert (
        denied.status
        == ToolExecutionStatus.DENIED
    )

    registry.whitelist(
        "safe_tool"
    )

    allowed = registry.execute(
        "safe_tool"
    )

    assert allowed.success is True
    assert allowed.response == "SAFE"

    print("WHITELIST_OK")


async def async_add(
    a: int,
    b: int,
) -> int:
    await asyncio.sleep(0.01)

    return a + b


async def test_async_tool() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "async_add",
        async_add,
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": [
                "a",
                "b",
            ],
        },
    )

    result = await registry.execute_async(
        "async_add",
        arguments={
            "a": 100,
            "b": 200,
        },
    )

    print(result.to_dict())

    assert result.success is True
    assert result.response == 300

    print("ASYNC_TOOL_OK")


async def test_sync_from_async() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "sync_add",
        lambda a, b: a + b,
    )

    result = await registry.execute_async(
        "sync_add",
        arguments={
            "a": 5,
            "b": 7,
        },
    )

    print(result.to_dict())

    assert result.success is True
    assert result.response == 12

    print("SYNC_FROM_ASYNC_OK")


def test_statistics() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "ping",
        lambda: "pong",
    )

    for _ in range(3):
        result = registry.execute(
            "ping"
        )

        assert result.success is True

    stats = registry.statistics()

    print(stats)

    assert (
        stats["execution_count"]
        == 3
    )

    assert (
        stats["success_count"]
        == 3
    )

    assert (
        stats["failure_count"]
        == 0
    )

    assert (
        stats["success_rate"]
        == 100.0
    )

    print("STATISTICS_OK")


def test_history() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "hello",
        lambda: "hello",
    )

    registry.execute(
        "hello"
    )

    history = registry.history()

    print(history)

    assert len(history) == 1
    assert (
        history[0]["tool"]
        == "hello"
    )

    print("HISTORY_OK")


def test_health() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "hello",
        lambda: "hello",
    )

    health = registry.health()

    print(health)

    assert health[
        "status"
    ] == "HEALTHY"

    print("HEALTH_OK")


def test_batch() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "add",
        lambda a, b: a + b,
    )

    results = registry.execute_many(
        [
            {
                "name": "add",
                "arguments": {
                    "a": 1,
                    "b": 2,
                },
            },
            {
                "name": "add",
                "arguments": {
                    "a": 10,
                    "b": 20,
                },
            },
        ]
    )

    assert len(results) == 2
    assert results[0].response == 3
    assert results[1].response == 30

    print("BATCH_EXECUTION_OK")


async def test_async_batch() -> None:
    registry = ToolRegistry()

    registry.register_function(
        "add",
        lambda a, b: a + b,
    )

    results = (
        await registry.execute_many_async(
            [
                {
                    "name": "add",
                    "arguments": {
                        "a": 1,
                        "b": 2,
                    },
                },
                {
                    "name": "add",
                    "arguments": {
                        "a": 10,
                        "b": 20,
                    },
                },
            ]
        )
    )

    assert len(results) == 2
    assert results[0].response == 3
    assert results[1].response == 30

    print("ASYNC_BATCH_EXECUTION_OK")


def main() -> None:
    print("=" * 60)
    print("ZAI TOOL PLATFORM TEST")
    print("=" * 60)

    test_imports()
    test_zero_argument()
    test_arguments()
    test_invalid_arguments()
    test_wrong_type()
    test_permission()
    test_whitelist()
    test_statistics()
    test_history()
    test_health()
    test_batch()

    asyncio.run(
        test_async_tool()
    )

    asyncio.run(
        test_sync_from_async()
    )

    asyncio.run(
        test_async_batch()
    )

    print("=" * 60)
    print("ZAI_TOOL_PLATFORM_ALL_TESTS_OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
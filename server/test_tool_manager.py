from __future__ import annotations

import asyncio

from ai.tools import ToolManager


async def main() -> None:
    manager = ToolManager()

    async def async_add(
        a: int,
        b: int,
    ) -> int:
        await asyncio.sleep(0)
        return a + b

    manager.register_function(
        "async_add",
        async_add,
        description="Async addition",
        category="math",
        schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                },
                "b": {
                    "type": "integer",
                },
            },
            "required": [
                "a",
                "b",
            ],
            "additionalProperties": False,
        },
    )

    manager.whitelist(
        "async_add"
    )

    result = await manager.execute_async(
        "async_add",
        arguments={
            "a": 100,
            "b": 200,
        },
    )

    print(
        result.to_dict()
    )

    assert result.success is True
    assert result.response == 300

    print(
        "TOOL_MANAGER_ASYNC_OK"
    )


if __name__ == "__main__":
    asyncio.run(main())
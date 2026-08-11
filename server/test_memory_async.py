import asyncio

from ai.memory.memory_manager import MemoryManager


async def main():
    manager = MemoryManager(
        "data/memory/test_async.json"
    )

    memory = await manager.acreate(
        "Async memory ZAI",
        memory_type="context",
        importance=0.7,
    )

    print(memory.to_dict())

    results = await manager.asearch(
        "Async memory",
        limit=10,
    )

    print(len(results))

    assert len(results) > 0

    print("MEMORY_ASYNC_OK")


asyncio.run(main())

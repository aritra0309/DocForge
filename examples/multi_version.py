"""Index PostgreSQL versions side by side and search one version."""

import asyncio

from docforge import DocForge


async def main() -> None:
    async with DocForge() as forge:
        for version in ("16", "17"):
            await forge.index("postgresql", version=version)

        results = await forge.search(
            "logical replication", software="postgresql", version="17", k=5
        )

    for result in results:
        print(f"v{result.metadata.version}: {result.metadata.title}")  # ruff: ignore[print]


if __name__ == "__main__":
    asyncio.run(main())

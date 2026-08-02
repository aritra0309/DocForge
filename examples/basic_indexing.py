"""Index PostgreSQL documentation, then search it.

Run after ``pip install docforge``. The first run downloads the configured
embedding model and indexes the documentation site.
"""

import asyncio

from docforge import DocForge


async def main() -> None:
    async with DocForge() as forge:
        await forge.index("postgresql", version="17")
        results = await forge.search(
            "How do I create an index?", software="postgresql", version="17", k=3
        )

    for result in results:
        print(f"{result.score:.3f} {result.metadata.title}\n{result.metadata.url}\n")  # ruff: ignore[print]


if __name__ == "__main__":
    asyncio.run(main())

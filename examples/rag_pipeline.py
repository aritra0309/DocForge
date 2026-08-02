"""Use DocForge search results as context for an OpenAI response.

Install optional dependency first: ``pip install 'docforge[openai]'``.
Set ``OPENAI_API_KEY`` before running.
"""

import asyncio

from openai import AsyncOpenAI

from docforge import DocForge


async def main() -> None:
    question = "How do I create an index in PostgreSQL?"
    async with DocForge() as forge:
        results = await forge.search(question, software="postgresql", k=4)

    context = "\n\n".join(f"Source: {result.metadata.url}\n{result.content}" for result in results)
    client = AsyncOpenAI()
    response = await client.responses.create(
        model="gpt-4.1-mini",
        input=(
            "Answer using only supplied documentation context. Cite source URLs.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        ),
    )
    print(response.output_text)  # ruff: ignore[print]


if __name__ == "__main__":
    asyncio.run(main())

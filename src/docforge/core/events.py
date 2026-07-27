from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

Handler = Callable[..., Any] | Callable[..., Awaitable[Any]]

PIPELINE_STARTED = "pipeline.started"
PIPELINE_COMPLETED = "pipeline.completed"
PIPELINE_ERROR = "pipeline.error"
PIPELINE_PROGRESS = "pipeline.progress"
DISCOVERY_STARTED = "discovery.started"
DISCOVERY_COMPLETED = "discovery.completed"
CRAWL_STARTED = "crawl.started"
CRAWL_PAGE_FETCHED = "crawl.page.fetched"
CRAWL_PAGE_SKIPPED = "crawl.page.skipped"
CRAWL_PAGE_FAILED = "crawl.page.failed"
CRAWL_COMPLETED = "crawl.completed"
EXTRACTION_STARTED = "extraction.started"
EXTRACTION_COMPLETED = "extraction.completed"
CLASSIFICATION_COMPLETED = "classification.completed"
CHUNKING_COMPLETED = "chunking.completed"
METADATA_GENERATED = "metadata.generated"
EMBEDDING_STARTED = "embedding.started"
EMBEDDING_BATCH_COMPLETED = "embedding.batch.completed"
EMBEDDING_COMPLETED = "embedding.completed"
STORAGE_UPSERTED = "storage.upserted"
VERSION_COMPLETED = "pipeline.version.completed"
UPDATE_STARTED = "update.started"
UPDATE_COMPLETED = "update.completed"
UPDATE_PAGE_REMOVED = "update.page.removed"
UPDATE_PAGE_SKIPPED = "update.page.skipped"
UPDATE_PAGE_REINDEXED = "update.page.reindexed"
UPDATE_CHUNK_DIFFED = "update.chunk.diffed"


@dataclass
class PipelineEvent:
    type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}

    def on(self, event_type: str, handler: Handler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Handler | None = None) -> None:
        if handler is None:
            self._handlers.pop(event_type, None)
        else:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    async def emit(self, event_type: str, **data: Any) -> None:
        event = PipelineEvent(type=event_type, data=data)
        for handler in self._handlers.get(event_type, []):
            result = handler(event)
            if isinstance(result, Awaitable):
                await result

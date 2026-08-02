"""DocForge CLI — built with Typer and Rich."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from docforge._version import __version__
from docforge.core.config import DocForgeConfig, load_config
from docforge.core.models import SearchResult
from docforge.core.pipeline import Pipeline, PipelineResult, PipelineVersionResult
from docforge.discovery.registry import load_registry
from docforge.embeddings.engine import EmbeddingEngine
from docforge.embeddings.providers.sentence_transformers import SentenceTransformersProvider
from docforge.storage.engine import StorageEngine
from docforge.storage.metadata_store import MetadataStore

MAX_TITLE_LENGTH = 80

app = typer.Typer(
    name="docforge",
    help="Automatically discover, crawl, version, chunk, and index software documentation into a RAG-ready knowledge base.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


def _run_async(coro: Any) -> Any:
    """Run an async coroutine, including when invoked from an active event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        result: list[Any] = []
        errors: list[BaseException] = []

        def run_in_thread() -> None:
            try:
                result.append(asyncio.run(coro))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]
        return result[0]
    return asyncio.run(coro)


def _get_metadata_store(config: DocForgeConfig) -> MetadataStore:
    """Get metadata store from config."""
    store_cfg = config.storage
    meta_db_path = store_cfg.path / "metadata.db"
    return MetadataStore(meta_db_path)


def _create_storage_engine(config: DocForgeConfig, software: str, version: str) -> StorageEngine:
    """Create and initialize a storage engine."""
    engine = StorageEngine(config, software=software, version=version)
    return engine


def _print_pipeline_result(result: PipelineResult, console_obj: Console) -> None:
    """Print pipeline result summary."""
    if result.status == "completed":
        console_obj.print("\n[bold green]✓ Pipeline completed successfully[/bold green]")
    elif result.status == "partial":
        console_obj.print("\n[bold yellow]⚠ Pipeline completed with some failures[/bold yellow]")
    else:
        console_obj.print("\n[bold red]✗ Pipeline failed[/bold red]")
        if result.error:
            console_obj.print(f"  Error: {result.error}")

    total_duration_ms = result.total_duration_ms
    total_duration = total_duration_ms / 1000 if isinstance(total_duration_ms, int | float) else 0.0
    console_obj.print(f"  Software: [cyan]{result.software}[/cyan]")
    console_obj.print(f"  Total Duration: [yellow]{total_duration:.2f}s[/yellow]")
    console_obj.print(f"  Versions Processed: [magenta]{len(result.versions)}[/magenta]")

    for vr in result.versions:
        console_obj.print(f"\n  [bold]Version: {vr.version}[/bold] ({vr.status})")
        if vr.error:
            console_obj.print(f"    Error: {vr.error}")

        stages = [
            ("Discovery", vr.discovery),
            ("Crawl", vr.crawl),
            ("Extraction", vr.extraction),
            ("Classification", vr.classification),
            ("Chunking", vr.chunking),
            ("Metadata", vr.metadata),
            ("Embedding", vr.embedding),
            ("Storage", vr.storage),
        ]

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Stage", style="cyan")
        table.add_column("Pages", justify="right", style="magenta")
        table.add_column("Chunks", justify="right", style="blue")
        table.add_column("Duration", justify="right", style="yellow")

        for name, stats in stages:
            table.add_row(
                name,
                str(stats.pages_processed) if stats.pages_processed else "—",
                str(stats.chunks_produced) if stats.chunks_produced else "—",
                (
                    f"{stats.duration_ms / 1000:.2f}s"
                    if isinstance(stats.duration_ms, int | float) and stats.duration_ms
                    else "—"
                ),
            )
        console_obj.print(table)


def _print_search_results(results: list[SearchResult], console_obj: Console) -> None:
    """Print search results in a formatted table."""
    if not results:
        console_obj.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Search Results ({len(results)} found)", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Score", style="green", width=8, justify="right")
    table.add_column("Software", style="cyan", width=15)
    table.add_column("Version", style="yellow", width=10)
    table.add_column("Title", style="bold white")
    table.add_column("URL", style="blue dim")

    for i, r in enumerate(results, 1):
        meta = r.metadata
        title = (
            meta.title[:MAX_TITLE_LENGTH] + "..."
            if len(meta.title) > MAX_TITLE_LENGTH
            else meta.title
        )
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            meta.software,
            meta.version,
            title,
            meta.url,
        )

    console_obj.print(table)
    for result in results:
        console_obj.print(result.content)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: Annotated[
        bool, typer.Option("--version", "-v", help="Show version and exit")
    ] = False,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """DocForge — Automatically discover, crawl, version, chunk, and index software documentation."""
    if version_flag:
        console.print(f"DocForge v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def index(
    software: Annotated[str, typer.Argument(help="Software name to index (e.g., postgresql)")],
    version: Annotated[
        str | None, typer.Option("--version", "-V", help="Specific version to index")
    ] = None,
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="Pipeline mode: full, incremental, reembed")
    ] = "full",
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Run the full indexing pipeline for a software package."""
    config = load_config(project_config_path=config_file) if config_file else load_config()
    pipeline = Pipeline(config)

    async def _run_index() -> PipelineResult:
        return await pipeline.run(software=software, version=version, mode=mode)

    console.print(f"[bold blue]Starting {mode} index for [green]{software}[/green]")
    if version:
        console.print(f"  Version: [yellow]{version}[/yellow]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(f"Indexing {software}...", total=None)
        try:
            result = _run_async(_run_index())
            progress.update(
                task, completed=True, description=f"[green]Indexing {software} complete[/green]"
            )
        except Exception as e:
            progress.update(task, completed=True, description=f"[red]Indexing failed: {e}[/red]")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

    _print_pipeline_result(result, console)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    software: Annotated[
        str | None, typer.Option("--software", "-s", help="Filter by software")
    ] = None,
    version: Annotated[
        str | None, typer.Option("--version", "-V", help="Filter by version")
    ] = None,
    k: Annotated[
        int, typer.Option("--top-k", "--k", "-k", help="Number of results to return")
    ] = 10,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Search the indexed documentation using semantic search."""
    config = load_config(project_config_path=config_file) if config_file else load_config()

    async def _run_search() -> list[SearchResult]:
        provider = SentenceTransformersProvider(model_name=config.embeddings.model)
        embedding_engine = EmbeddingEngine(
            provider=provider, batch_size=config.embeddings.batch_size
        )

        if software is None:
            console.print("[red]Error: --software is required for search[/red]")
            raise typer.Exit(1)

        ver = version
        if ver is None or ver == "latest":
            registry = load_registry()
            entry = registry.lookup(software)
            if entry:
                ver = entry.latest_version
        else:
            console.print(f"[red]Error: Software '{software}' not found in registry[/red]")
            raise typer.Exit(1)

        storage = StorageEngine(config, software=software, version=ver)
        await storage.initialize(dimension=provider.dimension, model_name=provider.model_name)

        query_vector = (await embedding_engine.provider.embed_batch([query]))[0]
        results = await storage.search(query_vector, k=k)

        await storage.close()
        await embedding_engine.close()
        return results

    console.print(f"[bold blue]Searching for:[/bold blue] [green]{query}[/green]")
    if software:
        console.print(f"  Software: [yellow]{software}[/yellow]")
    if version:
        console.print(f"  Version: [yellow]{version}[/yellow]")

    try:
        results = _run_async(_run_search())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    _print_search_results(results, console)


@app.command()
def update(
    software: Annotated[
        str | None, typer.Argument(help="Software name to update (optional, updates all)")
    ] = None,
    version: Annotated[
        str | None, typer.Option("--version", "-V", help="Specific version to update")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Incrementally update indexed documentation (only changed pages)."""
    config = load_config(project_config_path=config_file) if config_file else load_config()
    pipeline = Pipeline(config)

    async def _run_update() -> PipelineResult:
        if software:
            return await pipeline.run(software=software, version=version, mode="incremental")

        meta_store = _get_metadata_store(config)
        indexed = meta_store.list_software()
        meta_store.close()

        if not indexed:
            console.print("[yellow]No software indexed yet. Run 'docforge index' first.[/yellow]")
            raise typer.Exit(0)

        overall = PipelineResult(software="all", status="completed")
        for sw in indexed:
            console.print(f"[blue]Updating {sw['software']}...[/blue]")
            result = await pipeline.run(
                software=sw["software"], version=version, mode="incremental"
            )
            overall.versions.extend(result.versions)
            if result.status == "failed":
                overall.status = "partial"
        return overall

    console.print("[bold blue]Running incremental update[/bold blue]")
    if software:
        console.print(f"  Software: [yellow]{software}[/yellow]")
    if version:
        console.print(f"  Version: [yellow]{version}[/yellow]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Updating...", total=None)
        try:
            result = _run_async(_run_update())
            progress.update(task, completed=True, description="[green]Update complete[/green]")
        except Exception as e:
            progress.update(task, completed=True, description=f"[red]Update failed: {e}[/red]")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

    _print_pipeline_result(result, console)


@app.command()
def reembed(
    software: Annotated[str, typer.Argument(help="Software name to re-embed")],
    new_model: Annotated[str, typer.Option("--model", "-m", help="New embedding model to use")],
    version: Annotated[
        str | None, typer.Option("--version", "-V", help="Specific version to re-embed")
    ] = None,
    old_model: Annotated[
        str | None,
        typer.Option("--old-model", help="Old embedding model (auto-detected if not provided)"),
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Re-embed existing chunks with a different model (no re-crawl)."""
    config = load_config(project_config_path=config_file) if config_file else load_config()
    pipeline = Pipeline(config)

    async def _run_reembed() -> PipelineResult:
        kw: dict[str, Any] = {}
        if old_model:
            kw["old_model"] = old_model
        kw["new_model"] = new_model
        return await pipeline.run(software=software, version=version, mode="reembed", **kw)

    console.print(
        f"[bold blue]Re-embedding [green]{software}[/green] with model [yellow]{new_model}[/yellow]"
    )
    if version:
        console.print(f"  Version: [yellow]{version}[/yellow]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Re-embedding...", total=None)
        try:
            result = _run_async(_run_reembed())
            progress.update(
                task, completed=True, description="[green]Re-embedding complete[/green]"
            )
        except Exception as e:
            progress.update(
                task, completed=True, description=f"[red]Re-embedding failed: {e}[/red]"
            )
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

    _print_pipeline_result(result, console)


@app.command(name="list")
def list_cmd(
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """List all indexed software with their versions."""
    config = load_config(project_config_path=config_file) if config_file else load_config()
    meta_store = _get_metadata_store(config)
    indexed = meta_store.list_software()
    meta_store.close()

    if not indexed:
        console.print(
            "[yellow]No software indexed yet. Run 'docforge index <software>' to get started.[/yellow]"
        )
        return

    table = Table(title="Indexed Software", box=box.ROUNDED)
    table.add_column("Software", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="green")
    table.add_column("Versions", style="yellow")
    table.add_column("Total Pages", style="magenta", justify="right")
    table.add_column("Total Chunks", style="blue", justify="right")
    table.add_column("Last Indexed", style="dim")

    for sw in indexed:
        software_name = sw["software"]
        meta_store = _get_metadata_store(config)
        versions = meta_store.list_versions(software_name)
        meta_store.close()

        if versions:
            version_strs = ", ".join(v["version"] for v in versions)
            total_pages = sum(v["page_count"] for v in versions)
            total_chunks = sum(v["chunk_count"] for v in versions)
            last_indexed = max(v["indexed_at"] for v in versions) if versions else "N/A"
        else:
            version_strs = "—"
            total_pages = 0
            total_chunks = 0
            last_indexed = sw.get("last_indexed_at", "N/A")

        table.add_row(
            software_name,
            sw["display_name"],
            version_strs,
            str(total_pages),
            str(total_chunks),
            last_indexed,
        )

    console.print(table)


@app.command()
def stats(
    software: Annotated[str, typer.Argument(help="Software name to show stats for")],
    version: Annotated[
        str | None, typer.Option("--version", "-V", help="Specific version (default: latest)")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Show detailed statistics for indexed software."""
    config = load_config(project_config_path=config_file) if config_file else load_config()
    meta_store = _get_metadata_store(config)

    stats_data = meta_store.get_software_stats(software)
    if stats_data["version_count"] == 0:
        console.print(f"[red]Software '{software}' not indexed[/red]")
        meta_store.close()
        raise typer.Exit(1)

    versions = meta_store.list_versions(software)
    meta_store.close()

    console.print(f"[bold blue]Statistics for [green]{software}[/green][/bold blue]\n")

    table = Table(title=f"Versions for {software}", box=box.ROUNDED)
    table.add_column("Version", style="cyan", no_wrap=True)
    table.add_column("Pages", style="magenta", justify="right")
    table.add_column("Chunks", style="blue", justify="right")
    table.add_column("Embedding Model", style="green")
    table.add_column("Dimension", style="yellow", justify="right")
    table.add_column("Indexed At", style="dim")

    for v in versions:
        table.add_row(
            v["version"],
            str(v["page_count"]),
            str(v["chunk_count"]),
            v.get("embedding_model", "—"),
            str(v.get("embedding_dimension", "—")),
            v["indexed_at"],
        )

    console.print(table)
    console.print()
    console.print(f"  Total Versions: [bold]{stats_data['version_count']}[/bold]")
    console.print(f"  Total Pages: [bold]{stats_data['page_count']}[/bold]")
    console.print(f"  Total Chunks: [bold]{stats_data['chunk_count']}[/bold]")


@app.command()
def delete(
    software: Annotated[str, typer.Argument(help="Software name to delete")],
    version: Annotated[
        str | None,
        typer.Option("--version", "-V", help="Specific version to delete (default: all versions)"),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "--yes", "-f", help="Skip confirmation")
    ] = False,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Delete indexed software or a specific version."""
    config = load_config(project_config_path=config_file) if config_file else load_config()
    meta_store = _get_metadata_store(config)

    if not meta_store.get_software(software):
        console.print(f"[red]Software '{software}' not indexed[/red]")
        meta_store.close()
        raise typer.Exit(1)

    async def _delete_version(storage_engine: StorageEngine, version_to_delete: str) -> None:
        await storage_engine.initialize()
        await storage_engine.delete(filters={"software": software, "version": version_to_delete})
        await storage_engine.close()

    if version:
        if not meta_store.get_version(software, version):
            console.print(f"[red]Version '{version}' not found for '{software}'[/red]")
            meta_store.close()
            raise typer.Exit(1)

        if not force:
            confirm = typer.confirm(f"Delete version {version} of {software}?")
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                meta_store.close()
                return

        storage = StorageEngine(config, software=software, version=version)

        _run_async(_delete_version(storage, version))
        meta_store.delete_version(software, version)
        console.print(f"[green]Deleted version {version} of {software}[/green]")
    else:
        if not force:
            confirm = typer.confirm(f"Delete ALL versions of {software}?")
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                meta_store.close()
                return

        versions = meta_store.list_versions(software)
        for v in versions:
            ver_str = v["version"]
            storage = StorageEngine(config, software=software, version=ver_str)

            _run_async(_delete_version(storage, ver_str))

        meta_store.delete_software(software)
        console.print(f"[green]Deleted all versions of {software}[/green]")

    meta_store.close()


@app.command()
def config(
    show_source: Annotated[
        bool, typer.Option("--source", help="Show config source (file, env, default)")
    ] = False,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Show current configuration with source annotations."""
    config = load_config(project_config_path=config_file) if config_file else load_config()

    table = Table(title="DocForge Configuration", box=box.ROUNDED)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Key", style="green", no_wrap=True)
    table.add_column("Value", style="yellow")
    if show_source:
        table.add_column("Source", style="dim")

    # General
    g = config.general
    table.add_row("general", "data_dir", str(g.data_dir))
    table.add_row("general", "log_level", g.log_level)
    table.add_row("general", "parallelism", str(g.parallelism))

    # Crawler
    c = config.crawler
    table.add_row("crawler", "max_pages_per_version", str(c.max_pages_per_version))
    table.add_row("crawler", "rate_limit_rps", str(c.rate_limit_rps))
    table.add_row("crawler", "timeout_seconds", str(c.timeout_seconds))
    table.add_row("crawler", "retry_attempts", str(c.retry_attempts))
    table.add_row("crawler", "retry_backoff", c.retry_backoff)
    table.add_row("crawler", "respect_robots_txt", str(c.respect_robots_txt))
    table.add_row("crawler", "enable_js_rendering", str(c.enable_js_rendering))
    table.add_row("crawler", "cache_ttl_hours", str(c.cache_ttl_hours))

    # Chunker
    ch = config.chunker
    table.add_row("chunker", "target_chunk_size", str(ch.target_chunk_size))
    table.add_row("chunker", "max_chunk_size", str(ch.max_chunk_size))
    table.add_row("chunker", "overlap_tokens", str(ch.overlap_tokens))
    table.add_row("chunker", "strategy", ch.strategy)

    # Embeddings
    e = config.embeddings
    table.add_row("embeddings", "provider", e.provider)
    table.add_row("embeddings", "model", e.model)
    table.add_row("embeddings", "batch_size", str(e.batch_size))
    table.add_row("embeddings", "cache_embeddings", str(e.cache_embeddings))

    # Storage
    s = config.storage
    table.add_row("storage", "backend", s.backend)
    table.add_row("storage", "path", str(s.path))

    console.print(table)


__all__ = ["PipelineVersionResult", "app", "main"]

"""Entry point for running DocForge as a module: python -m docforge."""

from docforge.cli import app


def main() -> None:
    """Run the DocForge CLI."""
    app()


if __name__ == "__main__":
    main()

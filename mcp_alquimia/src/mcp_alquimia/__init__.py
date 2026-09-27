"""Alquimia subprocess runtime; MCP transport will be added in the next stage."""


def main() -> None:
    """Run the step-one local demonstration, without starting an MCP server."""
    from .run_pflotran_calcite import main as run_demo

    run_demo()

if __name__ == "__main__":
    main()

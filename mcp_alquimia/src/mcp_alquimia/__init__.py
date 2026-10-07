"""Alquimia MCP server and persistent native subprocess runtime."""


def main() -> None:
    """Run the local Alquimia MCP server over stdio."""
    from .mcp_server import main as run_server

    run_server()

if __name__ == "__main__":
    main()

import os

from src.server import main as server_main


def main():
    """Entry point for the chromadb-mcp package"""
    # Load environment variables from .env file

    # Print loaded environment variables for debugging
    print(
        f"Environment loaded: CHROMA_PERSIST_DIRECTORY={os.getenv('CHROMA_PERSIST_DIRECTORY', 'Not set')}"
    )

    server_main()


if __name__ == "__main__":
    main()

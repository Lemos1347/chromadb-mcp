from mcp.server.fastmcp import FastMCP

from src.tools import Tools

from .dependency_manager import DependencyContainer

mcp = FastMCP("chromadb")

tools = Tools(DependencyContainer.get_vector_store())


@mcp.tool()
async def get_similar_chunks(query: str, k: int = 2) -> str:
    """
    Retrieve chunks similar to the input query using the configured vector store.

    Args:
        query: The input text to find similar chunks for
        k: The number of chunks to retrieve

    Returns:
        A JSON containing the chunks and their metadata, or "No results were found" if no chunks found
    """
    return tools.retrieve_similar_chunks(query, k)


def main():
    print("MCP server chromadb started!")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

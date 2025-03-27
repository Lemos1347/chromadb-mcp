# ChromaDB-MCP 🚀

A powerful document similarity search service that uses ChromaDB vector database with Model Context Protocol (MCP) integration. This project enables semantic search capabilities through a lightweight MCP server.

## 📝 Description

ChromaDB-MCP processes documents, splits them into semantic chunks, embeds them using Google's text embedding models, and stores them in a ChromaDB vector database. It then exposes an MCP tool that allows you to query for similar chunks to any input text.

## 🏗️ Repository Structure

```
.
├── chroma_db/           # Persistent storage for vector database
├── files/               # Directory for input documents
├── scripts/             # Utility scripts
│   └── files_ingestion/ # Document processing and ingestion scripts
├── src/                 # Source code
│   ├── configs/         # Configuration settings
│   ├── infra/           # Infrastructure implementations
│   ├── ports/           # Interface definitions
│   ├── tools/           # MCP tool implementations
│   ├── dependency_manager.py # Dependency injection container
│   └── server.py        # MCP server implementation
├── .env                 # Environment variables
├── Justfile             # Command runner for common operations
├── main.py              # Application entry point
└── pyproject.toml       # Project dependencies and metadata
```

## ⚙️ Environment Variables

The following environment variables must be set (in a `.env` file or directly in your environment):

- `INPUT_DIR` 📁 - Directory containing documents to process (e.g., "/path/to/files")
- `CHROMA_PERSIST_DIRECTORY` 💾 - Directory to store the ChromaDB database (e.g., "/path/to/chroma_db")
- `GOOGLE_API_KEY` 🔑 - Your Google API key for text embeddings

Example `.env` file:
```
INPUT_DIR="/path/to/files"
CHROMA_PERSIST_DIRECTORY="/path/to/chroma_db"
GOOGLE_API_KEY="your-google-api-key"
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) command runner (optional)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chromadb-mcp.git
   cd chromadb-mcp
   ```

2. Create a `.env` file with required environment variables.

3. Set up the environment and ingest documents:
   ```bash
   just setup
   ```

### Running the Server

Run the MCP server:
```bash
just run
```

For development with MCP Inspector:
```bash
just dev
```

## 🎬 Demo

### Using with Claude Desktop

You can seamlessly integrate ChromaDB-MCP with Claude desktop for enhanced document retrieval capabilities. Follow these steps:

1. We’ll need to configure Claude for Desktop for whichever MCP servers you want to use. To do this, open your Claude for Desktop App configuration at `~/Library/Application Support/Claude/claude_desktop_config.json` in a text editor.

2. Add a new tool with the following configuration:
   ```json
   {
      "mcpServers": {
         "chromadb": {
            "command": "/path/to/bin/uv",
            "args": [
            "--directory",
            "/path/to/chromadb-mcp",
            "run",
            "main.py"
            ],
            "env": {
               "INPUT_DIR": "/path/to/files",
               "CHROMA_PERSIST_DIRECTORY": "/path/to/chroma_db",
               "GOOGLE_API_KEY": "YOUR-REAL-API-KEY-HERE"
            }
         }
      }
   }
   ```
   _Obs: You can either set the environment variables in the `.env` file or directly in the MCP server configuration._

3. Now you can ask Claude questions about your documents! For example:
   - "Find information about project architecture in my documents"
   - "What are the key components of the system design?"
   - "Summarize what the documents say about implementation details"

If you have any issues, please refer to official documentation [here](https://modelcontextprotocol.io/quickstart/server).

### Demo Video

https://github.com/user-attachments/assets/c954a6e0-cded-4f95-b63d-a50ba930bada

## 🔍 Features

- 📄 Document ingestion and chunking
- 🧠 Semantic embeddings using Google's text-embedding-004 model
- 🔎 Vector similarity search through ChromaDB
- 🔌 MCP server interface for easy integration with other applications

## 🛠️ API

The MCP server exposes the following tool:

- `get_similar_chunks(query: str, k: int = 2)` - Retrieves k chunks similar to the input query

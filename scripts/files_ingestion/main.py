import sys
import os
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from files_ingestion import DocumentProcessor
from dotenv import load_dotenv

console = Console()
console.print("[bold]MCP server setup starting...[/bold]")

def setup_document_processor(
    google_api_key: Optional[str] = None,
    embedding_model: str = "models/text-embedding-004",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_directory: str = "chroma_db",
    artifacts_path: Optional[str] = None
) -> DocumentProcessor:
    """
    Set up the document processor.
    
    Args:
        google_api_key: Google API key for embeddings
        embedding_model: The embedding model to use
        chunk_size: Approximate size of semantic chunks
        chunk_overlap: Minimum chunk size / overlap
        persist_directory: Directory to persist the vector database
        artifacts_path: Optional path to Docling model artifacts for offline use
        
    Returns:
        Initialized DocumentProcessor
    """
    # First try to get API key from environment (loaded from .env file)
    if not google_api_key:
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            console.print(
                "[bold red]Error:[/bold red] Google API key not provided. "
                "Please create a .env file in the project root with GOOGLE_API_KEY=your-key or provide it with --google-api-key"
            )
            sys.exit(1)
    
    with console.status("[bold green]Initializing document processor..."):
        processor = DocumentProcessor(
            google_api_key=google_api_key,
            embedding_model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist_directory=persist_directory,
            artifacts_path=artifacts_path
        )
    return processor


def ingest():
    """
    Ingest documents from a directory into the vector store.
    """

    # Directory containing documents to process
    input_dir = os.environ.get("INPUT_DIR", None)
    if not input_dir:
        console.print("[bold red]Error:[/bold red] INPUT_DIR environment variable not set.")
        sys.exit(1)

    google_api_key = os.environ.get("GOOGLE_API_KEY", None)
    if not google_api_key:
        console.print("[bold red]Error:[/bold red] GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    persist_directory = os.environ.get("CHROMA_PERSIST_DIRECTORY", "chroma_db")
    if not persist_directory:
        console.print("[bold red]Error:[/bold red] CHROMA_PERSIST_DIRECTORY environment variable not set.")
        sys.exit(1)

    console.print("[bold green]Document Ingestion[/bold green]")
    console.print(f"Processing documents from: [cyan]{input_dir}[/cyan]")
    
    # Set up document processor
    processor = setup_document_processor(
        google_api_key=google_api_key,
        embedding_model="models/text-embedding-004",
        # Target maximum size of semantic chunks
        chunk_size=1000,
        # Minimum chunk size
        chunk_overlap=200,
        persist_directory=persist_directory,
        artifacts_path=None
    )
    
    # Process directory
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Processing documents..."),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        stats = processor.process_directory(input_dir)
        progress.update(task, completed=True)
    
    # Print statistics
    console.print("[bold green]Document processing complete![/bold green]")
    console.print(f"Total documents: {stats['total']}")
    console.print(f"Successfully processed: {stats['successful']}")
    console.print(f"Failed: {stats['failed']}")


if __name__ == "__main__":
   load_dotenv()
   ingest()
"""
Document processing and vectorization module for legal agent.

This module handles the ingestion of various document formats (PDF, DOCX, etc.),
processes them using Docling, and stores the vectorized content in ChromaDB.
"""

import os
from typing import List, Dict, Optional, Union
from pathlib import Path

# Document processing with Docling
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

# Modern LangChain components - updated imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class DocumentProcessor:
    """Handles document processing, vectorization, and storage."""
    
    def __init__(
        self, 
        google_api_key: Optional[str] = None,
        embedding_model_name: str = "models/text-embedding-004",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = "chroma_db",
        artifacts_path: Optional[str] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            google_api_key: Google API key for embeddings
            embedding_model_name: The Google embedding model to use
            chunk_size: Approximate size of text chunks for vectorization
            chunk_overlap: Minimum size of chunks / overlap
            persist_directory: Directory to persist ChromaDB
            artifacts_path: Optional path to Docling model artifacts for offline use
        """
        self.google_api_key = google_api_key
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.artifacts_path = artifacts_path
        
        # Initialize the document converter with optional offline artifacts path
        if self.artifacts_path:
            self.doc_converter = DocumentConverter(artifacts_path=self.artifacts_path)
        else:
            self.doc_converter = DocumentConverter()
        
        # Initialize Google embeddings for vector store
        if not google_api_key:
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("Google API key must be provided or set as GOOGLE_API_KEY environment variable")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=google_api_key,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        # Use RecursiveCharacterTextSplitter for all text chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=True
        )
        
        # Initialize or load vector store
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize or load the vector store."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Loaded existing vector store from {self.persist_directory}")
        else:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Created new vector store at {self.persist_directory}")
    
    def process_file(self, file_path: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """
        Process a single file and add it to the vector store.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata to attach to the document
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return False
                
            # Default metadata if none provided
            if metadata is None:
                metadata = {"source": str(file_path), "filename": file_path.name}
            
            # Try to convert document with Docling
            try:
                # Convert the document using Docling's document converter
                conversion_result = self.doc_converter.convert(str(file_path))
                doc = conversion_result.document
                
                # Extract text as markdown and use LangChain text splitter
                text = doc.export_to_markdown()
                chunks = self.text_splitter.split_text(text)
                
                if chunks:
                    print(f"Successfully extracted and chunked content from {file_path}")
                else:
                    raise ValueError("No chunks created from document")
                
            except Exception as e:
                print(f"Error processing with Docling: {e}. Using fallback method.")
                # Fallback to simpler text extraction and chunking
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                chunks = self.text_splitter.split_text(text)
            
            # If we still have no chunks, return failure
            if not chunks:
                print(f"Error: Could not extract any content from {file_path}")
                return False
                
            # Create metadata for each chunk
            metadatas = [metadata.copy() for _ in range(len(chunks))]
            
            # Add chunk index to metadata for traceability
            for i, meta in enumerate(metadatas):
                meta["chunk_index"] = i
                meta["total_chunks"] = len(chunks)
            
            # Add to vector store
            self.vectorstore.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
            
            # The newer versions of Chroma automatically persist changes
            # No need to call persist() explicitly
            
            print(f"Successfully processed and vectorized {file_path} into {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return False
    
    def add_text(self, text: str, metadata: Dict) -> bool:
        """
        Add arbitrary text directly to the vector store.
        This is useful for web search results or other non-file text sources.
        Uses semantic chunking to split the text into meaningful segments.
        
        Args:
            text: The text to add
            metadata: Metadata to attach to the text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not text or not metadata:
                print("Text or metadata is empty")
                return False
            
            # If the text is in markdown format, try to use Docling for better conversion
            if text.startswith('#') or '```' in text or '[' in text and '](' in text:
                try:
                    # Try to create a Docling document from the markdown text
                    from docling.datamodel.document import InputDocument
                    from io import StringIO
                    
                    input_doc = InputDocument(
                        path_or_stream=StringIO(text),
                        format=InputFormat.MARKDOWN,
                        filename=metadata.get("filename", "web_content.md")
                    )
                    
                    # Convert the document
                    doc = self.doc_converter.convert(input_doc).document
                    processed_text = doc.export_to_markdown()
                    
                    # Use LangChain text splitter for chunking
                    chunks = self.text_splitter.split_text(processed_text)
                    
                    if chunks:
                        print(f"Successfully processed and chunked markdown text")
                    else:
                        raise ValueError("No chunks created from markdown")
                        
                except Exception as e:
                    print(f"Error using Docling for markdown text: {e}. Using direct chunking.")
                    # Fall back to direct chunking
                    chunks = self.text_splitter.split_text(text)
            else:
                # For plain text, use the text splitter directly
                chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                print(f"Warning: Chunking created no chunks for text with metadata: {metadata.get('title', 'Unknown')}")
                # If chunking fails to create chunks, create a single chunk with the entire text
                chunks = [text]
            
            # Create metadata for each chunk
            metadatas = [metadata.copy() for _ in range(len(chunks))]
            
            # Add chunk index to metadata for traceability
            for i, meta in enumerate(metadatas):
                meta["chunk_index"] = i
                meta["total_chunks"] = len(chunks)
            
            # Add to vector store
            self.vectorstore.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
            
            # The newer versions of Chroma automatically persist changes
            # No need to call persist() explicitly
            
            print(f"Successfully added {len(chunks)} chunks to vector store with metadata: {metadata.get('title', str(metadata))}")
            return True
            
        except Exception as e:
            print(f"Error adding text to vector store: {e}")
            return False
    
    def process_directory(self, directory_path: Union[str, Path]) -> Dict[str, int]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            Dict with statistics about processed files
        """
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            print(f"Directory not found: {directory_path}")
            return {"total": 0, "successful": 0, "failed": 0}
        
        stats = {"total": 0, "successful": 0, "failed": 0}
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = Path(root) / file
                
                # Skip hidden files and non-supported formats
                if file.startswith('.') or not any(file.lower().endswith(ext) for ext in ['.txt', '.pdf', '.docx', '.doc', '.html', '.md']):
                    continue
                
                stats["total"] += 1
                success = self.process_file(file_path)
                if success:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
        
        # The newer versions of Chroma automatically persist changes
        # No need to call persist() explicitly
        
        return stats
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks with their metadata
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        return formatted_results
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        Get a retriever for use with LangChain.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            A LangChain retriever
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        ) 
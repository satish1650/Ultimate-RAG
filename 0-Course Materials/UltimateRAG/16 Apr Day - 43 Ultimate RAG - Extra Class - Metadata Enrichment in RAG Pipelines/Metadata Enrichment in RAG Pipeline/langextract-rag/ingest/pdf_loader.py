"""
PDF Ingestion and Chunking Module

Handles loading PDF documents and splitting them into chunks
using RecursiveCharacterTextSplitter.
"""

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# =============================================================================
# PDF Loading
# =============================================================================

def load_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects, one per page
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"   ✓ Loaded {len(documents)} pages")
    
    return documents


def load_pdfs_from_directory(directory: str | Path) -> List[Document]:
    """
    Load all PDF files from a directory.
    
    Args:
        directory: Path to directory containing PDFs
        
    Returns:
        List of Document objects from all PDFs
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {directory}")
        return []
    
    print(f"📁 Found {len(pdf_files)} PDF file(s) in {directory}")
    
    all_documents = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(pdf_file)
            all_documents.extend(docs)
        except Exception as e:
            print(f"   ❌ Error loading {pdf_file}: {e}")
    
    print(f"   ✓ Total pages loaded: {len(all_documents)}")
    return all_documents


# =============================================================================
# Document Chunking
# =============================================================================

def get_text_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: List[str] = None
) -> RecursiveCharacterTextSplitter:
    """
    Get a configured RecursiveCharacterTextSplitter.
    
    Args:
        chunk_size: Size of each chunk (default: Config.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (default: Config.CHUNK_OVERLAP)
        separators: List of separators to use for splitting
        
    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Size of each chunk (default: Config.CHUNK_SIZE)
        chunk_overlap: Overlap between chunks (default: Config.CHUNK_OVERLAP)
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    
    print(f"✂️  Chunking {len(documents)} documents...")
    print(f"   - Chunk size: {chunk_size or Config.CHUNK_SIZE}")
    print(f"   - Chunk overlap: {chunk_overlap or Config.CHUNK_OVERLAP}")
    
    chunks = text_splitter.split_documents(documents)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    return chunks


def chunk_pdf(pdf_path: str | Path, **kwargs) -> List[Document]:
    """
    Convenience function to load and chunk a PDF in one step.
    
    Args:
        pdf_path: Path to the PDF file
        **kwargs: Additional arguments for chunk_documents
        
    Returns:
        List of chunked Document objects
    """
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents, **kwargs)
    return chunks


# =============================================================================
# Utility Functions
# =============================================================================

def get_document_stats(documents: List[Document]) -> dict:
    """
    Get statistics about a list of documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Dictionary with statistics
    """
    if not documents:
        return {
            "count": 0,
            "total_chars": 0,
            "avg_chars": 0,
            "sources": set()
        }
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    sources = set(doc.metadata.get("source", "unknown") for doc in documents)
    
    return {
        "count": len(documents),
        "total_chars": total_chars,
        "avg_chars": total_chars // len(documents),
        "sources": sources
    }


def print_document_sample(documents: List[Document], n: int = 3):
    """
    Print a sample of documents for debugging.
    
    Args:
        documents: List of Document objects
        n: Number of documents to print
    """
    print(f"\n📋 Sample of first {min(n, len(documents))} documents:")
    print("-" * 60)
    
    for i, doc in enumerate(documents[:n]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Content preview: {doc.page_content[:150]}...")


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    # Test the module
    import sys
    
    # Check if PDF path provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            chunks = chunk_pdf(pdf_path)
            stats = get_document_stats(chunks)
            print(f"\n📊 Statistics:")
            print(f"   - Total chunks: {stats['count']}")
            print(f"   - Total characters: {stats['total_chars']}")
            print(f"   - Average chunk size: {stats['avg_chars']} chars")
            print(f"   - Sources: {stats['sources']}")
            print_document_sample(chunks)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        print("\nThis module loads PDFs and chunks them for processing.")

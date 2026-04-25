#!/usr/bin/env python3
"""
LangExtract-Enhanced RAG System - Main Entry Point

This script provides end-to-end pipelines for:
1. Ingesting PDF documents into the vector store
2. Querying the RAG system with metadata-filtered retrieval
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

# Import configuration
from config import Config, validate_config

# Import ingestion modules
from ingest import (
    load_pdf,
    load_pdfs_from_directory,
    chunk_documents,
    enrich_chunks_with_metadata,
    extract_document_level_metadata,
    propagate_metadata_to_chunks,
    save_extraction_results,
    load_extraction_results,
    create_index_if_not_exists,
    upsert_documents,
    upsert_in_batches,
    print_index_stats,
    delete_namespace
)

# Import retrieval modules
from retrieval import (
    RAGQueryEngine,
    create_rag_chain,
    create_self_query_rag_chain,
    create_metadata_aware_chain,
    search_with_filter,
    self_query_search,
    compare_retrieval_methods,
    print_comparison_results,
    print_search_results
)


# =============================================================================
# Ingestion Pipeline
# =============================================================================

def ingest_pdf(
    pdf_path: str,
    use_document_level_extraction: bool = False,
    save_results: Optional[str] = None,
    batch_size: int = 100
) -> List[Document]:
    """
    Full pipeline: PDF → Chunks → LangExtract Metadata → Pinecone
    
    Args:
        pdf_path: Path to the PDF file
        use_document_level_extraction: Use document-level extraction (faster, cheaper)
        save_results: Optional path to save extraction results
        batch_size: Batch size for upsertion
        
    Returns:
        List of enriched Document objects
    """
    print("=" * 70)
    print("📥 INGESTION PIPELINE")
    print("=" * 70)
    
    # Step 1: Load PDF
    print(f"\n📄 Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    
    # Step 2: Chunk documents
    print(f"\n✂️  Chunking {len(documents)} pages...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 3: Extract metadata
    if use_document_level_extraction:
        # Document-level extraction (cost-effective)
        print(f"\n🧠 Extracting document-level metadata...")
        full_text = "\n\n".join(doc.page_content for doc in documents)
        doc_metadata = extract_document_level_metadata(full_text)
        enriched_docs = propagate_metadata_to_chunks(chunks, doc_metadata)
    else:
        # Chunk-level extraction (higher quality)
        print(f"\n🧠 Extracting metadata for each chunk (this may take a while)...")
        enriched_docs = enrich_chunks_with_metadata(chunks)
    
    # Step 4: Create Pinecone index if needed
    print(f"\n📦 Checking Pinecone index...")
    create_index_if_not_exists()
    
    # Step 5: Upsert to Pinecone
    print(f"\n💾 Upserting {len(enriched_docs)} documents to Pinecone...")
    if len(enriched_docs) > batch_size:
        upsert_in_batches(enriched_docs, batch_size=batch_size)
    else:
        upsert_documents(enriched_docs)
    
    # Step 6: Save results if requested
    if save_results:
        save_extraction_results(enriched_docs, save_results)
    
    print("\n" + "=" * 70)
    print(f"✅ INGESTION COMPLETE")
    print(f"   - Total chunks processed: {len(enriched_docs)}")
    print("=" * 70)
    
    return enriched_docs


def ingest_directory(
    directory: str,
    use_document_level_extraction: bool = False,
    save_results: Optional[str] = None,
    batch_size: int = 100
) -> List[Document]:
    """
    Ingest all PDFs from a directory.
    
    Args:
        directory: Path to directory containing PDFs
        use_document_level_extraction: Use document-level extraction
        save_results: Optional path to save extraction results
        batch_size: Batch size for upsertion
        
    Returns:
        List of enriched Document objects
    """
    print("=" * 70)
    print("📥 BATCH INGESTION PIPELINE")
    print("=" * 70)
    
    # Load all PDFs
    documents = load_pdfs_from_directory(directory)
    
    if not documents:
        print("❌ No documents to process")
        return []
    
    # Chunk all documents
    print(f"\n✂️  Chunking {len(documents)} total pages...")
    chunks = chunk_documents(documents)
    
    # Extract metadata
    if use_document_level_extraction:
        print(f"\n🧠 Using document-level extraction...")
        # Group chunks by source
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        enriched_docs = []
        for source, source_chunks in chunks_by_source.items():
            full_text = "\n\n".join(chunk.page_content for chunk in source_chunks)
            doc_metadata = extract_document_level_metadata(full_text)
            source_enriched = propagate_metadata_to_chunks(source_chunks, doc_metadata)
            enriched_docs.extend(source_enriched)
    else:
        print(f"\n🧠 Extracting metadata for each chunk...")
        enriched_docs = enrich_chunks_with_metadata(chunks)
    
    # Create index and upsert
    create_index_if_not_exists()
    
    print(f"\n💾 Upserting {len(enriched_docs)} documents...")
    if len(enriched_docs) > batch_size:
        upsert_in_batches(enriched_docs, batch_size=batch_size)
    else:
        upsert_documents(enriched_docs)
    
    if save_results:
        save_extraction_results(enriched_docs, save_results)
    
    print("\n" + "=" * 70)
    print(f"✅ BATCH INGESTION COMPLETE")
    print(f"   - Total chunks processed: {len(enriched_docs)}")
    print("=" * 70)
    
    return enriched_docs


# =============================================================================
# Query Pipeline
# =============================================================================

def query_rag_system(
    question: str,
    use_self_query: bool = False,
    use_metadata_aware: bool = False,
    verbose: bool = True
) -> str:
    """
    Query the RAG system with optional metadata-filtered retrieval.
    
    Args:
        question: The question to answer
        use_self_query: Use SelfQueryRetriever for automatic filtering
        use_metadata_aware: Use metadata-aware prompt
        verbose: Print detailed output
        
    Returns:
        Generated answer
    """
    if verbose:
        print("=" * 70)
        print("🔍 QUERY PIPELINE")
        print("=" * 70)
        print(f"\n❓ Question: {question}")
    
    # Create appropriate chain
    if use_self_query:
        if verbose:
            print("\n🤖 Using SelfQueryRetriever for automatic metadata filtering...")
        chain = create_self_query_rag_chain(verbose=verbose)
    elif use_metadata_aware:
        if verbose:
            print("\n🤖 Using metadata-aware RAG chain...")
        chain = create_metadata_aware_chain()
    else:
        if verbose:
            print("\n🤖 Using standard RAG chain...")
        chain = create_rag_chain()
    
    # Query
    if verbose:
        print("\n💭 Generating answer...")
        print("-" * 70)
    
    response = chain.invoke(question)
    
    if verbose:
        print("\n💬 Answer:")
        print("=" * 70)
    
    print(response)
    
    if verbose:
        print("=" * 70)
    
    return response


def interactive_query_loop():
    """
    Run an interactive query loop for testing.
    """
    print("\n" + "=" * 70)
    print("🤖 INTERACTIVE RAG QUERY MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  /sq          - Toggle SelfQueryRetriever")
    print("  /meta        - Toggle metadata-aware mode")
    print("  /quit        - Exit")
    print("-" * 70)
    
    use_self_query = False
    use_metadata_aware = False
    
    while True:
        try:
            question = input("\n❓ Query: ").strip()
            
            if not question:
                continue
            
            if question.lower() == '/quit':
                print("👋 Goodbye!")
                break
            
            if question.lower() == '/sq':
                use_self_query = not use_self_query
                print(f"   SelfQueryRetriever: {'ON' if use_self_query else 'OFF'}")
                continue
            
            if question.lower() == '/meta':
                use_metadata_aware = not use_metadata_aware
                print(f"   Metadata-aware: {'ON' if use_metadata_aware else 'OFF'}")
                continue
            
            query_rag_system(
                question,
                use_self_query=use_self_query,
                use_metadata_aware=use_metadata_aware
            )
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# Utility Commands
# =============================================================================

def show_stats():
    """Display Pinecone index statistics."""
    print("=" * 70)
    print("📊 INDEX STATISTICS")
    print("=" * 70)
    print_index_stats()


def clear_namespace(namespace: Optional[str] = None):
    """Clear all vectors from a namespace."""
    ns = namespace or Config.NAMESPACE
    print("=" * 70)
    print(f"🗑️  CLEAR NAMESPACE: {ns}")
    print("=" * 70)
    delete_namespace(namespace=ns)


def compare_methods(query: str):
    """Compare different retrieval methods."""
    print("=" * 70)
    print("🔬 RETRIEVAL METHOD COMPARISON")
    print("=" * 70)
    
    results = compare_retrieval_methods(query)
    print_comparison_results(results)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LangExtract-Enhanced RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a PDF
  python main.py ingest --pdf path/to/document.pdf
  
  # Ingest all PDFs in a directory
  python main.py ingest --dir path/to/pdfs/
  
  # Query the system
  python main.py query "What is the authentication process?"
  
  # Query with SelfQueryRetriever
  python main.py query "What are the API limits in v2.1?" --self-query
  
  # Interactive mode
  python main.py interactive
  
  # Show index stats
  python main.py stats
  
  # Compare retrieval methods
  python main.py compare "What are the API limits?"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest PDF documents')
    ingest_parser.add_argument('--pdf', type=str, help='Path to PDF file')
    ingest_parser.add_argument('--dir', type=str, help='Path to directory with PDFs')
    ingest_parser.add_argument(
        '--doc-level', 
        action='store_true',
        help='Use document-level metadata extraction (faster, cheaper)'
    )
    ingest_parser.add_argument(
        '--save', 
        type=str,
        help='Save extraction results to file'
    )
    ingest_parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for upsertion (default: 100)'
    )
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', type=str, help='Question to ask')
    query_parser.add_argument(
        '--self-query',
        action='store_true',
        help='Use SelfQueryRetriever for automatic metadata filtering'
    )
    query_parser.add_argument(
        '--metadata-aware',
        action='store_true',
        help='Use metadata-aware prompt template'
    )
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive query mode')
    
    # Stats command
    subparsers.add_parser('stats', help='Show index statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear namespace')
    clear_parser.add_argument(
        '--namespace',
        type=str,
        help='Namespace to clear (default: from config)'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare retrieval methods')
    compare_parser.add_argument('query', type=str, help='Query to test')
    
    args = parser.parse_args()
    
    # Validate configuration
    missing = validate_config()
    if missing:
        print("❌ Missing required configuration:")
        for item in missing:
            print(f"   - {item}")
        print("\nPlease set these in your .env file")
        sys.exit(1)
    
    # Execute command
    if args.command == 'ingest':
        if args.pdf:
            ingest_pdf(
                args.pdf,
                use_document_level_extraction=args.doc_level,
                save_results=args.save,
                batch_size=args.batch_size
            )
        elif args.dir:
            ingest_directory(
                args.dir,
                use_document_level_extraction=args.doc_level,
                save_results=args.save,
                batch_size=args.batch_size
            )
        else:
            ingest_parser.print_help()
            sys.exit(1)
    
    elif args.command == 'query':
        query_rag_system(
            args.question,
            use_self_query=args.self_query,
            use_metadata_aware=args.metadata_aware
        )
    
    elif args.command == 'interactive':
        interactive_query_loop()
    
    elif args.command == 'stats':
        show_stats()
    
    elif args.command == 'clear':
        clear_namespace(args.namespace)
    
    elif args.command == 'compare':
        compare_methods(args.query)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

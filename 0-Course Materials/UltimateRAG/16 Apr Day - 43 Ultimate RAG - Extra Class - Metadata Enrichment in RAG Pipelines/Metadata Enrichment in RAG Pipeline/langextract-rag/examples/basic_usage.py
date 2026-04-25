#!/usr/bin/env python3
"""
Basic Usage Example for LangExtract-Enhanced RAG System

This example demonstrates:
1. Loading and chunking a PDF
2. Extracting metadata with LangExtract
3. Storing in Pinecone
4. Querying with metadata filtering
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from config import validate_config

# Import ingestion modules
from ingest.pdf_loader import chunk_documents
from ingest.metadata_extractor import enrich_chunks_with_metadata
from ingest.pinecone_store import (
    create_index_if_not_exists,
    upsert_documents,
    print_index_stats
)

# Import retrieval modules
from retrieval.retriever import search_with_filter, print_search_results
from retrieval.rag_chain import query_rag_system


def main():
    """Run the basic usage example."""
    
    # Validate configuration
    missing = validate_config()
    if missing:
        print("❌ Missing required configuration:")
        for item in missing:
            print(f"   - {item}")
        print("\nPlease set these in your .env file")
        sys.exit(1)
    
    print("=" * 70)
    print("📚 LANGEXTRACT-ENHANCED RAG - BASIC USAGE EXAMPLE")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load and Process PDF
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Load and Process PDF")
    print("-" * 70)
    
    # For this example, we'll create a sample text document
    # In practice, replace with: pdf_path = "path/to/your/document.pdf"
    
    sample_text = """
    API Reference Guide v2.1 - Released March 2025
    
    Authentication
    --------------
    The Payment API uses OAuth 2.0 for authentication. Supported flows include:
    - Client Credentials Flow (for server-to-server)
    - Authorization Code Flow (for user-facing apps)
    
    Contact the FinTech Team for API credentials.
    
    Rate Limits
    -----------
    Version 2.1 introduces tiered rate limits:
    - Standard tier: 100 requests/minute
    - Premium tier: 1000 requests/minute
    - Enterprise tier: Unlimited
    
    Endpoints
    ---------
    POST /v2.1/payments
    GET /v2.1/payments/{id}
    DELETE /v2.1/payments/{id}
    """
    
    # Create a mock document
    documents = [
        Document(
            page_content=sample_text,
            metadata={"source": "api_reference.pdf", "page": 1}
        )
    ]
    
    print(f"✓ Loaded {len(documents)} document(s)")
    
    # Chunk the document
    chunks = chunk_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    # -------------------------------------------------------------------------
    # STEP 2: Extract Metadata with LangExtract
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Extract Metadata with LangExtract")
    print("-" * 70)
    
    print("🧠 Extracting metadata (this makes LLM API calls)...")
    enriched_docs = enrich_chunks_with_metadata(chunks)
    
    # Show extracted metadata
    print("\n📋 Extracted Metadata:")
    for i, doc in enumerate(enriched_docs):
        print(f"\n  Chunk {i + 1}:")
        print(f"    Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"    Category: {doc.metadata.get('category', 'N/A')}")
        print(f"    Version: {doc.metadata.get('version', 'N/A')}")
        print(f"    Entities: {doc.metadata.get('entities', 'N/A')}")
        print(f"    Summary: {doc.metadata.get('chunk_summary', 'N/A')[:80]}...")
    
    # -------------------------------------------------------------------------
    # STEP 3: Store in Pinecone
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: Store in Pinecone")
    print("-" * 70)
    
    # Create index if it doesn't exist
    create_index_if_not_exists()
    
    # Upsert documents
    print("💾 Upserting documents...")
    upsert_documents(enriched_docs)
    
    # Show stats
    print_index_stats()
    
    # -------------------------------------------------------------------------
    # STEP 4: Query with Metadata Filtering
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: Query with Metadata Filtering")
    print("-" * 70)
    
    # Example 1: Basic semantic search
    print("\n📌 Example 1: Basic Semantic Search")
    query1 = "What are the rate limits?"
    print(f"   Query: '{query1}'")
    
    results1 = search_with_filter(query1, filter_dict={})
    print_search_results(results1)
    
    # Example 2: Metadata-filtered search
    print("\n📌 Example 2: Metadata-Filtered Search")
    query2 = "API authentication methods"
    filter_dict = {"category": "technical"}
    print(f"   Query: '{query2}'")
    print(f"   Filter: {filter_dict}")
    
    results2 = search_with_filter(query2, filter_dict=filter_dict)
    print_search_results(results2)
    
    # Example 3: Full RAG query
    print("\n📌 Example 3: Full RAG Query")
    query3 = "What authentication flow should I use for a server application?"
    print(f"   Query: '{query3}'")
    
    answer = query_rag_system(query3, verbose=False)
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Copy .env.example to .env and add your API keys")
    print("  2. Run: python main.py ingest --pdf path/to/your.pdf")
    print("  3. Run: python main.py query 'Your question here'")
    print("  4. Run: python main.py interactive  # For interactive mode")


if __name__ == "__main__":
    main()

"""
Ingestion Module

Handles PDF loading, text chunking, metadata extraction, and vector store upsertion.
"""

from ingest.pdf_loader import (
    load_pdf,
    load_pdfs_from_directory,
    chunk_documents,
    chunk_pdf,
    get_document_stats,
    print_document_sample
)

from ingest.metadata_extractor import (
    extract_metadata_for_chunk,
    enrich_chunks_with_metadata,
    extract_document_level_metadata,
    propagate_metadata_to_chunks,
    save_extraction_results,
    load_extraction_results,
    print_metadata_sample,
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_EXAMPLES
)

from ingest.pinecone_store import (
    create_index_if_not_exists,
    get_index,
    delete_index,
    get_index_stats,
    print_index_stats,
    upsert_documents,
    add_to_existing_vectorstore,
    get_vectorstore,
    delete_namespace,
    build_metadata_filter,
    upsert_in_batches
)

__all__ = [
    # PDF Loading
    "load_pdf",
    "load_pdfs_from_directory",
    "chunk_documents",
    "chunk_pdf",
    "get_document_stats",
    "print_document_sample",
    
    # Metadata Extraction
    "extract_metadata_for_chunk",
    "enrich_chunks_with_metadata",
    "extract_document_level_metadata",
    "propagate_metadata_to_chunks",
    "save_extraction_results",
    "load_extraction_results",
    "print_metadata_sample",
    "DEFAULT_EXTRACTION_PROMPT",
    "DEFAULT_EXAMPLES",
    
    # Pinecone Store
    "create_index_if_not_exists",
    "get_index",
    "delete_index",
    "get_index_stats",
    "print_index_stats",
    "upsert_documents",
    "add_to_existing_vectorstore",
    "get_vectorstore",
    "delete_namespace",
    "build_metadata_filter",
    "upsert_in_batches"
]

"""
Retrieval Module

Provides retrieval strategies and RAG chain functionality.
"""

from retrieval.retriever import (
    get_basic_retriever,
    get_filtered_retriever,
    get_scored_retriever,
    get_self_query_retriever,
    search_with_filter,
    search_with_scores,
    self_query_search,
    compare_retrieval_methods,
    print_comparison_results,
    format_retrieved_context,
    print_search_results,
    DEFAULT_METADATA_FIELD_INFO,
    DOCUMENT_CONTENT_DESCRIPTION
)

from retrieval.rag_chain import (
    create_rag_chain,
    create_self_query_rag_chain,
    create_citation_rag_chain,
    create_metadata_aware_chain,
    create_scored_retrieval_chain,
    create_legacy_retrieval_chain,
    RAGQueryEngine,
    query_rag,
    query_with_self_query_retrieval,
    format_docs,
    format_docs_with_summary,
    format_docs_minimal,
    DEFAULT_RAG_PROMPT,
    CITATION_RAG_PROMPT,
    METADATA_AWARE_PROMPT,
    SUMMARY_PROMPT
)

__all__ = [
    # Retriever
    "get_basic_retriever",
    "get_filtered_retriever",
    "get_scored_retriever",
    "get_self_query_retriever",
    "search_with_filter",
    "search_with_scores",
    "self_query_search",
    "compare_retrieval_methods",
    "print_comparison_results",
    "format_retrieved_context",
    "print_search_results",
    "DEFAULT_METADATA_FIELD_INFO",
    "DOCUMENT_CONTENT_DESCRIPTION",
    
    # RAG Chain
    "create_rag_chain",
    "create_self_query_rag_chain",
    "create_citation_rag_chain",
    "create_metadata_aware_chain",
    "create_scored_retrieval_chain",
    "create_legacy_retrieval_chain",
    "RAGQueryEngine",
    "query_rag",
    "query_with_self_query_retrieval",
    "format_docs",
    "format_docs_with_summary",
    "format_docs_minimal",
    "DEFAULT_RAG_PROMPT",
    "CITATION_RAG_PROMPT",
    "METADATA_AWARE_PROMPT",
    "SUMMARY_PROMPT"
]

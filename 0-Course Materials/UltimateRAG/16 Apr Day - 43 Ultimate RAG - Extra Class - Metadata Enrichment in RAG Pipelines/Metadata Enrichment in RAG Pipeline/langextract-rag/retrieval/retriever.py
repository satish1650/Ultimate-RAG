"""
Retrieval Module with Metadata Filtering

Provides multiple retrieval strategies:
1. Basic metadata-filtered search
2. Self-query retriever (LLM generates filters from query)
3. Standard retriever with similarity thresholds
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config, get_openai_embeddings, get_openai_llm
from ingest.pinecone_store import get_vectorstore


# =============================================================================
# Metadata Field Definitions (for SelfQueryRetriever)
# =============================================================================

DEFAULT_METADATA_FIELD_INFO = [
    AttributeInfo(
        name="topic",
        description="The primary topic or subject of the document chunk",
        type="string",
    ),
    AttributeInfo(
        name="category",
        description=(
            "The document category. One of: "
            "'technical', 'legal', 'financial', 'medical', 'general'"
        ),
        type="string",
    ),
    AttributeInfo(
        name="subcategory",
        description="More specific subcategory within the main category",
        type="string",
    ),
    AttributeInfo(
        name="version",
        description="The version number mentioned in the document (e.g., 'v2.1', '3.0')",
        type="string",
    ),
    AttributeInfo(
        name="release_date",
        description="Release date mentioned in the document",
        type="string",
    ),
    AttributeInfo(
        name="entities",
        description="Key named entities (people, organizations, products) mentioned in the chunk",
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="Source file name of the document",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="Page number in the source document",
        type="integer",
    ),
]

DOCUMENT_CONTENT_DESCRIPTION = (
    "Content from PDF documents enriched with "
    "LangExtract structured metadata including topic, category, version, "
    "and key entities."
)


# =============================================================================
# Retriever Factory Functions
# =============================================================================

def get_basic_retriever(
    vectorstore: Optional[PineconeVectorStore] = None,
    search_kwargs: Optional[Dict[str, Any]] = None
) -> BaseRetriever:
    """
    Get a basic retriever from a vector store.
    
    Args:
        vectorstore: PineconeVectorStore instance
        search_kwargs: Search configuration (k, filter, etc.)
        
    Returns:
        BaseRetriever instance
    """
    vectorstore = vectorstore or get_vectorstore()
    
    default_kwargs = {
        "k": Config.DEFAULT_K
    }
    
    if search_kwargs:
        default_kwargs.update(search_kwargs)
    
    return vectorstore.as_retriever(search_kwargs=default_kwargs)


def get_filtered_retriever(
    filter_dict: Dict[str, Any],
    vectorstore: Optional[PineconeVectorStore] = None,
    k: int = None
) -> BaseRetriever:
    """
    Get a retriever with fixed metadata filters.
    
    Args:
        filter_dict: Metadata filter dictionary
        vectorstore: PineconeVectorStore instance
        k: Number of results to return
        
    Returns:
        BaseRetriever instance with filters applied
    """
    vectorstore = vectorstore or get_vectorstore()
    k = k or Config.DEFAULT_K
    
    search_kwargs = {
        "k": k,
        "filter": filter_dict
    }
    
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def get_scored_retriever(
    score_threshold: float = None,
    vectorstore: Optional[PineconeVectorStore] = None,
    k: int = None
) -> BaseRetriever:
    """
    Get a retriever with similarity score threshold.
    
    Args:
        score_threshold: Minimum similarity score (0-1)
        vectorstore: PineconeVectorStore instance
        k: Number of results to return
        
    Returns:
        BaseRetriever instance with score threshold
    """
    vectorstore = vectorstore or get_vectorstore()
    score_threshold = score_threshold or Config.SCORE_THRESHOLD
    k = k or Config.DEFAULT_K
    
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold
        }
    )


def get_self_query_retriever(
    llm: Optional[ChatOpenAI] = None,
    vectorstore: Optional[PineconeVectorStore] = None,
    metadata_field_info: Optional[List[AttributeInfo]] = None,
    document_contents: Optional[str] = None,
    verbose: bool = True,
    k: int = None
) -> SelfQueryRetriever:
    """
    Get a SelfQueryRetriever that uses LLM to generate metadata filters.
    
    This is the most powerful retrieval method - it automatically extracts
    filter conditions from natural language queries.
    
    Args:
        llm: Language model for query parsing
        vectorstore: PineconeVectorStore instance
        metadata_field_info: Metadata field definitions
        document_contents: Description of document contents
        verbose: Whether to print intermediate steps
        k: Number of results to return
        
    Returns:
        SelfQueryRetriever instance
    """
    llm = llm or get_openai_llm(temperature=0)
    vectorstore = vectorstore or get_vectorstore()
    metadata_field_info = metadata_field_info or DEFAULT_METADATA_FIELD_INFO
    document_contents = document_contents or DOCUMENT_CONTENT_DESCRIPTION
    k = k or Config.DEFAULT_K
    
    print(f"🔍 Initializing SelfQueryRetriever...")
    print(f"   - LLM: {Config.LLM_MODEL}")
    print(f"   - Top K: {k}")
    
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_contents,
        metadata_field_info=metadata_field_info,
        verbose=verbose,
        search_kwargs={"k": k}
    )
    
    return retriever


# =============================================================================
# Search Functions
# =============================================================================

def search_with_filter(
    query: str,
    filter_dict: Dict[str, Any],
    vectorstore: Optional[PineconeVectorStore] = None,
    k: int = None
) -> List[Document]:
    """
    Perform a similarity search with metadata filtering.
    
    Args:
        query: Search query
        filter_dict: Metadata filter dictionary
        vectorstore: PineconeVectorStore instance
        k: Number of results to return
        
    Returns:
        List of matching Document objects
    """
    vectorstore = vectorstore or get_vectorstore()
    k = k or Config.DEFAULT_K
    
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter=filter_dict
    )
    
    return results


def search_with_scores(
    query: str,
    vectorstore: Optional[PineconeVectorStore] = None,
    k: int = None,
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[tuple[Document, float]]:
    """
    Perform a similarity search with scores.
    
    Args:
        query: Search query
        vectorstore: PineconeVectorStore instance
        k: Number of results to return
        filter_dict: Optional metadata filter
        
    Returns:
        List of (Document, score) tuples
    """
    vectorstore = vectorstore or get_vectorstore()
    k = k or Config.DEFAULT_K
    
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        filter=filter_dict
    )
    
    return results


def self_query_search(
    query: str,
    retriever: Optional[SelfQueryRetriever] = None
) -> List[Document]:
    """
    Perform a self-query search where the LLM extracts filters from the query.
    
    Args:
        query: Natural language query
        retriever: SelfQueryRetriever instance (creates new if None)
        
    Returns:
        List of matching Document objects
    """
    retriever = retriever or get_self_query_retriever()
    
    results = retriever.invoke(query)
    return results


# =============================================================================
# Comparison and Evaluation
# =============================================================================

def compare_retrieval_methods(
    query: str,
    vectorstore: Optional[PineconeVectorStore] = None,
    filter_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Document]]:
    """
    Compare different retrieval methods for the same query.
    
    Args:
        query: Search query
        vectorstore: PineconeVectorStore instance
        filter_dict: Metadata filter for filtered search
        
    Returns:
        Dictionary mapping method name to results
    """
    vectorstore = vectorstore or get_vectorstore()
    filter_dict = filter_dict or {"category": "technical"}
    
    print(f"\n🔬 Comparing retrieval methods for query: '{query}'")
    print("-" * 60)
    
    # Method 1: Basic semantic search (no filters)
    basic_results = vectorstore.similarity_search(query, k=5)
    
    # Method 2: Metadata-filtered search
    filtered_results = search_with_filter(query, filter_dict, vectorstore)
    
    # Method 3: Self-query retriever
    try:
        self_query_retriever = get_self_query_retriever(vectorstore=vectorstore, verbose=False)
        self_query_results = self_query_search(query, self_query_retriever)
    except Exception as e:
        print(f"   ⚠️  SelfQueryRetriever failed: {e}")
        self_query_results = []
    
    return {
        "basic": basic_results,
        "filtered": filtered_results,
        "self_query": self_query_results
    }


def print_comparison_results(results: Dict[str, List[Document]]):
    """
    Print comparison results in a readable format.
    
    Args:
        results: Dictionary from compare_retrieval_methods
    """
    for method, docs in results.items():
        print(f"\n📌 {method.upper()} RETRIEVAL ({len(docs)} results):")
        print("-" * 40)
        for i, doc in enumerate(docs[:3], 1):  # Show top 3
            print(f"  {i}. Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"     Category: {doc.metadata.get('category', 'N/A')}")
            print(f"     Version: {doc.metadata.get('version', 'N/A')}")
            print(f"     Preview: {doc.page_content[:80]}...")


# =============================================================================
# Utility Functions
# =============================================================================

def format_retrieved_context(
    documents: List[Document],
    include_metadata: bool = True
) -> str:
    """
    Format retrieved documents into a context string for LLM.
    
    Args:
        documents: List of Document objects
        include_metadata: Whether to include metadata in context
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."
    
    formatted_docs = []
    
    for i, doc in enumerate(documents, 1):
        if include_metadata:
            source = doc.metadata.get('source', 'N/A')
            topic = doc.metadata.get('topic', 'N/A')
            category = doc.metadata.get('category', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            
            header = f"[Source: {source} | Page: {page} | Topic: {topic} | Category: {category}]"
        else:
            header = f"[Document {i}]"
        
        formatted_docs.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted_docs)


def print_search_results(
    documents: List[Document],
    scores: Optional[List[float]] = None
):
    """
    Print search results in a readable format.
    
    Args:
        documents: List of Document objects
        scores: Optional list of similarity scores
    """
    print(f"\n🔍 Found {len(documents)} results:")
    print("-" * 60)
    
    for i, doc in enumerate(documents):
        score_str = f" (Score: {scores[i]:.3f})" if scores else ""
        print(f"\n  Result {i + 1}{score_str}:")
        print(f"    Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"    Category: {doc.metadata.get('category', 'N/A')}")
        print(f"    Entities: {doc.metadata.get('entities', 'N/A')}")
        print(f"    Version: {doc.metadata.get('version', 'N/A')}")
        print(f"    Source: {doc.metadata.get('source', 'N/A')}")
        print(f"    Preview: {doc.page_content[:100]}...")


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Validate configuration
    missing = []
    if not Config.PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not Config.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    
    if missing:
        print(f"❌ Missing configuration: {', '.join(missing)}")
        sys.exit(1)
    
    print("Testing retrieval module...")
    print("-" * 60)
    
    # Test basic retriever creation
    print("\n1. Testing basic retriever...")
    try:
        basic_retriever = get_basic_retriever()
        print("   ✓ Basic retriever created")
    except Exception as e:
        print(f"   ⚠️  Basic retriever failed (expected if index empty): {e}")
    
    # Test self-query retriever creation
    print("\n2. Testing SelfQueryRetriever...")
    try:
        self_query_retriever = get_self_query_retriever()
        print("   ✓ SelfQueryRetriever created")
    except Exception as e:
        print(f"   ⚠️  SelfQueryRetriever failed: {e}")
    
    print("\n✅ Retrieval module test complete")

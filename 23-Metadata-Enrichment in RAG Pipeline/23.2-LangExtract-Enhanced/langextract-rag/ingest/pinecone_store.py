"""
Pinecone Vector Store Module

Handles Pinecone index creation, document upsertion, and vector store management
for the LangExtract-Enhanced RAG System.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config, get_openai_embeddings, get_pinecone_client


# =============================================================================
# Index Management
# =============================================================================

def create_index_if_not_exists(
    pc: Optional[Pinecone] = None,
    index_name: Optional[str] = None,
    dimension: Optional[int] = None,
    metric: Optional[str] = None,
    cloud: Optional[str] = None,
    region: Optional[str] = None
) -> bool:
    """
    Create a Pinecone serverless index if it doesn't already exist.
    
    Args:
        pc: Pinecone client (creates new if None)
        index_name: Name of the index (default: Config.PINECONE_INDEX_NAME)
        dimension: Vector dimension (default: Config.INDEX_DIMENSION)
        metric: Distance metric (default: Config.INDEX_METRIC)
        cloud: Cloud provider (default: Config.INDEX_CLOUD)
        region: Cloud region (default: Config.INDEX_REGION)
        
    Returns:
        True if index was created, False if it already existed
    """
    pc = pc or get_pinecone_client()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    dimension = dimension or Config.INDEX_DIMENSION
    metric = metric or Config.INDEX_METRIC
    cloud = cloud or Config.INDEX_CLOUD
    region = region or Config.INDEX_REGION
    
    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"📦 Index '{index_name}' already exists")
        return False
    
    print(f"🔨 Creating new index: {index_name}")
    print(f"   - Dimension: {dimension}")
    print(f"   - Metric: {metric}")
    print(f"   - Cloud: {cloud}")
    print(f"   - Region: {region}")
    
    # Create serverless index
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=Metric(metric.upper()),
        spec=ServerlessSpec(
            cloud=CloudProvider(cloud.upper()),
            region=region
        )
    )
    
    # Wait for index to be ready
    print("   ⏳ Waiting for index to be ready...")
    while not pc.describe_index(index_name).status.ready:
        time.sleep(1)
    
    print(f"   ✓ Index '{index_name}' is ready")
    return True


def get_index(pc: Optional[Pinecone] = None, index_name: Optional[str] = None):
    """
    Get a Pinecone index instance.
    
    Args:
        pc: Pinecone client
        index_name: Name of the index
        
    Returns:
        Pinecone Index instance
    """
    pc = pc or get_pinecone_client()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    
    return pc.Index(index_name)


def delete_index(
    pc: Optional[Pinecone] = None,
    index_name: Optional[str] = None,
    confirm: bool = True
):
    """
    Delete a Pinecone index.
    
    Args:
        pc: Pinecone client
        index_name: Name of the index to delete
        confirm: Whether to ask for confirmation
    """
    pc = pc or get_pinecone_client()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    
    if confirm:
        response = input(f"⚠️  Are you sure you want to delete index '{index_name}'? (yes/no): ")
        if response.lower() != "yes":
            print("   Deletion cancelled")
            return
    
    print(f"🗑️  Deleting index: {index_name}")
    pc.delete_index(index_name)
    print(f"   ✓ Index deleted")


def get_index_stats(pc: Optional[Pinecone] = None, index_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics about a Pinecone index.
    
    Args:
        pc: Pinecone client
        index_name: Name of the index
        
    Returns:
        Dictionary with index statistics
    """
    pc = pc or get_pinecone_client()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    return {
        "total_vector_count": stats.total_vector_count,
        "dimension": stats.dimension,
        "index_fullness": stats.index_fullness,
        "namespaces": stats.namespaces
    }


def print_index_stats(pc: Optional[Pinecone] = None, index_name: Optional[str] = None):
    """
    Print index statistics in a readable format.
    
    Args:
        pc: Pinecone client
        index_name: Name of the index
    """
    stats = get_index_stats(pc, index_name)
    
    print(f"\n📊 Index Statistics ({index_name or Config.PINECONE_INDEX_NAME}):")
    print(f"   - Total vectors: {stats['total_vector_count']}")
    print(f"   - Dimension: {stats['dimension']}")
    print(f"   - Index fullness: {stats['index_fullness']}")
    
    if stats['namespaces']:
        print("   - Namespaces:")
        for ns, ns_stats in stats['namespaces'].items():
            print(f"     • '{ns}': {ns_stats.vector_count} vectors")


# =============================================================================
# Vector Store Operations
# =============================================================================

def upsert_documents(
    documents: List[Document],
    embeddings: Optional[OpenAIEmbeddings] = None,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    batch_size: int = 100
) -> PineconeVectorStore:
    """
    Upsert documents to Pinecone via LangChain's PineconeVectorStore.
    
    Args:
        documents: List of Document objects to upsert
        embeddings: Embeddings model (default: OpenAI)
        index_name: Name of the Pinecone index
        namespace: Namespace for the vectors
        batch_size: Batch size for upsertion
        
    Returns:
        PineconeVectorStore instance
    """
    embeddings = embeddings or get_openai_embeddings()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    namespace = namespace or Config.NAMESPACE
    
    print(f"💾 Upserting {len(documents)} documents to Pinecone...")
    print(f"   - Index: {index_name}")
    print(f"   - Namespace: {namespace}")
    
    # Use from_documents to create and upsert in one call
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )
    
    print(f"   ✓ Upsert complete")
    return vectorstore


def add_to_existing_vectorstore(
    vectorstore: PineconeVectorStore,
    documents: List[Document]
):
    """
    Add documents to an existing vector store.
    
    Args:
        vectorstore: Existing PineconeVectorStore instance
        documents: List of Document objects to add
    """
    print(f"➕ Adding {len(documents)} documents to existing vectorstore...")
    vectorstore.add_documents(documents)
    print(f"   ✓ Documents added")


def get_vectorstore(
    embeddings: Optional[OpenAIEmbeddings] = None,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    text_key: str = "text"
) -> PineconeVectorStore:
    """
    Get a PineconeVectorStore instance for querying existing data.
    
    Args:
        embeddings: Embeddings model
        index_name: Name of the Pinecone index
        namespace: Namespace for the vectors
        text_key: Key for text content in metadata
        
    Returns:
        PineconeVectorStore instance
    """
    embeddings = embeddings or get_openai_embeddings()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    namespace = namespace or Config.NAMESPACE
    
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key=text_key,
        namespace=namespace
    )


def delete_namespace(
    pc: Optional[Pinecone] = None,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    confirm: bool = True
):
    """
    Delete all vectors in a namespace.
    
    Args:
        pc: Pinecone client
        index_name: Name of the index
        namespace: Namespace to delete
        confirm: Whether to ask for confirmation
    """
    pc = pc or get_pinecone_client()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    namespace = namespace or Config.NAMESPACE
    
    if confirm:
        response = input(f"⚠️  Delete all vectors in namespace '{namespace}'? (yes/no): ")
        if response.lower() != "yes":
            print("   Deletion cancelled")
            return
    
    print(f"🗑️  Deleting namespace: {namespace}")
    index = pc.Index(index_name)
    index.delete(delete_all=True, namespace=namespace)
    print(f"   ✓ Namespace deleted")


# =============================================================================
# Metadata Filtering Helpers
# =============================================================================

def build_metadata_filter(
    category: Optional[str] = None,
    topic: Optional[str] = None,
    version: Optional[str] = None,
    entities: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a metadata filter dictionary for Pinecone queries.
    
    Args:
        category: Document category filter
        topic: Document topic filter
        version: Version filter
        entities: Entities filter (partial match)
        **kwargs: Additional filter fields
        
    Returns:
        Filter dictionary for Pinecone
    """
    filter_dict = {}
    
    if category:
        filter_dict["category"] = category
    if topic:
        filter_dict["topic"] = topic
    if version:
        filter_dict["version"] = version
    if entities:
        filter_dict["entities"] = {"$contains": entities}
    
    # Add any additional filters
    filter_dict.update(kwargs)
    
    return filter_dict if filter_dict else None


# =============================================================================
# Batch Operations
# =============================================================================

def upsert_in_batches(
    documents: List[Document],
    embeddings: Optional[OpenAIEmbeddings] = None,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    batch_size: int = 100
):
    """
    Upsert documents in batches to avoid timeouts with large document sets.
    
    Args:
        documents: List of Document objects
        embeddings: Embeddings model
        index_name: Name of the Pinecone index
        namespace: Namespace for the vectors
        batch_size: Number of documents per batch
    """
    embeddings = embeddings or get_openai_embeddings()
    index_name = index_name or Config.PINECONE_INDEX_NAME
    namespace = namespace or Config.NAMESPACE
    
    total = len(documents)
    print(f"💾 Upserting {total} documents in batches of {batch_size}...")
    
    # Process first batch with from_documents
    first_batch = documents[:batch_size]
    vectorstore = PineconeVectorStore.from_documents(
        documents=first_batch,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )
    
    # Process remaining batches with add_documents
    for i in range(batch_size, total, batch_size):
        batch = documents[i:i + batch_size]
        print(f"   Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}: {len(batch)} documents")
        vectorstore.add_documents(batch)
    
    print(f"   ✓ All {total} documents upserted")
    return vectorstore


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
    
    # Test index operations
    print("Testing Pinecone index operations...")
    print("-" * 60)
    
    # Create index if needed
    created = create_index_if_not_exists()
    
    # Print stats
    print_index_stats()
    
    print("\n✅ Pinecone module test complete")

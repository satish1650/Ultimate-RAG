"""
RAG Chain Module

Builds the complete Retrieval-Augmented Generation chain using LangChain.
Includes prompt templates, context formatting, and response generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config, get_openai_llm
from retrieval.retriever import (
    format_retrieved_context,
    get_basic_retriever,
    get_self_query_retriever,
    get_scored_retriever
)


# =============================================================================
# Prompt Templates
# =============================================================================

# Standard RAG prompt with metadata awareness
DEFAULT_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer the question. If you cannot find the answer
in the context, say "I don't have enough information to answer this question."

The context includes metadata tags showing the source, topic, category, and version
of each document chunk. Use this information to provide accurate and contextual answers.

Context:
{context}

Question: {question}

Answer:""")


# Detailed RAG prompt with citation instructions
CITATION_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a knowledgeable assistant that answers questions based on the provided documents.
Use only the information in the context to answer. Be concise but complete.

When referencing specific information, cite the source using the format [Source: filename, Page: X].

Context:
{context}

Question: {question}

Provide your answer below. If the context doesn't contain the answer, say so clearly.

Answer:""")


# Metadata-aware prompt that encourages using document metadata
METADATA_AWARE_PROMPT = ChatPromptTemplate.from_template("""
You are an expert assistant analyzing documents with rich metadata.
Answer the question using ONLY the provided context.

Each context section includes metadata tags showing:
- Source: The original document filename
- Page: Page number in the source
- Topic: The main topic of this section
- Category: Document type (technical, legal, financial, etc.)
- Version: Document/API version if mentioned
- Entities: Key people, organizations, or products mentioned

Use this metadata to provide accurate, contextual answers. If multiple sources
conflict, mention the discrepancy.

Context:
{context}

Question: {question}

Answer:""")


# Summary prompt for generating document summaries
SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
Provide a concise summary of the following documents. Highlight key points,
main topics, and important entities mentioned.

Documents:
{context}

Summary:""")


# =============================================================================
# Context Formatting
# =============================================================================

def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a context string with metadata.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant documents found."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        topic = doc.metadata.get('topic', 'N/A')
        category = doc.metadata.get('category', 'N/A')
        version = doc.metadata.get('version', '')
        
        version_str = f" | Version: {version}" if version else ""
        
        header = f"[{i}] Source: {source} | Page: {page} | Topic: {topic} | Category: {category}{version_str}"
        formatted.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)


def format_docs_with_summary(docs: List[Document]) -> str:
    """
    Format documents with their LangExtract summaries if available.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted context with summaries
    """
    if not docs:
        return "No relevant documents found."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        topic = doc.metadata.get('topic', 'N/A')
        category = doc.metadata.get('category', 'N/A')
        summary = doc.metadata.get('chunk_summary', '')
        
        header = f"[{i}] Source: {source} | Topic: {topic} | Category: {category}"
        
        if summary:
            formatted.append(f"{header}\nSummary: {summary}\nContent: {doc.page_content}")
        else:
            formatted.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)


def format_docs_minimal(docs: List[Document]) -> str:
    """
    Format documents with minimal metadata (just content).
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted context string with minimal metadata
    """
    if not docs:
        return "No relevant documents found."
    
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# =============================================================================
# RAG Chain Builders
# =============================================================================

def create_rag_chain(
    retriever=None,
    llm: Optional[ChatOpenAI] = None,
    prompt: Optional[ChatPromptTemplate] = None,
    format_func=None
) -> RunnableSerializable:
    """
    Create a standard RAG chain.
    
    Args:
        retriever: Document retriever (default: basic retriever)
        llm: Language model (default: from config)
        prompt: Prompt template (default: DEFAULT_RAG_PROMPT)
        format_func: Function to format documents (default: format_docs)
        
    Returns:
        Configured RAG chain
    """
    llm = llm or get_openai_llm(temperature=0)
    retriever = retriever or get_basic_retriever()
    prompt = prompt or DEFAULT_RAG_PROMPT
    format_func = format_func or format_docs
    
    # Build the chain
    chain = (
        {
            "context": retriever | format_func,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def create_self_query_rag_chain(
    llm: Optional[ChatOpenAI] = None,
    prompt: Optional[ChatPromptTemplate] = None,
    format_func=None,
    verbose: bool = True
) -> RunnableSerializable:
    """
    Create a RAG chain with SelfQueryRetriever for automatic metadata filtering.
    
    Args:
        llm: Language model
        prompt: Prompt template
        format_func: Function to format documents
        verbose: Whether to show retriever verbose output
        
    Returns:
        Configured RAG chain with self-query retrieval
    """
    llm = llm or get_openai_llm(temperature=0)
    prompt = prompt or DEFAULT_RAG_PROMPT
    format_func = format_func or format_docs
    
    # Create self-query retriever
    retriever = get_self_query_retriever(llm=llm, verbose=verbose)
    
    # Build the chain
    chain = (
        {
            "context": retriever | format_func,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def create_citation_rag_chain(
    retriever=None,
    llm: Optional[ChatOpenAI] = None
) -> RunnableSerializable:
    """
    Create a RAG chain that encourages source citations in responses.
    
    Args:
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Configured RAG chain with citation prompt
    """
    return create_rag_chain(
        retriever=retriever,
        llm=llm,
        prompt=CITATION_RAG_PROMPT,
        format_func=format_docs
    )


def create_metadata_aware_chain(
    retriever=None,
    llm: Optional[ChatOpenAI] = None
) -> RunnableSerializable:
    """
    Create a RAG chain with metadata-aware prompting.
    
    Args:
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Configured metadata-aware RAG chain
    """
    return create_rag_chain(
        retriever=retriever,
        llm=llm,
        prompt=METADATA_AWARE_PROMPT,
        format_func=format_docs
    )


def create_scored_retrieval_chain(
    score_threshold: float = None,
    llm: Optional[ChatOpenAI] = None,
    prompt: Optional[ChatPromptTemplate] = None
) -> RunnableSerializable:
    """
    Create a RAG chain with similarity score threshold filtering.
    
    Args:
        score_threshold: Minimum similarity score
        llm: Language model
        prompt: Prompt template
        
    Returns:
        Configured RAG chain with score filtering
    """
    llm = llm or get_openai_llm(temperature=0)
    score_threshold = score_threshold or Config.SCORE_THRESHOLD
    prompt = prompt or DEFAULT_RAG_PROMPT
    
    # Create scored retriever
    retriever = get_scored_retriever(score_threshold=score_threshold)
    
    # Build the chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# =============================================================================
# Legacy Chain Builders (using LCEL)
# =============================================================================

def create_legacy_retrieval_chain(
    retriever=None,
    llm: Optional[ChatOpenAI] = None
):
    """
    Create a retrieval chain using the older LangChain API.
    Useful for compatibility with older LangChain versions.
    
    Args:
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Legacy retrieval chain
    """
    llm = llm or get_openai_llm(temperature=0)
    retriever = retriever or get_basic_retriever()
    
    # Create the document combination chain
    document_prompt = PromptTemplate.from_template(
        "Content: {page_content}\nMetadata: {metadata}"
    )
    
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=DEFAULT_RAG_PROMPT,
        document_prompt=document_prompt
    )
    
    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )
    
    return retrieval_chain


# =============================================================================
# Query Interface
# =============================================================================

class RAGQueryEngine:
    """
    High-level query engine for the RAG system.
    Provides a simple interface for asking questions.
    """
    
    def __init__(
        self,
        chain: Optional[RunnableSerializable] = None,
        retriever=None,
        llm: Optional[ChatOpenAI] = None,
        verbose: bool = False
    ):
        """
        Initialize the RAG query engine.
        
        Args:
            chain: Pre-configured RAG chain (creates default if None)
            retriever: Document retriever
            llm: Language model
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        
        if chain:
            self.chain = chain
        else:
            self.chain = create_rag_chain(retriever=retriever, llm=llm)
        
        if retriever:
            self.retriever = retriever
        else:
            from retrieval.retriever import get_basic_retriever
            self.retriever = get_basic_retriever()
    
    def query(self, question: str) -> str:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        if self.verbose:
            print(f"\n❓ Question: {question}")
            print("-" * 60)
        
        response = self.chain.invoke(question)
        return response
    
    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with source documents.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.verbose:
            print(f"\n❓ Question: {question}")
            print("-" * 60)
        
        # Retrieve documents
        docs = self.retriever.invoke(question)
        
        if self.verbose:
            print(f"📚 Retrieved {len(docs)} documents")
        
        # Get answer
        answer = self.chain.invoke(question)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def query_rag(
    question: str,
    chain: Optional[RunnableSerializable] = None
) -> str:
    """
    Simple function to query the RAG system.
    
    Args:
        question: The question to answer
        chain: Optional pre-configured chain
        
    Returns:
        Generated answer
    """
    if chain is None:
        chain = create_rag_chain()
    
    return chain.invoke(question)


def query_with_self_query_retrieval(question: str) -> str:
    """
    Query using SelfQueryRetriever for automatic metadata filtering.
    
    Args:
        question: The question to answer
        
    Returns:
        Generated answer
    """
    chain = create_self_query_rag_chain()
    return chain.invoke(question)


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Validate configuration
    missing = []
    if not Config.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    
    if missing:
        print(f"❌ Missing configuration: {', '.join(missing)}")
        sys.exit(1)
    
    print("Testing RAG chain module...")
    print("-" * 60)
    
    # Test chain creation
    print("\n1. Testing RAG chain creation...")
    try:
        chain = create_rag_chain()
        print("   ✓ Standard RAG chain created")
    except Exception as e:
        print(f"   ⚠️  RAG chain creation failed: {e}")
    
    # Test self-query chain creation
    print("\n2. Testing SelfQuery RAG chain...")
    try:
        sq_chain = create_self_query_rag_chain()
        print("   ✓ SelfQuery RAG chain created")
    except Exception as e:
        print(f"   ⚠️  SelfQuery RAG chain failed: {e}")
    
    # Test query engine
    print("\n3. Testing QueryEngine...")
    try:
        engine = RAGQueryEngine()
        print("   ✓ QueryEngine created")
    except Exception as e:
        print(f"   ⚠️  QueryEngine failed: {e}")
    
    print("\n✅ RAG chain module test complete")

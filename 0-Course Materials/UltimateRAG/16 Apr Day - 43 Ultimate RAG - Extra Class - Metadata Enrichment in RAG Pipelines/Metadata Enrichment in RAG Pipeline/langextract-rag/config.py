"""
Configuration module for LangExtract-Enhanced RAG System.
Loads environment variables and initializes clients.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

# Load environment variables
load_dotenv()


# =============================================================================
# Environment Variables
# =============================================================================

class Config:
    """Application configuration loaded from environment variables."""
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "langextract-rag")
    
    # LangExtract (falls back to OPENAI_API_KEY if not set)
    LANGEXTRACT_API_KEY: str = os.getenv("LANGEXTRACT_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    
    # Application Settings
    NAMESPACE: str = os.getenv("NAMESPACE", "langextract-enriched")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    METADATA_MODEL: str = os.getenv("METADATA_MODEL", "gpt-4o-mini")
    
    # Pinecone Index Settings
    INDEX_DIMENSION: int = 1536  # text-embedding-3-small dimension
    INDEX_METRIC: str = "cosine"
    INDEX_CLOUD: str = "aws"
    INDEX_REGION: str = "us-east-1"
    
    # Text Splitter Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval Settings
    DEFAULT_K: int = 5
    SCORE_THRESHOLD: float = 0.7


# =============================================================================
# Initialize Clients
# =============================================================================

def get_openai_embeddings() -> OpenAIEmbeddings:
    """Initialize and return OpenAI embeddings client."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment variables")
    
    return OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.OPENAI_API_KEY
    )


def get_openai_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Initialize and return OpenAI LLM client."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment variables")
    
    return ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=temperature,
        api_key=Config.OPENAI_API_KEY
    )


def get_pinecone_client() -> Pinecone:
    """Initialize and return Pinecone client."""
    if not Config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in environment variables")
    
    return Pinecone(api_key=Config.PINECONE_API_KEY)


def validate_config() -> list[str]:
    """Validate configuration and return list of missing settings."""
    missing = []
    
    if not Config.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not Config.PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    
    return missing


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    # Validate configuration
    missing = validate_config()
    
    if missing:
        print(f"❌ Missing configuration: {', '.join(missing)}")
        print("Please set these in your .env file")
    else:
        print("✅ Configuration validated successfully")
        print(f"   - Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"   - LLM Model: {Config.LLM_MODEL}")
        print(f"   - Pinecone Index: {Config.PINECONE_INDEX_NAME}")

"""
LangExtract Metadata Extraction Module

This module uses Google's LangExtract to generate rich, structured metadata
from text chunks, enabling metadata-filtered retrieval in the RAG pipeline.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import langextract as lx
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# =============================================================================
# Extraction Schema Definition
# =============================================================================

# Default extraction prompt for generic documents
DEFAULT_EXTRACTION_PROMPT = textwrap.dedent("""\
    Extract structured metadata from the given text document chunk.
    For each chunk, identify:
    - document_topic: The primary topic/subject of this text
    - document_category: The category (e.g., technical, legal, financial, medical, general)
    - key_entities: Important named entities (people, organizations, products, locations)
    - document_version: Any version numbers or dates mentioned
    - summary: A 1-2 sentence summary of this chunk
    
    Use exact text for extractions where possible. Do not paraphrase or infer
    information not present in the text.
""")


# =============================================================================
# Few-Shot Examples
# =============================================================================

DEFAULT_EXAMPLES = [
    lx.data.ExampleData(
        text=("Section 3.2: Authentication API v2.1 - Released January 2025. "
              "The OAuth2 module supports both client credentials and "
              "authorization code flows. Maximum token lifetime is 3600 seconds. "
              "Contact the Platform Engineering team for API key provisioning."),
        extractions=[
            lx.data.Extraction(
                extraction_class="document_topic",
                extraction_text="Authentication API",
                attributes={
                    "specific_topic": "OAuth2 authentication flows"
                }
            ),
            lx.data.Extraction(
                extraction_class="document_category",
                extraction_text="technical",
                attributes={
                    "subcategory": "API documentation"
                }
            ),
            lx.data.Extraction(
                extraction_class="key_entities",
                extraction_text="Platform Engineering",
                attributes={
                    "entity_type": "team"
                }
            ),
            lx.data.Extraction(
                extraction_class="key_entities",
                extraction_text="OAuth2",
                attributes={
                    "entity_type": "protocol"
                }
            ),
            lx.data.Extraction(
                extraction_class="document_version",
                extraction_text="v2.1",
                attributes={
                    "release_date": "January 2025"
                }
            ),
            lx.data.Extraction(
                extraction_class="summary",
                extraction_text=("OAuth2 authentication API documentation "
                               "covering client credentials and authorization "
                               "code flows with token management."),
                attributes={}
            ),
        ]
    )
]


# =============================================================================
# Metadata Extraction Functions
# =============================================================================

def extract_metadata_for_chunk(
    chunk_text: str,
    prompt_description: Optional[str] = None,
    examples: Optional[List] = None,
    model_id: str = None,
    api_key: Optional[str] = None,
    max_workers: int = 1
) -> Dict[str, Any]:
    """
    Run LangExtract on a single chunk and return structured metadata dict.
    
    IMPORTANT for OpenAI:
    - fence_output=True is required
    - use_schema_constraints=False (schema constraints not yet supported for OpenAI)
    
    Args:
        chunk_text: The text content to extract metadata from
        prompt_description: Custom extraction prompt (uses default if None)
        examples: Few-shot examples (uses default if None)
        model_id: LLM model ID (default: Config.METADATA_MODEL)
        api_key: API key for LLM (default: Config.LANGEXTRACT_API_KEY)
        max_workers: Parallel workers for extraction
        
    Returns:
        Dictionary containing extracted metadata
    """
    prompt = prompt_description or DEFAULT_EXTRACTION_PROMPT
    examples = examples or DEFAULT_EXAMPLES
    model_id = model_id or Config.METADATA_MODEL
    api_key = api_key or Config.LANGEXTRACT_API_KEY
    
    try:
        # Run LangExtract extraction
        result = lx.extract(
            text_or_documents=chunk_text,
            prompt_description=prompt,
            examples=examples,
            model_id=model_id,
            api_key=api_key,
            fence_output=True,           # Required for OpenAI
            use_schema_constraints=False,  # Required for OpenAI
            max_workers=max_workers
        )
        
        # Parse extractions into a flat metadata dict
        metadata = _parse_extractions(result.extractions)
        return metadata
        
    except Exception as e:
        print(f"   ⚠️  Extraction error: {e}")
        # Return empty metadata on error to allow processing to continue
        return {
            "topic": "",
            "category": "general",
            "entities": "",
            "version": "",
            "release_date": "",
            "chunk_summary": "",
            "extraction_error": str(e)
        }


def _parse_extractions(extractions: List) -> Dict[str, Any]:
    """
    Parse LangExtract extractions into a flat metadata dictionary.
    
    Args:
        extractions: List of Extraction objects from LangExtract
        
    Returns:
        Flat dictionary with metadata fields
    """
    metadata = {
        "topic": "",
        "specific_topic": "",
        "category": "general",
        "subcategory": "",
        "entities": "",
        "version": "",
        "release_date": "",
        "chunk_summary": ""
    }
    
    entities_list = []
    
    for extraction in extractions:
        cls = extraction.extraction_class
        text = extraction.extraction_text
        attrs = extraction.attributes or {}

        # gpt-4o-mini (and other models) may return a list for key_entities
        # instead of separate scalar Extraction objects. LangExtract's internal
        # validator logs a schema error and skips the extraction when it sees a
        # list, but we still get it here — coerce to a safe scalar string.
        if text is None:
            text = ""
        elif isinstance(text, list):
            text = ", ".join(str(t) for t in text)
        elif not isinstance(text, (str, int, float)):
            text = str(text)
        text = str(text).strip()

        if cls == "document_topic":
            metadata["topic"] = text
            metadata["specific_topic"] = attrs.get("specific_topic", "")

        elif cls == "document_category":
            metadata["category"] = text
            metadata["subcategory"] = attrs.get("subcategory", "")

        elif cls == "key_entities":
            if text:
                entities_list.append(text)

        elif cls == "document_version":
            metadata["version"] = text
            metadata["release_date"] = attrs.get("release_date", "")

        elif cls == "summary":
            metadata["chunk_summary"] = text

    # Join all entities as comma-separated string
    if entities_list:
        metadata["entities"] = ", ".join(entities_list)
    
    return metadata


def enrich_chunks_with_metadata(
    chunks: List[Document],
    prompt_description: Optional[str] = None,
    examples: Optional[List] = None,
    model_id: str = None,
    show_progress: bool = True
) -> List[Document]:
    """
    Process each chunk through LangExtract to generate rich metadata,
    then return LangChain Document objects with enriched metadata.
    
    Args:
        chunks: List of Document chunks to enrich
        prompt_description: Custom extraction prompt
        examples: Few-shot examples
        model_id: LLM model ID
        show_progress: Whether to print progress updates
        
    Returns:
        List of enriched Document objects
    """
    enriched_docs = []
    total = len(chunks)
    
    if show_progress:
        print(f"🧠 Extracting metadata with LangExtract for {total} chunks...")
        print(f"   - Model: {model_id or Config.METADATA_MODEL}")
        print(f"   - Note: This makes {total} LLM API calls (one per chunk)")
    
    for i, chunk in enumerate(chunks):
        if show_progress and (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{total} chunks...")
        
        # Extract metadata using LangExtract
        extracted_meta = extract_metadata_for_chunk(
            chunk_text=chunk.page_content,
            prompt_description=prompt_description,
            examples=examples,
            model_id=model_id
        )
        
        # Merge with existing metadata (page number, source, etc.)
        combined_metadata = {
            **chunk.metadata,           # Original metadata (source, page, etc.)
            **extracted_meta,          # LangExtract metadata
            "chunk_index": i,
        }
        
        # Create enriched document
        enriched_doc = Document(
            page_content=chunk.page_content,
            metadata=combined_metadata
        )
        enriched_docs.append(enriched_doc)
    
    if show_progress:
        print(f"   ✓ Metadata extraction complete for {len(enriched_docs)} chunks")
    
    return enriched_docs


def extract_document_level_metadata(
    full_text: str,
    prompt_description: Optional[str] = None,
    examples: Optional[List] = None,
    model_id: str = None,
    max_workers: int = 5,
    extraction_passes: int = 2
) -> Dict[str, Any]:
    """
    Extract metadata from the full document (cost-effective for large PDFs).
    This extracts once per document instead of once per chunk.
    
    Args:
        full_text: Full document text
        prompt_description: Custom extraction prompt
        examples: Few-shot examples
        model_id: LLM model ID
        max_workers: Parallel workers
        extraction_passes: Multiple passes for higher recall
        
    Returns:
        Dictionary containing extracted metadata
    """
    prompt = prompt_description or DEFAULT_EXTRACTION_PROMPT
    examples = examples or DEFAULT_EXAMPLES
    model_id = model_id or Config.METADATA_MODEL
    api_key = Config.LANGEXTRACT_API_KEY
    
    print(f"🧠 Extracting document-level metadata...")
    print(f"   - Model: {model_id}")
    print(f"   - Passes: {extraction_passes}")
    print(f"   - Workers: {max_workers}")
    
    try:
        result = lx.extract(
            text_or_documents=full_text,
            prompt_description=prompt,
            examples=examples,
            model_id=model_id,
            api_key=api_key,
            fence_output=True,
            use_schema_constraints=False,
            max_workers=max_workers,
            extraction_passes=extraction_passes
        )
        
        metadata = _parse_extractions(result.extractions)
        print(f"   ✓ Document-level extraction complete")
        return metadata
        
    except Exception as e:
        print(f"   ⚠️  Document extraction error: {e}")
        return {
            "topic": "",
            "category": "general",
            "entities": "",
            "version": "",
            "release_date": "",
            "chunk_summary": "",
            "extraction_error": str(e)
        }


def propagate_metadata_to_chunks(
    chunks: List[Document],
    document_metadata: Dict[str, Any]
) -> List[Document]:
    """
    Propagate document-level metadata to all chunks.
    Useful for cost optimization - extract once, apply to all chunks.
    
    Args:
        chunks: List of Document chunks
        document_metadata: Metadata extracted at document level
        
    Returns:
        List of enriched Document objects
    """
    enriched_docs = []
    
    for i, chunk in enumerate(chunks):
        # Merge document-level metadata with chunk metadata
        combined_metadata = {
            **chunk.metadata,
            **document_metadata,
            "chunk_index": i,
            "metadata_source": "document_level"
        }
        
        enriched_doc = Document(
            page_content=chunk.page_content,
            metadata=combined_metadata
        )
        enriched_docs.append(enriched_doc)
    
    return enriched_docs


# =============================================================================
# Caching and Persistence
# =============================================================================

def save_extraction_results(
    documents: List[Document],
    output_path: str | Path
):
    """
    Save extraction results to a JSONL file for later use.
    
    Args:
        documents: List of enriched Document objects
        output_path: Path to save the results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for doc in documents:
            record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"💾 Saved {len(documents)} documents to {output_path}")


def load_extraction_results(
    input_path: str | Path
) -> List[Document]:
    """
    Load previously saved extraction results.
    
    Args:
        input_path: Path to the JSONL file
        
    Returns:
        List of Document objects
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    documents = []
    with open(input_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            doc = Document(
                page_content=record["page_content"],
                metadata=record["metadata"]
            )
            documents.append(doc)
    
    print(f"📂 Loaded {len(documents)} documents from {input_path}")
    return documents


# =============================================================================
# Utility Functions
# =============================================================================

def print_metadata_sample(documents: List[Document], n: int = 3):
    """
    Print metadata for a sample of documents.
    
    Args:
        documents: List of Document objects
        n: Number of documents to print
    """
    print(f"\n📋 Metadata sample (first {min(n, len(documents))} documents):")
    print("-" * 60)
    
    for i, doc in enumerate(documents[:n]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"  Category: {doc.metadata.get('category', 'N/A')}")
        print(f"  Entities: {doc.metadata.get('entities', 'N/A')}")
        print(f"  Version: {doc.metadata.get('version', 'N/A')}")
        print(f"  Summary: {doc.metadata.get('chunk_summary', 'N/A')[:100]}...")


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    # Test the module with sample text
    sample_text = """
    API Reference Guide v3.2 - March 2025
    
    The Payment Processing API enables secure transaction handling
    for merchants. Supported payment methods include credit cards,
    digital wallets, and bank transfers. Contact the FinTech Team
    for API credentials and rate limit information.
    """
    
    print("Testing LangExtract metadata extraction...")
    print("-" * 60)
    
    metadata = extract_metadata_for_chunk(sample_text)
    
    print("\n✅ Extracted Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

# LangExtract-Enhanced RAG System

A production-ready RAG (Retrieval-Augmented Generation) pipeline that uses **Google's LangExtract** to generate rich, structured metadata from PDF documents, then leverages **LangChain** for orchestration, **OpenAI** for embeddings & LLM, and **Pinecone** as the vector store — enabling **metadata-filtered retrieval** for dramatically improved precision.

## 🌟 Key Features

- **🧠 LangExtract Integration**: Automatically extracts structured metadata (topic, category, entities, version, summary) from documents
- **🔍 Metadata-Filtered Retrieval**: Filter search results by document type, version, entities, and more
- **🤖 Self-Querying Retriever**: LLM automatically generates metadata filters from natural language queries
- **📄 PDF Processing**: Load and chunk PDF documents with intelligent text splitting
- **💾 Pinecone Vector Store**: Scalable vector storage with metadata filtering
- **🚀 Multiple Retrieval Strategies**: Basic, filtered, scored, and self-query retrieval

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│   PDF Input   │────▶│  PDF Loader       │────▶│  Text Splitter       │
│  (Documents)  │     │  (PyPDFLoader)    │     │  (RecursiveChar...)  │
└──────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │   LangExtract        │
                                               │   Metadata Engine    │
                                               │   (OpenAI GPT-4o)    │
                                               └──────────┬───────────┘
                                                          │
                                    Enriched Chunks       │
                                    (text + metadata)     ▼
                                               ┌──────────────────────┐
                                               │  OpenAI Embeddings   │
                                               │ (text-embedding-3-   │
                                               │       small/large)   │
                                               └──────────┬───────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │    Pinecone          │
                                               │    Vector Store      │
                                               │  (vectors+metadata)  │
                                               └──────────┬───────────┘
                                                          │
                             ┌────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│              QUERY PHASE                                  │
│  User Query ──▶ Metadata Filter Generation (LLM)        │
│             ──▶ Filtered Similarity Search              │
│             ──▶ Context Assembly ──▶ LLM Response       │
└──────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Framework** | LangChain | 0.3+ |
| **Metadata Engine** | LangExtract (Google) | 1.1+ |
| **LLM** | OpenAI GPT-4o / GPT-4o-mini | Latest |
| **Embeddings** | OpenAI text-embedding-3-small | Latest |
| **Vector DB** | Pinecone (Serverless) | 6.x |
| **PDF Processing** | PyPDFLoader | Latest |
| **Chunking** | RecursiveCharacterTextSplitter | Latest |

## 📦 Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- Pinecone API key

### Setup

1. **Clone the repository** (or create the project structure):
```bash
cd langextract-rag
```

2. **Create virtual environment**:
```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
# Core pipeline
uv pip install -e .

# Core + notebook dependencies
uv pip install -e ".[notebook]"

# Core + notebook + dev tools
uv pip install -e ".[notebook,dev]"
```

4. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-xxxxx

# Pinecone Configuration
PINECONE_API_KEY=pcsk_xxxxx
PINECONE_INDEX_NAME=langextract-rag

# LangExtract Configuration (Optional - uses OPENAI_API_KEY if not set)
LANGEXTRACT_API_KEY=sk-xxxxx

# Application Configuration
NAMESPACE=langextract-enriched
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
METADATA_MODEL=gpt-4o-mini
```

## 🚀 Usage

### 1. Ingest a PDF Document

```bash
# Ingest a single PDF
python main.py ingest --pdf path/to/document.pdf

# Ingest all PDFs in a directory
python main.py ingest --dir path/to/pdfs/

# Use document-level extraction (faster, cheaper)
python main.py ingest --pdf document.pdf --doc-level

# Save extraction results for debugging
python main.py ingest --pdf document.pdf --save results.jsonl
```

### 2. Query the System

```bash
# Basic query
python main.py query "What are the API rate limits?"

# Query with SelfQueryRetriever (automatic metadata filtering)
python main.py query "What are the API limits in v2.1?" --self-query

# Query with metadata-aware prompt
python main.py query "Explain the authentication process" --metadata-aware
```

### 3. Interactive Mode

```bash
python main.py interactive
```

Commands in interactive mode:
- `/sq` - Toggle SelfQueryRetriever
- `/meta` - Toggle metadata-aware mode
- `/quit` - Exit

### 4. Other Commands

```bash
# Show index statistics
python main.py stats

# Compare retrieval methods
python main.py compare "What are the API limits?"

# Clear namespace (deletes all vectors)
python main.py clear
```

## 📖 Python API Usage

### Basic Example

```python
from ingest import load_pdf, chunk_documents, enrich_chunks_with_metadata
from ingest.pinecone_store import create_index_if_not_exists, upsert_documents
from retrieval.rag_chain import query_rag_system

# Load PDF
documents = load_pdf("document.pdf")

# Chunk
chunks = chunk_documents(documents)

# Extract metadata with LangExtract
enriched_docs = enrich_chunks_with_metadata(chunks)

# Store in Pinecone
create_index_if_not_exists()
upsert_documents(enriched_docs)

# Query
answer = query_rag_system("What are the rate limits?")
print(answer)
```

### Using Self-Query Retriever

```python
from retrieval.rag_chain import create_self_query_rag_chain

# Create chain with automatic metadata filtering
chain = create_self_query_rag_chain(verbose=True)

# Query - LLM automatically extracts filters
answer = chain.invoke("What authentication methods are available in API version 2.1?")
```

### Metadata-Filtered Search

```python
from retrieval.retriever import search_with_filter

# Search with explicit metadata filters
results = search_with_filter(
    query="OAuth2 authentication",
    filter_dict={
        "category": "technical",
        "version": "v2.1"
    }
)
```

## 📊 Project Structure

```
langextract-rag/
├── .env                          # Environment variables (gitignored)
├── .env.example                  # Example environment file
├── pyproject.toml                # Dependencies and tool config (replaces requirements.txt)
├── config.py                     # Configuration and client initialization
├── main.py                       # CLI entry point
├── README.md                     # This file
│
├── ingest/                       # Ingestion pipeline
│   ├── __init__.py
│   ├── pdf_loader.py            # PDF loading and chunking
│   ├── metadata_extractor.py    # LangExtract metadata extraction
│   └── pinecone_store.py        # Pinecone vector store operations
│
├── retrieval/                    # Retrieval and RAG
│   ├── __init__.py
│   ├── retriever.py             # Retrieval strategies
│   └── rag_chain.py             # RAG chain builders
│
├── evaluation/                   # Evaluation and visualization
│   ├── __init__.py
│   └── visualize.py             # Debugging and visualization tools
│
├── notebooks/                    # Interactive tutorials
│   └── metadata_enrichment_tutorial.ipynb  # Metadata enrichment walkthrough
│
└── data/                         # Data directory
    └── pdfs/                    # Input PDF documents
```

## 🎯 Key Capabilities

### 1. Rich Metadata Extraction

LangExtract automatically extracts:
- **Topic**: Primary subject of the text
- **Category**: Document type (technical, legal, financial, medical, general)
- **Entities**: Key people, organizations, products
- **Version**: Document/API version numbers
- **Summary**: Brief summary of the chunk

### 2. Multiple Retrieval Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Basic** | Pure semantic search | General queries |
| **Filtered** | Metadata filter + semantic search | Known constraints |
| **Self-Query** | LLM generates filters automatically | Natural language queries |
| **Scored** | Similarity threshold filtering | Quality control |

### 3. Cost Optimization

- Use `gpt-4o-mini` for metadata extraction (10x cheaper than GPT-4o)
- Document-level extraction: Extract once, apply to all chunks
- Batch processing with configurable batch sizes
- Caching extraction results to JSONL

## 📓 Interactive Tutorial Notebook

`notebooks/metadata_enrichment_tutorial.ipynb` walks through the full metadata enrichment story for advanced learners who already know RAG:

| Section | Topic | API calls? |
|---------|-------|------------|
| 1 | The sparse metadata problem (synthetic data) | None |
| 2 | How LangExtract works under the hood | None |
| 3 | Chunk-level vs. document-level extraction (cost trade-off) | None |
| 4 | Live enrichment pipeline | Yes (1 per chunk) |
| 5 | Four retrieval strategies head-to-head with Chroma | Yes (embeddings) |
| 6 | Metadata in the LLM prompt: chain comparison | Yes (3 LLM calls) |
| 7 | Exercises with solutions | Optional |

**Quick start:**
```bash
uv pip install -e ".[notebook]"
jupyter lab notebooks/metadata_enrichment_tutorial.ipynb
```

---

## 🧪 Evaluation

Compare retrieval performance:

```python
from retrieval.retriever import compare_retrieval_methods, print_comparison_results

results = compare_retrieval_methods("What are the API limits?")
print_comparison_results(results)
```

## 🐛 Debugging

### Visualize Metadata

```python
from evaluation.visualize import (
    debug_document_metadata,
    print_metadata_analysis,
    generate_html_visualization
)

# Print detailed metadata
debug_document_metadata(documents, n=5)

# Analyze metadata coverage
print_metadata_analysis(documents)

# Generate HTML visualization
generate_html_visualization("results.jsonl", "visualization.html")
```

### Debug Retrieval

```python
from evaluation.visualize import debug_retrieval_results

results = search_with_filter(query, filter_dict)
debug_retrieval_results(query, results, method="filtered")
```

## ⚠️ Important Notes

1. **OpenAI + LangExtract Constraints**: When using OpenAI models with LangExtract, you **must** set `fence_output=True` and `use_schema_constraints=False` — schema constraints are not yet implemented for OpenAI in LangExtract.

2. **LangExtract Entity Coercion**: Some OpenAI models (e.g. `gpt-4o-mini`) return `key_entities` as a Python list instead of scalar strings. The `_parse_extractions` function in `ingest/metadata_extractor.py` automatically coerces list values to comma-separated strings, so no intervention is needed.

3. **Chroma Filter Format**: When using Chroma as the vector store (e.g. in the notebook), multi-field filters must use Chroma's `$and` operator — a flat dict with multiple keys is rejected. Use the `to_chroma_filter()` helper (defined in the notebook) to convert automatically.

4. **Pinecone Version**: This project requires `pinecone>=6.0.0,<8.0.0` due to `langchain-pinecone 0.2.x` compatibility constraints. Do not upgrade to Pinecone 8.x until `langchain-pinecone` adds support.

5. **Pinecone Metadata Limitations**: 
   - Metadata values must be strings, numbers, booleans, or lists of strings
   - No nested objects
   - Keep total metadata under 40KB per vector

6. **Cost Considerations**: Each chunk processed by LangExtract = 1 LLM API call. For 500 chunks, that's 500 calls. Consider document-level extraction for large document sets.

7. **Python Version**: Requires Python 3.12+.

## 📚 References

- [LangExtract Documentation](https://github.com/google/langextract)
- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## 📄 License

This project is provided as an example implementation. Please ensure compliance with the licenses of all dependencies:
- LangExtract: Apache 2.0
- LangChain: MIT
- Pinecone: Proprietary (see Pinecone terms)
- OpenAI: Proprietary (see OpenAI terms)

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows the existing style
2. Add tests for new features
3. Update documentation as needed

## 🙏 Acknowledgments

- Google LangExtract team for the powerful metadata extraction library
- LangChain team for the excellent orchestration framework
- Pinecone team for the scalable vector database

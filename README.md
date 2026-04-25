![alt text](Ultimate_RAG.png) <br>
---

Ultimate RAG Course
---
Designed for AI engineers, ML practitioners, and software developers, this course transforms RAG theory into production-grade implementations. Whether you're new to retrieval systems or experienced with LLMs, you'll gain expertise in Building scalable RAG pipelines (from basic
similarity search to knowledge graph-enhanced retrieval), Implementing enterprise features (hybrid search, MCP integration, self-improving agents), Deploying optimized systems (Docker, FastAPI, Kubernetes, AWS EKS), Solving real-world challenges (multimodal processing,
conversational memory, code-aware RAG). Through hands-on projects, you'll create portfolio-worthy systems including MCP-integrated code assistants and cloud-deployed multimodal RAG, all with professional monitoring and evaluation frameworks. <br>

Learning Objectives <br>
---
✅ Implement End-to-End RAG: From document ingestion (Unstructured.io, LlamaParse) to response synthesis (GPT-5, Claude 3.5) <br>
✅ Self-RAG with confidence scoring <br>
✅ Graph RAG with entity resolution <br>
✅ Model Context Protocol (MCP) integration <br>
✅ Optimize Production Systems <br>
✅ Kubernetes orchestration (EKS) <br>
✅ Advanced caching/load balancing <br>
✅ CI/CD with GitHub Actions/AWS CodePipeline <br>
✅ Build Specialized Applications <br>
✅ Multimodal RAG (CLIP + Whisper) <br>
✅ Smart code review with MCP servers <br>
✅ Agentic workflows with ReAct <br>

# 📚 Learning Topics Related to Ultimate RAG Course
This course is designed to provide students with a foundational, depth and practical understanding of RAG and Agents.
## 02 Nov Day - 1 👉 Induction Session <br>
### Induction Session Regarding Ultimate RAG Course
#### 🎯 Learning Objectives
- 📝 Agenda of Ultimate RAG course
- ⏱️ Timeline to complete the course
- 🎯 Objectives
- 📚 Learning Topics of Ultimate RAG course
- 💬 Q&A

## 08 Nov Day - 2 👉 Tools And Installation <br>
### General Software Checklist
#### 🎯 Learning Objectives
- [ ]  Python Distribution
- [ ]  IDE
- [ ]  GIT ——> GIT BASH
- [ ]  DOCKER
- [ ]  POSTMAN (Optional)
- [ ]  WARP (Optional)
- [ ]  AWS CLI
### Python Distributions(Any One)
- [ ] Anaconda (Majorly we will use in this course)
- [ ] Python
### Package Managers(Any One)
- [ ]  Conda (*Rarely*)
- [ ]  pip (*Majorly we will use in this course*)
- [ ]  uv (*Optional*)
### Local IDE or Text Editor(Any One)
- [ ]  VSCODE (*Majorly we will use in this course*)
- [ ]  CURSOR *PAID*
- [ ]  WINDSURF *PAID*
- [ ]  TRAE
- [ ]  PYCHARM Community or Professional
- [ ]  KIRO
### OPTIONAL : Browser based Online IDE
- [ ]  Lightning.ai
- [ ]  Replit
- [ ]  Firebase Studio
### Working with virtual environment
- [ ]  venv
- [ ]  conda
- [ ]  uv
### Version Control System
- [ ]  GIT (*Always we use in the course*)
- [ ]  Github Desktop (*Optional)*
### Extra Tools
- [ ] Markdown
    - Markdown Basics
- [ ] Docker
    - Docker Basics
- [ ] Cloud CLI
    - AWS CLI
    - GCP CLI
- [ ] API Tool
    - Postman
    - Warp Terminal

## 09 Nov Day - 3 👉 Introduction To RAG <br>
### 📚🔎🧠 Introduction to RAG
#### 🎯 Learning Objectives
- RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP
    - Core Innovation
    - How It Works
    - Key Results
    - Advantages
    - RAG Architecture
- Additional Study Materials
    - NLP Book → https://web.stanford.edu/~jurafsky/slp3/
    - Seq2Seq Model → https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
    - Transformers → https://jalammar.github.io/illustrated-transformer/
    - Udemy NLP Course → https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/?couponCode=ACCAGE0923
    - Transformer Explainer → https://www.youtube.com/watch?v=csWluHwfsB8

#### `Explained This Research Paper in breif` 👉 https://arxiv.org/pdf/2005.11401

## 15 Nov Day - 4 👉 RAG Architectures & Survey <br>
### 📚🔎🧠 RAG Detailed Survey
#### Retrieval-Augmented Generation for Large Language Models: A Survey
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
- Key Contributions
    - Three RAG Paradigms
    -  Core Components Analysis
    - Comprehensive Evaluation Framework
- Key Insights
    - RAG Challenges
    - RAG vs. Fine-tuning
    - Future Directions
- Practical Value
- Research Papers
    1. Pre-training (Orange/Yellow Branch - Left)
    2. Fine-tuning (Green Branch - Center)
    3.  Inference (Dark Blue/Teal Branch - Right)
- Three RAG Paradigms
    1. Naive RAG (Left - Blue)
    2. Advanced RAG (Middle - Orange)
    3. Modular RAG (Right - Blue/Orange)
- Evolution Timeline
- Practical Examples
    - Naive RAG
    - Advanced RAG
    - Modular RAG
    - Iterative RAG
    - Recursive RAG
    - Adaptive RAG
- When to Use Each
    - Use Iterative when
    - Use Recursive when
    - Use Adaptive when

#### `Explained This Research Paper in breif` 👉 https://arxiv.org/pdf/2312.10997

## 16 Nov Day - 5 👉 Exploring Langchain For RAG <br>
### Exploring Langchain
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
- Updated Requirements.txt
- LLM Selection OPENAI or GEMINI
- Google Gemini Embeddings
- 🎯 Maximum Marginal Relevance (MMR)
    - How MMR Works
    - MMR vs Cosine Similarity
    - MMR would return
### 📋 All LangChain Vector Store Search Types
1. Similarity Search (Default)
1. MMR (Maximum Marginal Relevance)
1. Similarity Score Threshold
1. Metadata Filtering
### 📝Assignment
#### 🚀Building a Hybrid Retriever System
You are tasked with building a **Study Assistant** naive rag system that helps students learn about a technical topic (e.g., Machine Learning, Python Programming, or Data Science).

The challenge: Students often need both **specific course material** (lecture notes, textbook excerpts) and **general background knowledge** (definitions, historical context, broader concepts).

💿Dataset
- Build your own dataset with multiple file formats including PDFS, HTML, TXT Files.
- Create 5-7 sample documents on a topic of your choice

***Choose any embedding model, any LLM, or any vector database of your choice***

**Your solution should:**
- Use a vector store to search through local study materials
- Use an external retriever (Wikipedia) to provide general knowledge
- Combine both sources intelligently to give comprehensive answers

## 22 Nov Day - 6 👉 Exploring Langchain: Part-1
### 📁 Notebook 02: Document Loaders
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will be able to:
1. Load documents from **PDF files** using PyPDFLoader
2. Load structured data from **CSV files**
3. Load JSON data from **API responses** or files
4. Scrape and load content from **web pages** (HTML)
5. Load **text files** and **markdown files**
6. **Batch process** multiple files using DirectoryLoader
7. Understand Document object structure
### 🦙 Local Offline RAG with Ollama
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
This notebook demonstrates building a **completely offline RAG (Retrieval-Augmented Generation)** system using **Ollama** for local LLMs and embeddings.
#### 🚀 Benefits of Local RAG:
- **100% Offline**: No internet required after setup
- **Privacy First**: Your documents never leave your machine
- **No API Costs**: Free to run unlimited queries
- **Fast**: No network latency
- **Full Control**: Customize models and parameters

## 23 Nov Day - 7 👉 Exploring Langchain: Part-2
### ✂️ Notebook 03: Text Splitting Strategies
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
1. Understand **why** text splitting is necessary for RAG
2. Master **RecursiveCharacterTextSplitter** (the recommended default)
3. Learn other splitters: Character, HTMLHeader, RecursiveJson, Token
4. Choose optimal **chunk sizes** and **overlap**
5. Compare splitters side-by-side
6. Apply the right splitter for different content types
### 🔢 Notebook 04: Embeddings and Vector Representations
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
1. Understand what embeddings are
2. Use OpenAI Embeddings
3. Use Google Gemini Embeddings
4. Compare embedding models
5. Calculate similarity between vectors
📚 By the end of this notebook, you will Learn:
### 🗄️ Notebook 05: Vector Stores
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
1. Understand vector stores
2. Use InMemoryVectorStore (testing)
3. Use FAISS (production)
4. Use Chroma (persistent)
5. Compare vector stores
### 🔍 Notebook 06: Retrieval Strategies
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
1. Create retrievers from vector stores
2. Use similarity search
3. Use MMR (diversity)
4. Custom retrievers with @chain
5. Compare strategies
### 🚀 Notebook 07: Complete RAG Pipeline
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
1. Build a complete RAG application
2. Use LCEL to chain components
3. Create production-ready code
4. Handle errors properly
5. Implement best practices
### 08 - External Index Retrievers 🌐
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
1. **What are External Index Retrievers** and how they differ from vector store retrievers
2. **ArxivRetriever** - Search and retrieve scholarly articles from arxiv.org
3. **WikipediaRetriever** - Access Wikipedia articles for general knowledge
4. **TavilySearchAPIRetriever** - Perform real-time internet searches
5. **Integration with RAG Chains** - Combine external retrievers with LLMs
6. **Best Practices** - When and how to use each retriever effectively

## 29 Nov Day - 8 👉 Exploring Langchain: Part-3
### Vector Stores Tutorial: Qdrant & Weaviate
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
- What vector stores are and why they're essential for RAG (Retrieval-Augmented Generation)
- How to work with **Qdrant** (local, in-memory, and persistent storage)
- How to work with **Weaviate** (Docker-based setup)
- How to use **Ollama embeddings** for real semantic search
- Metadata filtering with different vector store syntaxes
- When to use each vector store based on your use case

### 📚 Notebook 10: RAG Evaluation with Ragas Framework
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
This notebook covers the complete lifecycle of building and evaluating a production-quality RAG system:
1. ✅ **Basic RAG Pipeline** - Document loading, chunking, embeddings, retrieval, generation
2. ✅ **Building RAG Applications** - Prompt engineering, response formatting, handling edge cases
3. ✅ **Evaluation and Testing** - Comprehensive evaluation with Ragas framework

#### Why Evaluation Matters
Building a RAG system is only half the battle. Without proper evaluation, you can't:
- Know if your system is producing accurate answers
- Compare different configuration choices
- Identify areas for improvement
- Ensure production-ready quality

**This notebook introduces the Ragas evaluation framework** - the industry-standard tool for measuring RAG system performance.

#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
- Understand the importance of RAG evaluation
- Implement a complete RAG pipeline from scratch
- Integrate the Ragas evaluation framework
- Create comprehensive test datasets with ground truth
- **Evaluate RAG systems across all 6 key non-multimodal metrics**
- Compare multiple RAG configurations systematically
- Visualize and interpret evaluation results
- Apply production best practices

## 30 Nov Day - 9 Exploring Langchain Part-4 <br>
### 📊 Notebook 12: RAGAS Metrics Deep Dive
**Understanding How RAG Evaluation Metrics Work Internally**
#### 🎯 Learning Objectives
📚 By the end of this notebook, you will Learn:
1. **Understand the internal calculation process** for each of the 6 core RAGAS metrics
2. **See intermediate outputs** like extracted claims, generated questions, and identified entities
3. **Learn to interpret scores with confidence** using threshold guidelines
4. **Debug evaluation issues** by understanding what each metric actually measures

## 04 Dec Day - 10 👉 Exploring RAGAS and LLM as Judge
### 📚 Notebook 11: RAG Evaluation with LLM-as-Judge
**Module 3: RAG Evaluation Techniques**
#### 🎯 Learning Objectives
📚 By completing this notebook, you will Learn:
- Understand the **LLM-as-Judge evaluation methodology**
- Learn when to use LLM-as-Judge vs automated frameworks like Ragas
- Implement a **complete RAG pipeline** (same as Notebook 10)
- Design **structured evaluation prompts** with Pydantic models
- Evaluate RAG systems across **3 core metrics**:
  - 🎯 **Groundedness**: Is the answer faithful to retrieved context?
  - 🎯 **Answer Relevance**: Does the answer address the question?
  - 🎯 **Retrieval Quality**: Are the retrieved contexts relevant?
- Analyze results with **detailed reasoning** and visualizations
- Understand **cost and performance trade-offs**

## 06 Dec Day - 11 Document Parsers for RAG Part-1 
### 3.1 Document Parsing with Docling for RAG Systems
#### 🎯 Learning Objectives
📚 By completing this notebook, you will Learn:
1. **Basic Document Conversion** - Convert PDFs and other formats to Markdown, JSON, HTML
2. **Multiple File Formats** - PDF, DOCX, XLSX, PPTX, HTML, Markdown, Images, Audio
3. **Pipeline Configuration** - OCR engines, table extraction, layout analysis, VLM
4. **LangChain Integration** - DoclingLoader and RAG pipeline with Chroma
5. **Advanced Topics** - Enrichment, error handling

## 07 Dec Day - 12 Document Parsers for RAG Part-2 
### 3.2 Document Parsing with Unstructured for RAG Systems
#### 🎯 Learning Objectives
📚 By completing this notebook, you will Learn:
1. **Core Concepts**: Understanding partition functions, elements, and metadata
2. **Partitioning Strategies**: AUTO, FAST, HI_RES, and OCR_ONLY approaches
3. **File Format Support**: Working with PDFs, Office documents, HTML, Markdown, Images, and more
4. **LangChain Integration**: Seamless integration with LangChain for RAG pipelines
5. **Complete RAG Example**: End-to-end implementation with vector stores

## 13 Dec Day - 13 Document Parsers for RAG Part-3 
### 3.3 Document Parsing with LlamaParse for RAG Systems
#### 🎯 Learning Objectives
- Core Features & Parsing Fundamentals
LlamaParse is a document parsing service by LlamaIndex that excels at:
- **Complex Document Handling**: Financial reports, research papers, scanned PDFs
- **Precise Extraction**: Tables, charts, images, and diagrams
- **70+ File Formats**: PDF, DOCX, XLSX, PPTX, HTML, images, audio, and more
- **LLM-Ready Output**: Clean markdown, text, or structured JSON

#### Part 1 Contents
1. Introduction & Setup
2. Core Concepts
3. Basic Document Parsing
4. Presets & Built-in Configurations
5. Parse Modes (Fast, Premium, Auto)
6. Supported File Formats
7. Multimodal Parsing Features
8. Layout Extraction
9. Structured Output

**Part 2** covers: Custom Prompts, Advanced Configuration, Async Operations, LlamaIndex Integration, RAG Examples, CLI Usage, and Best Practices.

## 14 Dec Day - 14 👉 Poject-1: RAG Q&A System 
### 04 🤖 RAG Q&A System
#### 🎯 Learning Objectives
#### 📖 **Overview**
A **production-ready** RAG (Retrieval-Augmented Generation) system that enables intelligent Q&A over your documents. Built with modern AI stack and battle-tested in production environments.

##### 🎯 **What is RAG?**
RAG combines the power of **retrieval** (finding relevant information) with **generation** (creating coherent answers) to provide accurate, context-aware responses to your questions based on your own documents.

##### 🌟 **Key Highlights**
🚀 Production Ready: Docker + CI/CD + AWS deployment <br>
🧠 Smart AI: Powered by OpenAI GPT-4o & LangChain <br>
📊 Observable: LangSmith integration for full tracing <br>
✅ Evaluated: RAGAS metrics for answer quality <br>
🔒 Secure: Non-root Docker, API validation, error handling <br>
⚡ Fast: Async operations, streaming responses <br>
📈 Scalable: Cloud-native architecture <br>

##### ✨ **Features**
📄 **Document Management**
- ✅ Upload PDF, TXT, and CSV files
- ✅ Automatic text extraction and chunking
- ✅ Smart document splitting with overlap
- ✅ Vector storage in Qdrant Cloud

💬 **Intelligent Q&A**
- ✅ Natural language questions
- ✅ Context-aware answers
- ✅ Source attribution (see which docs were used)
- ✅ Streaming responses for real-time feedback
- ✅ Multiple query modes (standard, search-only)

🔍 **Observability & Quality**
- ✅ LangSmith Tracing: Full chain visibility, token tracking, cost analysis
- ✅ RAGAS Evaluation: Faithfulness & answer relevancy metrics
- ✅ Structured Logging: Comprehensive error tracking
- ✅ Health Checks: Readiness & liveness endpoints

🛠️ **Developer Experience**
- ✅ Auto-generated Swagger docs at /docs
- ✅ Type-safe Pydantic models
- ✅ Comprehensive tests with 70%+ coverage
- ✅ Hot reload in development
- ✅ CI/CD pipeline with GitHub Actions

## 18 Dec Day - 15 👉 Extra Class: Pydantic-FastAPI-Tutorials 
### 05-Pydantic-FastAPI-Tutorials
#### 🎯 Learning Objectives
- Data validation and type safety
- Creating models for complex data structures
- Custom validation logic
- Configuration management
- Building APIs with FastAPI
1. Introduction & Setup
- What is Pydantic? <br>
Pydantic is a data validation library that uses Python type annotations to:
   - ✅ Validate data automatically
   - ✅ Parse and convert data types
   - ✅ Provide clear error messages
   - ✅ Generate JSON schemas
   - ✅ Power frameworks like FastAPI
- Why Use Pydantic? <br>
- Quick Intro to Type Hints
- Installation
  - !uv pip install pydantic
2. Basic Models
3. Field Types & Validation
4. Nested Models & Complex Types
5. Custom Validators

## 20 Dec Day - 16 👉 Poject-1: AWS Deployment 
### 🤖 06-RAG-QA-Project
#### 🎯 Learning Objectives
- rag-project-class project deployment using CI/CD Pipeline with Github Actions
- CI/CD pipeline with GitHub Actions

#### 🤖 RAG Q&A System
✅ rag-qa-project/README.md
- 📖 Overview
- 🎯 What is RAG?
- 🌟 Key Highlights
- ✨ Features
- 🏗️ Architecture
- 🚀 Quick Start
    - Prerequisites
        - 🐍 Python 3.12+
        - 🔑 OpenAI API key
        - 🗄️ Qdrant Cloud account
    - 1️⃣ Clone & Install
    - 2️⃣ Configure Environment
    - 3️⃣ Run Application
    - 4️⃣ Access API
        - 🌐 Swagger UI: http://localhost:8000/docs 
        - 📚 ReDoc: http://localhost:8000/redoc 
        - 🔍 Health Check: http://localhost:8000/health
    - 📝 API Endpoints
        - Document Management
        - Query & Search
        - Health & Monitoring
    - 💡 Usage Examples
        - Upload a Document
        - Ask a Question
        - Query with RAGAS Evaluation
- 🐳 Docker Deployment
    - Using Docker Compose (Recommended)
    - Using Docker Directly
- ☁️ Deployment using AWS
    - AWS App Runner (Serverless)
      Fully automated deployment with GitHub Actions: <br>
        1️⃣ Setup AWS Resources
        2️⃣ Configure GitHub Secrets
        3️⃣ Deploy
        4️⃣ Access Your API
- 🧪 Testing
    - Run All Tests
    - Run Specific Tests
    - Code Quality
- ⚙️ Configuration
    - Environment Variables
- 📊 Project Structure
- 🔄 CI/CD Pipeline
    - Continuous Integration (.github/workflows/ci.yml) <br>
    ✅ Code Quality <br>
    ✅ Testing <br>
    ✅ Docker Build <br>
    ✅ Security <br>
    - Continuous Deployment (.github/workflows/deploy.yml) <br>
    🏗️ Build & Push <br>
    🚀 Deploy <br>
    ✅ Verify <br>
- 🤝 Contributing
    - 🐛 Report Bugs
    - 💡 Suggest Features
    - 🔧 Submit Pull Requests
    - 📜 Development Guidelines

## 21 Dec Day - 17 👉 Llama-Index-Tutorials: Part-1
### 🦙 07-Llama-Index-Tutorials
#### 🎯 Learning Objectives
#### 📚 Notebook 1: Setup & Basics
✅ Understand LlamaIndex architecture and modular ecosystem <br>
✅ Install and configure LlamaIndex with the latest modular packages <br>
✅ Configure the Settings object (LLM, embeddings, chunk size) <br>
✅ Create your first VectorStoreIndex from documents <br>
✅ Execute basic queries and analyze responses <br>
✅ Understand the Document → Node → Index flow <br>

#### 📚 Notebook 2: Documents & Chunking
✅ Load documents from multiple sources (local files, PDFs, web) <br>
✅ Implement different chunking strategies (sentence, token, semantic) <br>
✅ Add custom metadata at document and node levels <br>
✅ Create and manage node relationships <br>
✅ Optimize chunking for retrieval quality <br>
✅ Apply batch embedding optimization <br>

#### 📚 Notebook 3: Indexing & Simple Queries
✅ Integrate external vector stores (Qdrant, Chroma) <br>
✅ Compare embedding models (OpenAI vs HuggingFace) <br>
✅ Persist and load indexes from storage <br>
✅ Configure query engines with different modes <br>
✅ Implement VectorIndexRetriever and VectorIndexAutoRetriever <br>
✅ Understand response synthesis modes <br>
✅ Implement streaming responses <br>

## 03 Jan Day - 18 👉 Llama-Index: Part-2 & 🦛 Chonkie ✨
### 🦙 07-Llama-Index-Tutorials
#### 🎯 Learning Objectives
#### 📚 Notebook 4: Advanced Retrieval
✅ Implement RecursiveRetriever for hierarchical document retrieval  <br>
✅ Use QueryFusionRetriever to combine multiple retrieval strategies  <br>
✅ Build custom retrievers with specialized ranking logic  <br>
✅ Apply query transformation techniques (HyDE, rewriting, multi-query)  <br>
✅ Implement complex metadata filtering (AND/OR/NOT logic)  <br>
✅ Evaluate retrieval quality with metrics  <br>

#### 📚 Notebook 5: Hybrid Search & Reranking
✅ Understand dense vs sparse vectors and BM25 algorithm <br>
✅ Implement hybrid search combining semantic and keyword matching <br>
✅ Configure alpha parameter for optimal score fusion <br>
✅ Apply reranking models (Cohere, SentenceTransformer cross-encoders) <br>

#### 🔍 Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex
- Introduction
- Why Chunk Size Matters
- Setting Up Evaluators
- Response Evaluation For A Chunk Size
- Testing Across Different Chunk Sizes

#### 🦛 Get Started with Chonkie
- 🦛 Chonkie  Documentation ✨
- Installation
- CHONK! 🦛✨

## 04 Jan Day - 19 👉 🦛 Chonkie ✨ Chonkie-Chunkers-Tutorials
### 🦛 08-Chonkie-Chunkers-Tutorials
#### 🎯 Learning Objectives
- **9 Different Chunking Strategies:** From basic token splitting to advanced LLM-powered chunking
- **Google Gemini Integration:** Using Gemini embeddings for semantic chunking
- **Practical Examples:** Real-world applications with technical docs and research papers
- **Performance Comparisons:** Side-by-side analysis of all chunkers
- **Best Practices:** How to choose the right chunker for your use case

#### Tutorial Structure
1. Introduction & Setup
1. Foundation Chunkers - Token, Sentence, Recursive
1. Specialized Chunkers - Table, Code
1. Semantic Chunkers - Semantic, Late, Neural
1. Advanced Chunker - Slumber (LLM-powered)
1. Comparative Analysis
1. Best Practices

## 10 Jan Day - 20 👉 Advanced RAG: Part-1 👉 advanced-rag-tutorials
### 09-Advanced-Rag-Tutorials
#### 🎯 Learning Objectives
#### 📚 Notebook 1: 01_HyDe.ipynb
#### **Hypothetical Document Embedding (HyDE) in Document Retrieval**
#### **Key Components**
1. PDF processing and text chunking
1. Vector store creation using FAISS and OpenAI embeddings
1. Language model for generating hypothetical documents
1. Custom HyDERetriever class implementing the HyDE technique

#### 📚 Notebook 2: 02_fusion_retrieval.ipynb
#### **Fusion Retrieval in Document Search**
#### **Key Components**
1. PDF processing and text chunking
1. Vector store creation using FAISS and OpenAI embeddings
1. BM25 index creation for keyword-based retrieval
1. Custom fusion retrieval function that combines both methods

#### 📚 Notebook 3: adv_sparse_embeddings.ipynb
#### **Advanced Sparse Embeddings**
#### **Key Components**
1. Setup and Installation
1. Dataset and Corpus Preparation <br>
   ✅ Sparse Retrieval Fundamentals <br>
   ✅ BM25 (Statistical Method) <br>
   ✅ SPLADE (Neural Method) <br>
   ✅ Hybrid Search with RRF <br>
   ✅ Score Interpretation <br>
   ✅ Practical Decision Making <br>
1. Cleanup and Next Steps
1. When to Use Each Method
1. Understanding Scores and Metrics
1. Visualizing Score Comparisons

## 11 Jan Day - 21 👉 Advanced RAG: Part-2 👉 advanced-rag-tutorials
### 09-Advanced-Rag-Tutorials
#### 🎯 Learning Objectives
#### 📚 Notebook 4: 03_reranking.ipynb
#### **Reranking Methods in RAG Systems**
#### **Key Components**
1. Initial Retriever: Often a vector store using embedding-based similarity search.
1. Reranking Model: This can be either
   - A Large Language Model (LLM) for scoring relevance
   - A Cross-Encoder model specifically trained for relevance assessment 
1. Scoring Mechanism: A method to assign relevance scores to documents
1. Sorting and Selection Logic: To reorder documents based on new scores


#### 📚 Notebook 5: 04_query_transformations.ipynb
#### **Query Transformations for Improved Retrieval in RAG Systems**
#### **Key Components**
1. Query Rewriting: Reformulates queries to be more specific and detailed.
1. Step-back Prompting: Generates broader queries for better context 1. retrieval.
1. Sub-query Decomposition: Breaks down complex queries into simpler sub-queries.

## 17 Jan Day - 22 👉 Advanced RAG: Part-1 👉 Text-to-SQL with LlamaIndex
### 10-Text2Sql-RAG
#### 🎯 Learning Objectives
#### 📚 Notebook 1: 01_basic_text_to_sql.ipynb
#### **Basic Text-to-SQL with LlamaIndex**
📚 By the end of this notebook, you will Learn:
- Understand text-to-SQL fundamentals
- Use NLSQLTableQueryEngine for structured queries
- Work with SQLite databases
- Inspect and understand generated SQL
- Handle basic error cases
- Apply security best practices

#### 📚 Notebook 2: 02_intermediate_text_to_sql.ipynb
#### **Intermediate Text-to-SQL with Dynamic Table Retrieval**
📚 By the end of this notebook, you will Learn:
- Work with multi-table databases
- Use SQLTableRetrieverQueryEngine for large schemas
- Implement dynamic table retrieval with ObjectIndex
- Understand when to use different query engines
- Query CSV files directly with DuckDB

#### 📚 Notebook 3: 03_advanced_text_to_sql.ipynb
#### **Advanced Text-to-SQL with Full WikiTableQuestions Dataset**
📚 By the end of this notebook, you will Learn:
- Understand LlamaIndex Workflow architecture
- Implement query-time table retrieval workflows
- Add query-time row retrieval with vector indices
- Work with the FULL WikiTableQuestions dataset (2,000+ tables)
- Build production-ready text-to-SQL systems
- Implement error handling and SQL validation

#### 📚 Notebook 4: 04_postgresql_text_to_sql.ipynb
#### **PostgreSQL/Supabase Integration for Text-to-SQL**
📚 By the end of this notebook, you will Learn:
- Connect LlamaIndex to PostgreSQL/Supabase databases
- Implement secure connection patterns
- Query real data with natural language

## 18 Jan Day - 23 👉 Advanced RAG: Part-1 👉 Project-2 👉 multidata-rag-project
## 11-Multidata-RAG-Project
#### 🎯 Learning Objectives
#### **Project-2 Setup & Implementation 👉 Multi-Source RAG + Text-to-SQL** <br>
✅ Python 3.12+ <br>
✅ Configure your .env <br>
✅ Install Packages <br>
✅ OpenAI API Key (for embeddings and LLM) <br>
✅ Vanna AI <br>
✅ Pinecone Account (for vector storage) <br>
   - Create an index with dimension=1536, metric=cosine <br>
   - Pinecone API Key <br>
   
✅ PostgreSQL Database (for Text-to-SQL) <br>
   - Supabase recommended for easy setup
   - Supabase DB Creation

#### 📚 Notebook 1: notebooks/vanna_ai_text_to_sql_complete.ipynb
#### **Text-to-SQL with Vanna.ai - Complete Tutorial**
📚 By the end of this notebook, you will Learn:
1. What Text-to-SQL is and why it matters
1. How to set up Vanna.ai 2.0 with OpenAI and PostgreSQL
1. Understanding the Agent framework architecture
1. Generating SQL from natural language questions
1. Executing queries and handling results
1. Best practices for production deployment

#### **Key Components**
1. Section 1: Introduction to Text-to-SQL
1. Section 2: Environment Setup
1. Section 3: Initialize Vanna 2.0 Agent
1. Section 4: Understanding the Database
1. Section 5: Provide Schema Documentation to Agent
1. Section 6: Querying with the Agent
1. Section 7: Understanding How the Agent Works
1. Section 8: Testing Various Question Types
1. Section 9: Error Handling and Best Practices

#### 📚 Notebook 2: data/generate_sample_data.py
- Sample Data Generation Script
- Generates realistic sample data for the e-commerce database.

#### Notebook 3: supabase_con_test.py
- Check Database connection

## 24 Jan Day - 24 👉 Advanced RAG: Part-2 👉 Project-2 👉 text2sqlrag-project 👉 Code Explanation 👉 Configuration Setup for Project-2 in AWS
## 12-Text2Sql-RAG-Project-2
#### 🎯 Learning Objectives
#### **Project-2 Workflow**
1. System Architecture
    - text2sqlrag-project/workflows/01-system-architecture.md
1. Unified Query Flow
    - text2sqlrag-project/workflows/02-unified-query-flow.md
1. Document Upload Pipeline
    - text2sqlrag-project/workflows/03-document-upload-pipeline.md
1. RAG Query Execution
    - text2sqlrag-project/workflows/04-rag-query-execution.md
1. SQL Query Execution
    - text2sqlrag-project/workflows/05-sql-query-execution.md
1. Multi-Level Cache Architecture
    - text2sqlrag-project/workflows/06-multi-level-cache.md
1. Service Initialization
    - text2sqlrag-project/workflows/07-service-initialization.md

#### **Prerequisites**
1. For Local Development
    - **Python 3.12+**
    - **OpenAI API Key** (for embeddings and LLM)
    - **Pinecone Account** (for vector storage)
        - Create an index with dimension=1536, metric=cosine
    - **PostgreSQL Database** (for Text-to-SQL)
        - Supabase recommended for easy setup
    - **OPIK API Key** (optional, for monitoring)
    - **Upstash Redis** (optional, for query caching - 40-60% cost savings)
1. For AWS Lambda Deployment
    - **AWS Account** with admin access or permissions for ECR, Lambda, IAM, API Gateway
    - **AWS CLI** (version 2.x) configured with credentials
    - **Docker** for building Lambda container images
    - **GitHub Repository** for CI/CD pipeline
    - **Estimated Setup Time:** 30-45 minutes (one-time)

## 25 Jan Day - 25 👉 Advanced RAG: Part-3 👉 Project-2 👉 text2sqlrag-project 👉 Code Explanation
## 12-Text2Sql-RAG-Project-2
#### 🎯 Learning Objectives
#### Multi-Source RAG + Text-to-SQL System
#### **Project-2 Workflow 👉 text2sqlrag-project/workflows/deployment.md**
**AWS Lambda Deployment Architecture - Mermaid Diagrams**
1. CI/CD Deployment Flow
1. Runtime Architecture & Request Flow
1. Cost Breakdown (Monthly Estimate)
1. Key Architecture Benefits
1. Technical Specifications
    - Lambda Configuration
    - Function URL Configuration
    - Docker Image
    - Environment Variables
    - Quick Reference
        - Test Endpoints
        - Monitoring
        - Deployment
#### **Project-2 Deployment 👉 text2sqlrag-project/docs/new_deploy.md**
#### **Complete AWS Deployment Guide for Fresh Account**
- Multi-Source RAG + Text-to-SQL Systems

#### **Project-2 👉 Code Explanation of below python files**
    ✅ text2sqlrag-project/app/config.py
	✅ text2sqlrag-project/app/logging_config.py
	✅ text2sqlrag-project/app/utils.py
	✅ text2sqlrag-project/app/main.py
	✅ text2sqlrag-project/app/services/document_service.py
	✅ text2sqlrag-project/app/services/docling_service.py
	✅ text2sqlrag-project/app/services/embedding_service.py
	✅ text2sqlrag-project/app/services/vector_service.py
	✅ text2sqlrag-project/app/services/s3_storage.py
	✅ text2sqlrag-project/app/services/rag_service.py
	✅ text2sqlrag-project/app/services/local_storage.py

## 31 Jan Day - 26 👉 Advanced RAG: Part-4 👉 Project-2 👉 text2sqlrag-project 👉 AWS Deployment
## 12-Text2Sql-RAG-Project-2
#### 🎯 Learning Objectives
#### Multi-Source RAG + Text-to-SQL System
#### **Project-2 Deployment 👉 text2sqlrag-project/docs/new_deploy.md**
#### **Complete AWS Deployment Guide for Fresh Account**
✅ text2sqlrag-project/docs/new_deploy.md
- Multi-Source RAG + Text-to-SQL Systems
    - Overview
        - What Will Be Deployed
        - Architecture Diagram
        - Prerequisites Checklist
    - Prerequisites & Requirements
        - Install AWS CLI
        - Install Docker (Optional but Recommended)
        - Configure AWS CLI
    - AWS IAM Setup
        - Create Lambda Execution Role
        - Add Deployment Permissions to User
    - AWS Infrastructure Setup
        - Create S3 Bucket for Document Cache
        - Create ECR Repository
        - Build and Push Initial Docker Image
        - Create Lambda Function
        - Create Lambda Function URL
    - External Services Configuration
        - OpenAI Setup
        - Pinecone Setup
        - Supabase/PostgreSQL Setup
        - Upstash Redis Setup (Optional)
        - OPIK Monitoring Setup (Optional)
    - GitHub Repository Setup
        - Fork or Clone Repository
        - Configure GitHub Secrets
        - Verify Workflow Configuration
    -  Database Initialization
        - Connect to Database
        - Run Schema SQL
        - Generate Sample Data (Optional but Recommended)

#### **Project-2 👉 Code Explanation of below python files**
    ✅ text2sqlrag-project/app/services/sql_service.py
	✅ text2sqlrag-project/app/services/query_cache_service.py
	✅ text2sqlrag-project/app/services/router_service.py
	✅ text2sqlrag-project/docs/new_deploy.md
	✅ text2sqlrag-project/.github/workflows/deploy.yml
	✅ text2sqlrag-project/trust-policy.json
	✅ text2sqlrag-project/github-actions-policy.json
	✅ text2sqlrag-project/docker-compose.yml
	✅ text2sqlrag-project/Dockerfile.lambda

## 01 Feb Day - 27 👉 Advanced RAG: Part-5 👉 Project-2 👉 text2sqlrag-project 👉 CI/CD Deployment using AWS
## 12-Text2Sql-RAG-Project-2
#### 🎯 Learning Objectives
#### Multi-Source RAG + Text-to-SQL System
#### **Project-2 👉 Code Explanation of below python files**
    ✅ .github/workflows/deploy.yml
	✅ docs/new_deploy.md

## 07 Feb Day - 28 👉 Advanced RAG: Part-6 👉 Project-2 👉 text2sqlrag-project 👉 Notebook & ECR, lambda funciton and AWS Dployment
## 12-Text2Sql-RAG-Project-2
#### 🎯 Learning Objectives
#### Multi-Source RAG + Text-to-SQL System
#### **Project-2 👉 Code Explanation of below python files**
    ✅ text2sqlrag-project/app/main.py
	✅ text2sqlrag-project/lambda_handler.py
	✅ text2sqlrag-project/DEPLOYMENT_OPTIMIZATION.md
	✅ text2sqlrag-project/.github/workflows/deploy.yml
	✅ text2sqlrag-project/Dockerfile.lambda.base
	✅ text2sqlrag-project/docs/new_deploy.md
	✅ text2sqlrag-project/Dockerfile.lambda
	✅ text2sqlrag-project/build-base-image.sh
	✅ text2sqlrag-project/requirements.txt

#### **Project-2 👉 DEPLOYMENT_OPTIMIZATION.md 👉 AWS Lambda Deployment Optimization Summary**
- 🚀 NEW: Fully Automated Setup for New Users
- 🎯 Results Overview
- 🔄 How Auto-Build Works
    - For New Users (First Deployment)
    - For Existing Users (All Subsequent Deployments)
    - When Base Image Needs Updating
- ✅ Optimizations Implemented
    - Tier 1: High-Impact Changes (25-28 minutes saved)
    - Tier 2: Quick Wins (1-2 minutes saved)
- 📊 Time Breakdown Comparison
    - Before Optimization
    - After Optimization
- 💰 Cost Analysis
    - New Monthly Costs
    - Monthly Savings
    - Net Impact
- 📁 Files Modified
    - New Files Created
    - Modified Files
    - AWS Resources Created
- 🚀 Usage Guide
    - First-Time Setup (New AWS Account)
    - Normal Deployment (Fast)
    - Deployment with S3 Tests
    - Release Deployment (Full Tests)
    - Updating Base Image (When System Dependencies Change)
        - Automatic (Recommended)
        - Manual (Faster for testing)
- 🔍 Verification Steps
    - After First Optimized Deployment
        1. Check GitHub Actions Logs
        1. Verify Base Image in ECR
        1. Check BuildKit Cache
        1. Test Lambda Function
- 🐛 Troubleshooting
    - Issue: Base Image Not Found
    - Issue: BuildKit Cache Not Working
    - Issue: S3 Tests Always Skip
    - Issue: Deployment Fails After Base Image Update
- 🔄 Rollback Plan
    - Rollback Step 1: Restore Original Dockerfile
    - Rollback Step 2: Restore Original Workflow
    - Rollback Step 3: Clean Up (Optional)
- 🌟 Benefits for New Users & Organizations
    - Zero-Configuration Deployment
    - Perfect for CI/CD
    - Cost-Effective for Multiple Environments
    - Developer Experience
- 🎉 Success Metrics <br>
        ✅ Deployment time: 32 min → 5 min (84% faster) <br>
        ✅ Docker build stage: 28 min → 2 min (93% faster) <br>
        ✅ GitHub Actions cost: -$21/month savings <br>
        ✅ ECR storage cost: +$0.40/month (minimal) <br>
        ✅ Net monthly savings: +$21/month <br>
        ✅ Developer velocity: 6-7x faster iteration <br>

## 08 Feb Day - 29 👉 Advanced RAG: 👉 advanced-rag-tutorials 👉 Query Transformations, CRAG, SRAG
## 09-Advanced-Rag-Tutorials
#### 🎯 Learning Objectives
#### 📚 Advanced RAG Tutorials
#### **Code Explanation of below python files**
    ✅ advanced-rag-tutorials/04_query_transformations.ipynb
    ✅ advanced-rag-tutorials/crag.ipynb
    ✅ advanced-rag-tutorials/self_rag.ipynb
    ✅ CRAG(Corrective RAG) arXiv Paper 👉 https://arxiv.org/pdf/2401.15884
    ✅ SRAG(Self RAG) arXiv Paper 👉 https://arxiv.org/pdf/2310.11511

#### **Query Transformations for Improved Retrieval in RAG Systems**
- **Query Rewriting:** Reformulates queries to be more specific and detailed.
- **Step-back Prompting:** Generates broader queries for better context retrieval.
- **Sub-query Decomposition:** Breaks down complex queries into simpler sub-queries.

#### **Corrective RAG Process:** Retrieval-Augmented Generation with Dynamic Correction
***Benefits of the Corrective RAG Approach***
- **Dynamic Correction:** Adapts to the quality of retrieved information, ensuring relevance and accuracy.
- **Flexibility:** Leverages both pre-existing knowledge and web search as needed.
- **Accuracy:** Evaluates the relevance of information before using it, ensuring high-quality responses.
- **Transparency:** Provides source information, allowing users to verify the origin of the information.
- **Efficiency:** Uses vector search for quick retrieval from large knowledge bases.
- **Contextual Understanding:** Combines multiple sources of information when necessary to provide comprehensive responses.
- **Up-to-date Information:** Can supplement or replace outdated local knowledge with current web information.

#### **Self-RAG:** A Dynamic Approach to Retrieval-Augmented Generation
***Benefits of the Approach***
- **Dynamic Retrieval:** By deciding whether retrieval is necessary, the system can adapt to different types of queries efficiently.
- **Relevance Filtering:** The relevance evaluation step ensures that only pertinent information is used, reducing noise in the generation process.
- **Quality Assurance:** The support assessment and utility evaluation provide a way to gauge the quality of generated responses.
- **Flexibility:** The system can generate responses with or without retrieval, adapting to the available information.
- **Improved Accuracy:** By grounding responses in relevant retrieved information and assessing their support, the system can produce more accurate outputs.

## 14 Feb Day - 30 👉 Project-3 👉 Part-1 👉 corrective-self-reflective-rag
## 13-Corrective-Self-Reflective-RAG
#### 🎯 Learning Objectives
#### 🔮 Corrective + Self-Reflective RAG
#### **Code Explanation of below python files**
	✅ corrective_self_reflective_rag/workflows/project_architecture.md
	✅ corrective_self_reflective_rag/app/config.py
	✅ corrective_self_reflective_rag/workflows/crag_mode.md
	✅ corrective_self_reflective_rag/workflows/self_reflective_mode.md
	✅ corrective_self_reflective_rag/workflows/both_mode.md

## 15 Feb Day - 31 👉 Project-3 👉 Part-2 👉 corrective-self-reflective-rag
## 13-Corrective-Self-Reflective-RAG
#### 🎯 Learning Objectives
#### 🔮 Corrective + Self-Reflective RAG
#### **Code Explanation of below python files**
	✅ corrective_self_reflective_rag/workflows/hybrid_search.md
	✅ corrective_self_reflective_rag/app/services/document_processor.py
	✅ corrective_self_reflective_rag/app/services/vector_store.py
	✅ corrective_self_reflective_rag/app/services/sparse_vector_service.py
	✅ corrective_self_reflective_rag/app/services/embedding_service.py
	✅ corrective_self_reflective_rag/app/api/upload.py
	✅ corrective_self_reflective_rag/app/api/query.py
    ✅ corrective_self_reflective_rag/app/core/retrieval.py	
	✅ corrective_self_reflective_rag/app/models.py
	✅ corrective_self_reflective_rag/app/services/hyde.py
	✅ corrective_self_reflective_rag/app/services/web_search.py
	✅ corrective_self_reflective_rag/app/services/llm_service.py

## 21 Feb Day - 32 👉 Project-3 👉 Part-3 👉 corrective-self-reflective-rag  👉 Docker mode + AWS Deployment Using BeanStalk
## 14-Corrective-Self-Reflective-RAG-Deployment-AWS-Project-3
#### 🎯 Learning Objectives
#### 🔮 Corrective + Self-Reflective RAG
#### **Code Explanation of below python files**
	✅ corrective_self_reflective_rag/Dockerfile
	✅ corrective_self_reflective_rag/deployment.md
	✅ corrective_self_reflective_rag/Dockerrun.aws.json
	✅ corrective_self_reflective_rag/pyproject.toml

### AWS Elastic Beanstalk Deployment Guide
#### **corrective_self_reflective_rag/deployment.md**
- Architecture Overview
- Prerequisites
    - Phase 1 — Build & Push Image to ECR
    - Phase 2 — Grant EC2 Instance ECR Pull Access
    - Phase 3 — Update Dockerrun.aws.json
    - Phase 4 — Initialize Beanstalk Application
    - Phase 5 — Create the Beanstalk Environment
    - Phase 6 — Set Environment Variables (API Keys)
    - Phase 7 — Deploy
- Verify the Deployment
    - End-to-end verification checklist
    - Re-deploy After Code Changes
- Cost Estimate
- Stop / Teardown AWS Services
- Troubleshooting
- Key Design Notes
- Related Files 

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the container image |
| `.dockerignore` | Excludes `.venv`, secrets, tests from the build context |
| `Dockerrun.aws.json` | Tells Beanstalk which ECR image to pull and run |
| `app/config.py` | `pydantic-settings` — reads all config from environment variables |
| `pyproject.toml` + `uv.lock` | Required in build context for `uv sync --frozen` |
| `workflows/aws_eb_deployment.md` | Mermaid architecture and deployment flow diagrams |

## 22 Feb Day - 33 👉 Project-3 👉 Part-4 👉 corrective-self-reflective-rag 👉 Complete CI/CD Pipeline Deployment
## 15-Corrective-Self-Reflective-RAG-Deploy-AWS-EB-CICD Pipeline-Project-3
#### 🎯 Learning Objectives
#### 🔮 Corrective + Self-Reflective RAG + CI/CD Pipeline Deployment
#### **Code Explanation of below python files**
	✅ corrective_self_reflective_rag/AWS_Deployment.md
	✅ corrective_self_reflective_rag/AWS_Deployment_Architecture.md
	✅ corrective_self_reflective_rag/.github/workflows/deploy.yml

### AWS Elastic Beanstalk Deployment Guide
#### **corrective_self_reflective_rag-deploy-aws-eb/AWS_Deployment.md**
- Infrastructure Summary
- Repository Structure (Deployment-Relevant Files)
- CI/CD Workflow
    - Steps
        - Checkout
        - Configure AWS Credentials
        -  Login to Amazon ECR
        - Build and Push Docker Image
        - Package EB Bundle
        - Upload Bundle to S3
        - Create EB Application Version
        - Update EB Environment
        - Poll for Readiness
- GitHub Secrets
    - All secrets are configured under Repository **Settings → Secrets and variables → Actions**.
- IAM Permissions Required
- Environment Variables on the EC2 Instance
- Dockerrun.aws.json
- Health Check
- Deployment Version Naming Convention
- Known Issues & Fixes Applied

### AWS Deployment Architecture
#### **corrective_self_reflective_rag-deploy-aws-eb/AWS_Deployment_Architecture.md**
- CI/CD Pipeline Flow
- GitHub Secrets Flow
- EB Bundle Structure
- Deployment Sequence

## 28 Feb Day - 34 👉 Ultimate RAG 👉 Graph RAG
## 16-Graph-RAG
#### 🎯 Learning Objectives
#### 🔮 GraphRAG with LangChain + Neo4j
#### **Code Explanation of below python files**
	✅ graph-rag/neo4j_browser_guide.md
	✅ graph-rag/Getting_Started_GraphRAG.ipynb

### Neo4j & Cypher: Step-by-Step Browser Guide
#### **graph-rag/neo4j_browser_guide.md**
1. What is Neo4j?
1. Opening Neo4j Browser
1. Step 1 — Verify Connection
1. Step 2 — CREATE Nodes
    1. Create a Single Person Node
    1. Create Multiple Nodes in One Query
    1. Create a Movie Node
    1. See the Full Graph So Far
1. Step 3 — CREATE Relationships
    1. Relationship Syntax
    1. Link Alice to The Matrix
    1. Create a KNOWS Relationship Between People
    1. Visualize the Full Graph
1. Step 4 — MATCH & Query
    1. Return All Nodes
    1. Filter by Label and Property
    1. Traverse a Relationship
    1. ORDER BY and LIMIT
    1. Multi-Hop Traversal (Friends-of-Friends Pattern)
    1. Find Relationship Types Dynamically
1. Step 5 — MERGE (Upsert)
    1.  MERGE with ON CREATE / ON MATCH
    1. MERGE a Relationship Safely
1. Step 6 — UPDATE Properties (SET)
    1. Update Node Properties
    1. Add a Property to a Relationship
1. Step 7 — DELETE Nodes and Relationships
    1. Delete a Single Relationship
    1. Delete a Node (DETACH DELETE)
    1. Clear the Entire Database (Demo Reset)

## 07 Mar Day - 35 👉 Ultimate RAG 👉 Graph RAG 👉 Ontologies + Vector GraphRAG
## 17-Vector-Graph-RAG
#### 🎯 Learning Objectives
#### 🔮 Hybrid RAG: Vector Search + Knowledge Graph
#### **Code Explanation of below python files**
    ✅ graph-rag/graphrag_ontologies.md
	✅ graph-rag/graphrag_langchain.ipynb
    ✅ vector-graph-rag/main.ipynb
	✅ vector-graph-rag/config/llm.py
	✅ vector-graph-rag/config/neo4j.py
	✅ vector-graph-rag/config/pinecone_cfg.py
	✅ vector-graph-rag/ingest/neo4j.py
	✅ vector-graph-rag/ingest/pinecone_ingest.py
	✅ vector-graph-rag/retrieve/neo4j_pinecone.py
	✅ vector-graph-rag/Falkordb_Graphiti_Demo26.ipynb

### GraphRAG Ontologies & Knowledge GraphRAG
#### **graph-rag/graphrag_ontologies.md**
- 📊 USE CASE 1 — Financial Documents
    - 1.1 — Document Ontology (Financial)
        - Node & Edge Type Registry
    - 1.2 — Attribute Schemas (Financial)
    - 1.3 — Knowledge Graph Build Pipeline (Financial)
- 📱 USE CASE 2 — Social Media
    - 2.1 — Document Ontology (Social Media)
        - Node & Edge Type Registry
    - 2.2 — Attribute Schemas (Social Media)
    - 2.3 — Knowledge Graph Build Pipeline (Social Media)
- 🔁 Cross-Use-Case Comparison

### Hybrid RAG: Vector Search + Knowledge Graph
#### **vector-graph-rag/README.md**
- What is RAG?
- Why Hybrid? The Limitation of Standard RAG
- Architecture
- The Jupyter Notebook (vector-graph-rag/main.ipynb)
    - Step 1 — Connect to Databases
    - Step 2 — Define Data Sources
    - Step 3 — Ingest into Neo4j: Create Nodes
    - Step 4 — Create Relationships in the Graph
    - Step 5 — Ingest into Pinecone: Vector Embeddings
    - Step 6 — Define the Graph Traversal Query
    - Step 7 — Run Hybrid Search
- Project Structure
- Why This Approach Is Valuable
    - For learners
    - For practitioners
    - For researchers
- Prerequisites
    - Setup
        1. Install dependencies
        1. Create a .env file in the project root
        1. Open the notebook
- Technology Stack
    - Pinecone : Vector database — stores and searches embeddings
    - Neo4j : Graph database — stores entities and their relationships
    - OpenAI : Embedding model (`text-embedding-3-small`)
    - LangChain : Neo4j graph client and text splitting utilities
    - uv : Python package and project manager

#### **Discussed below arXiv Paper of GrapRAG**
- [When to use Graphs in RAG](https://arxiv.org/pdf/2506.05690v1) : A Comprehensive Analysis for Graph Retrieval-Augmented Generation
- [KG-RAG](https://arxiv.org/pdf/2405.12035) : Bridging the Gap Between Knowledge and Creativity
- [From Local to Global](https://arxiv.org/pdf/2404.16130) : A GraphRAG Approach to Query-Focused Summarization
- [OG-RAG](https://arxiv.org/pdf/2412.15235): ONTOLOGY-GROUNDED RETRIEVAL-AUGMENTED GENERATION FOR LARGE LANGUAGE MODELS

## 08 Mar Day - 36 👉 Ultimate RAG 👉 Haystack-AI-Tutorials
## 18-Heystack-AI-Tutorials
#### 🎯 Learning Objectives
- Fundamentals
    - RAG architecture and core components
    - Document stores and retrieval mechanisms
    - Prompt engineering for RAG
    - Building basic Q&A pipelines
- Advanced Retrieval
    - Dense embeddings and semantic search
    - Hybrid retrieval (BM25 + vector search)
    - Query expansion techniques
    - Reranking strategies for precision
- Agentic & Adaptive RAG
    - Conditional routing and decision-making
    - Web search fallback mechanisms
    - Self-correcting RAG pipelines
    - Conversational RAG with memory
- Specialized Techniques
    - Corrective RAG (CRAG) patterns
    - GraphRAG with knowledge graphs
    - Long-context document handling
    - Multimodal RAG (vision + text)

#### 🔮 Haystack AI RAG Tutorials
#### **Code Explanation of below python files**
    ✅ haystack-ai-tutorials/README.md
	✅ haystack-ai-tutorials/01_rag_fundamentals.ipynb
	✅ haystack-ai-tutorials/02_advanced_retrieval.ipynb
	✅ haystack-ai-tutorials/03_agentic_rag.ipynb
	✅ haystack-ai-tutorials/04_specialized_techniques.ipynb

### Haystack AI RAG Tutorials
#### **Notebooks**
1. **01_rag_fundamentals.ipynb** - RAG Fundamentals & Basic Implementation
    - Core RAG concepts and architecture
    - Building your first RAG pipeline
    - Prompt engineering basics
    - Simple Q&A systems
1. **02_advanced_retrieval.ipynb** - Advanced Retrieval Techniques
    - Semantic search with embeddings
    - Hybrid retrieval strategies
    - Query expansion
    - Reranking for improved precision
1. **03_agentic_rag.ipynb** - Advanced RAG Patterns & Agentic Behavior
    - Conditional routing
    - Web search fallback
    - Self-correcting pipelines
    - Conversational RAG with memory
1. **04_specialized_techniques.ipynb** - Specialized RAG Techniques
    - Corrective RAG (CRAG)
    - GraphRAG with Neo4j
    - Long-context handling
    - Multimodal RAG

#### **haystack-ai-tutorials/README.md**
- Prerequisites
    - System Requirements
        - **Python:** 3.9-3.12
        - **RAM:** 8GB minimum (16GB recommended for local embeddings)
        - **Disk Space:** 10GB+ for datasets and models
        - **Internet:** Required for API calls and dataset downloads
        - **Optional:** CUDA-capable GPU for local model inference
    - Knowledge Prerequisites
        - Intermediate Python programming
        - Basic understanding of machine learning
        - Familiarity with LLMs and embeddings
        - Basic command line usage
- Setup Instructions
1. Clone or Download this Repository
1. Create a Virtual Environment
1. Install Dependencies
1. Configure Environment Variables
1. (Optional) Setup Neo4j for Notebook 4
    - **Option A: Docker (Recommended)**
    - **Option B: Desktop Installation** Download from: https://neo4j.com/download/
1. Launch JupyterLab

## 15 Mar Day - 37 👉 Ultimate RAG 👉 Multimodal RAG 👉 Project-4 Discussion & Lightning AI
## Lightning AI 👉 [click here](https://lightning.ai/)
#### 🎯 Learning Objectives
- Project-4 Discussion using Multimodal RAG
- Sign up Lightning AI
- Lightning AI Complete setup 
- Demo `multi_model_rag_with_colpali.ipynb` in Lightning AI

## 22 Mar Day - 38 👉 Ultimate RAG 👉 Paradigm 👉 Colpali & (Layout Detection + OCR) 👉 MinerU, Mistral and Nemotron Parse OCR
## Colpali Research Paper 👉 [click here](https://arxiv.org/pdf/2407.01449)
#### 🎯 Learning Objectives
#### **Code Explanation of below python files**
    ✅ Multimodal_Rag_with_Colpali.ipynb
    ✅ Docling_Granite_Parsing.ipynb
    ✅ Mineru_test.ipynb
    ✅ mineru_multimodal_rag.ipynb

#### Paradigm A: Colpali
#### **Explained Colpali Research Paper in brief**
1. INTRODUCTION
1. PROBLEM FORMULATION & RELATED WORK
1. THE ViDoRe BENCHMARK
1. LATE INTERACTION BASED VISION RETRIEVAL
1. RESULTS
1. ABLATION STUDY
1. CONCLUSIONS

#### Paradigm B: MinerU, Mistral and Nemotron Parse OCR
1. MinerU Implementation
1. Mistral Implementation
1. Nemotron Implementation

## 28 Mar Day - 39 👉 Ultimate RAG 👉 Multimodal RAG 👉 🧠 MultiModal RAG Pipeline Implementation using 🦙 Ollama (Locally) and Lightning AI(GUP), JinaAI
## A complete pipeline for Multi-Modal RAG with GLM OCR with Ollama support.
## 19-Multi-Modal-RAG
#### 🎯 Learning Objectives
- Local Setup
- `multi-modal-rag` master branch
#### **Code Explanation of below python files**
    ✅ multi-modal-rag/TESTING.md
    ✅ multi-modal-rag/README.md
#### 🧠 MultiModal RAG Pipeline 
- ✨ What Is This?
- 🚀 Key Features
- 📋 Table of Contents
    - 🛠️ Prerequisites
    - ⚡ Quick Start
    - 🖥️ CLI Reference
        - multi-modal-rag/scripts/parse.py — Parse PDFs into Markdown + JSON
        - multi-modal-rag/scripts/ingest.py — Embed + Upsert to Qdrant
        - multi-modal-rag/scripts/search.py — Query + Re-rank
    - 🌐 REST API
        - Interactive docs at http://localhost:8000/docs
            - POST /search
            - POST /ingest
    - 🔌 Embedding Providers
    - 🎯 Re-ranker Backends
        - openai (default)
        - Jina
        - bge
        - qwen
    - 🦙 Local Mode (Ollama)
        - Setup
        - Enable in .env
        - Parse directly from the Streamlit visualizer
        - Cloud vs Local
    - 🎨 Visual Inspector
        - Cloud API inspector
        - Ollama inspector
    - 📦 Output Formats
        - 📝 Markdown output (document.md)
        - 🗂️ JSON output (document.json)
        - 🧩 Chunks output (document_chunks.json)
    - ⚙️ Configuration Reference
    - 🗂️ Project Structure
    - 🧪 Running Tests
    - 🔧 Development
    - 🔍 Troubleshooting
        - ModuleNotFoundError: No module named 'glmocr'
        - ValidationError: Z_AI_API_KEY is required when PARSER_BACKEND=cloud
        - ValueError: JINA_API_KEY must be set
        - ImportError: BGE / Qwen / Gemini reranker
        - Qdrant connection refused
        - Ollama parse fails: Connection refused
        - Bounding boxes appear misaligned in Streamlit
    - 🏗️ Tech Stack

## 29 Mar Day - 40 👉 Ultimate RAG 👉 Multimodal RAG 👉 🧠 MultiModal RAG Pipeline Implementation Using Deployment Branch
## A complete pipeline for Multi-Modal RAG with GLM OCR with Ollama support.
## 20-Multi-Modal-RAG-Deployment
#### 🎯 Learning Objectives
- `Lightning AI` GPU Deployment Guide
- `multi-modal-rag` deployment branch
- `multi-modal-rag/workflows` master branch
##### **Code Explanation of below python files**
    ✅ multi-modal-rag/workflows/01-system-overview.md
    ✅ multi-modal-rag/workflows/02-full-rag-pipeline.md
    ✅ multi-modal-rag/workflows/03-ingestion-pipeline.md
    ✅ multi-modal-rag/workflows/04-retrieval-pipeline.md
    ✅ multi-modal-rag/workflows/05-data-structures.md
    ✅ multi-modal-rag/workflows/06-parsing-pipeline.md
    ✅ multi-modal-rag/workflows/11-reranking-backends.md

## 05 Apr Day - 41 👉 Ultimate RAG 👉 Multimodal RAG 👉 🧠 MultiModal RAG Pipeline Implementation Using Qwen Branch 👉 Lightning AI GPU Deployment
## A complete pipeline for Multi-Modal RAG with GLM OCR with Ollama support.
## 21-Multi-Modal-RAG-Qwen-Model
#### 🎯 Learning Objectives
- `Lightning AI` GPU Deployment Guide
- `multi-modal-rag` deployment branch
- `multi-modal-rag` qwen branch
#### **Code Explanation of below python files**
- `multi-modal-rag` deployment branch <br>
    ✅ multi-modal-rag/LIGHTNING_AI_DEPLOY.md
- `multi-modal-rag` qwen brach <br>
    ✅ multi-modal-rag/Qwen_testing.md

#### Lightning AI — GPU Deployment Guide (LIGHTNING_AI_DEPLOY.md)
- Step 1 — Open a Terminal in Lightning AI
- Step 2 — Verify GPU is Available
- Step 3 — Clone the Repository
- Step 4 — Create the .env File
- Step 5 — Pull the GLM-OCR Model into Ollama
- Step 6 — Build and Start the Full Stack
- Step 7 — Verify All Services are Running
- Step 8 — Ingest a PDF
- Step 9 — Search
- Step 10 — Generate (Full RAG Answer)

#### Qwen Branch — Testing Guide (Qwen_testing.md)
- Two Testing Modes
    - Mode A — FastAPI Direct
    - Mode B — Full Docker Compose
- Stack Overview
1. API Keys Required
1. Ollama Setup — Pull the GLM-OCR Model
1. Environment Setup
    - Confirm branch
    - Install dependencies
    - Create .env
    - Start Infrastructure Services (required for both modes)
1. Unit Tests (no API keys needed — ~3–4 s)
1. Integration Tests (require infrastructure running)
1. Manual Pipeline Testing (step by step)
    - Step 1 — Parse only (verify GLM-OCR output)
    - Step 2 — Full ingestion (parse → caption → embed → upsert)
    - Step 3 — Search + rerank
1. FastAPI Server Testing
    - Mode A — FastAPI Direct (app runs on host)
        - Step 1 — Ensure infrastructure is running
        - Step 2 — Warm up models (run once before starting the server)
        - Step 3 — Start the FastAPI server
        - Step 4 — Model loading sequence on first request
        - Step 5 — Health check
        - Step 6 — Ingest a PDF
        - Step 7 — Search
        - Step 9 — List collections
    - Mode B — Full Docker Compose (everything in Docker)
        - Step 1 — Start the full stack
        - Step 2 — Pull GLM-OCR model (first time only)
        - Step 3 — Watch the warm-up sequence
        - Step 4 — Verify the stack
        - Step 5 — Test via API (same curl commands as Mode A)
1. GPU / Model Loading Verification
1. Linting
1. Troubleshooting
1. Key Config Defaults (qwen branch)
1. Local vLLM Mode (No RunPod Required)
    - GPU Requirements
    - .env Settings for Local vLLM
    - Starting the Stack
    - Watch vLLM Start Up
    - Verify vLLM from the Host
    - Troubleshooting
1. Running Everything on Lightning.ai (Full Local GPU)
    - Choose the Right GPU Instance
    - Open a Terminal on the Instance
    - Verify Prerequisites
    - Clone the Repository and Switch to qwen Branch
    - Create and Fill .env
    - Start the Full Stack
        - Pull the GLM-OCR Model into Ollama (first time only)
    - Wait for vLLM to Load (First Run Only)
    - Verify the Full Stack is Healthy
    - Run the Pipeline
        - Option A — CLI scripts (recommended for learning and debugging)
        - Option B — REST API
    - What to Watch for in Logs
    - Port Reference
    - Key Reminders
    - Troubleshooting (Lightning.ai specific)

## 11 Apr Day - 42 👉 Ultimate RAG 👉 Multimodal RAG 👉 🧠 MultiModal RAG Pipeline Implementation Using Qwen/Qwen3-VL-Embedding-2B 👉 Lightning AI GPU Deployment
## A complete pipeline for Multi-Modal RAG with GLM OCR with Ollama support.
## 22-Multi-Modal-RAG-Qwen3-VL-Embedding-Model
#### 🎯 Learning Objectives
    ✅ multi-modal-rag-Qwen3-VL-Embedding-2B/docker-compose.yml
    ✅ multi-modal-rag-Qwen3-VL-Embedding-2B/Docker_Compose_Testing.md
    ✅ multi-modal-rag-Qwen3-VL-Embedding-2B/ollama/config.docker.yaml
    ✅ multi-modal-rag-Qwen3-VL-Embedding-2B/src/doc_parser/ingestion/image_captioner.py
    ✅ multi-modal-rag-Qwen3-VL-Embedding-2B/src/doc_parser/retrieval/reranker.py

#### Docker Compose Testing Guide — Local GPU Stack (Docker_Compose_Testing.md)
1. Prerequisites
1. Configure `.env`
1. Build and start everything
1. Confirm all containers are running
1. Per-service health checks
    - Qdrant
    - Ollama (GLM-OCR)
    - vLLM (Qwen3-VL-4B-Instruct-AWQ)
    - App (FastAPI)
1. GPU verification
    - Host view
    - Inside the app container
1. Inspect logs
    - Per-service log commands
    - What to look for during a healthy startup
        - **vLLM** (`logs vllm`)
        - **Ollama** (`logs ollama`)
        - **App** (`logs app`)
        - **App on first** (`/ingest/file`)
1. End-to-end smoke test
    - Ingest a PDF
    - Verify chunks contain real text
    - Generate an answer
1. Common operations
    - Restart one service (no rebuild)
    - Force-recreate a service after editing `.env` or compose file
    - Rebuild after editing source code or Dockerfile
    - Stop everything (containers persist, can be restarted)
    - Tear down everything (containers gone, volumes preserved)
    - Tear down INCLUDING volumes (wipes Qdrant + HF cache + Ollama models)
    - Open a shell inside the app container
    - Inspect the embedder/reranker on GPU mid-run
1. Troubleshooting
    - Symptom: `nvidia-smi` shows GPU but `cuda available:` False in app
    - Symptom: `/generate` returns `404 — model X does not exist`
    - Symptom: vLLM crashes at startup with OOM
    - Symptom: Chunks contain `"##"` or `"[figure]"` placeholders, no real text
    - Symptom: Image embedding fails with `argument of type 'NoneType' is not iterable`
    - Symptom: Reranker fails with `Unrecognized configuration class … for AutoModelForSequenceClassification`
    - Symptom: Qdrant version skew warning in app logs
    - Symptom: Ollama `ollama ps` shows `100% CPU` instead of GPU
1. Quick reference — full happy-path command sequence

## 16 Apr Day - 43 👉 Ultimate RAG 👉 Extra Class 👉 Metadata Enrichment in RAG Pipelines 👉 metadata-hybrid-rag, langextract-rag and semantic_highlight_hhem_rag
## Metadata Enrichment in RAG Pipelines
## 23-Metadata-Enrichment-RAG
- 23.1-MetaData-Hybrid-RAG/metadata-hybrid-rag
- 23.2-LangExtract-Enhanced/langextract-rag
- 23.3-Semantic-Highlight-HHEM-RAG/semantic-highlight-hhem-rag
### 🎯 Learning Objectives
 - GLINER2 👉 HYBRID RAG with Metadata Enrichment
 - LANGEXTRACT 👉 LangExtract-Enhanced RAG System
 - SEMANTIC HIGHLIGHTING & HALLUCINATION DETECTION
#### 23.1-MetaData-Hybrid-RAG/metadata-hybrid-rag
#### HYBRID RAG with Metadata Enrichment
Production-ready RAG system with metadata enrichment using GLiNER2, Qdrant, and OpenAI.
##### 🌟 Key Features
- Multi-format document upload (PDF, Markdown, TXT, JSON)
- Docling HybridChunker for intelligent semantic chunking
- GLiNER2 for zero-shot metadata extraction (entities, domain, content type, tech specs)
- Qdrant vector store with hybrid search (BM25 + dense embeddings)
- OpenAI GPT-4o-mini for answer generation
- Metadata filtering by domain, content type, and entities

##### 📓 Notebooks
##### `metadata-hybrid-rag/notebooks/metadata_enrichment_tutorial.ipynb` <br>
End-to-end tutorial comparing RAG without metadata vs with GLiNER2-enriched metadata.
- How to design metadata for any RAG system (4 categories, decision framework)
- Baseline ingestion (baseline_rag collection) — structural fields only
- Enriched ingestion (enriched_rag collection) — full GLiNER2 pipeline with timing
- Domain, content-type, entity, and combined filter demos
- Side-by-side retrieval quality comparison across 4 benchmark queries
- Ingestion and retrieval timing tables

> **Requirements:** OpenAI API key, Qdrant running (`docker-compose up -d`)

##### `metadata-hybrid-rag/notebooks/gliner2_complete_features.ipynb`
- Comprehensive showcase of all GLiNER2 capabilities — no Qdrant or OpenAI key needed.
- Basic NER (list labels vs. descriptions)
- Single-label and multi-label text classification
- Confidence scores with filtering
- Structured extraction (basic fields, choices/enum, per-field thresholds)
- RegexValidator (full match, partial match, exclude modes)
- Multi-task combined extraction in one forward pass
- Batch processing (batch_extract, batch_classify_text) with timing
- Schema Builder API (full HR/CV pipeline)
- Schema caching best practices
- Base vs Large model comparison
- End-to-end news intelligence pipeline

> **Requirements:** `pip install gliner` only (models download on first run ~500MB)

#### 23.2-LangExtract-Enhanced/langextract-rag
#### LangExtract-Enhanced RAG System
##### 🌟 Key Features
- 🧠 LangExtract Integration: Automatically extracts structured metadata (topic, category, entities, version, summary) from documents
- 🔍 Metadata-Filtered Retrieval: Filter search results by document type, version, entities, and more
- 🤖 Self-Querying Retriever: LLM automatically generates metadata filters from natural language queries
- 📄 PDF Processing: Load and chunk PDF documents with intelligent text splitting
- 💾 Pinecone Vector Store: Scalable vector storage with metadata filtering
- 🚀 Multiple Retrieval Strategies: Basic, filtered, scored, and self-query retrieval

##### 📓 Notebooks
##### `langextract-rag/notebooks/metadata_enrichment_tutorial.ipynb`
- Metadata Enrichment in RAG Pipelines
- From Sparse Chunks to Precision Retrieval with LangExtract + LangChain
- Audience: Advanced — you know RAG, embeddings, and LangChain basics.
- This notebook focuses entirely on the metadata enrichment layer and how it transforms retrieval quality.
- **What you will build:** <br>
    Raw text chunks (2 metadata fields) <br>
        → LangExtract enrichment  (LLM extracts structured metadata per chunk) <br>
        → Rich documents (10+ metadata fields) <br>
        → 4 retrieval strategies  (basic → filtered → scored → self-query) <br>
        → Measurable improvement in retrieval precision + timing data <br>

> **Before you start:** Only `OPENAI_API_KEY` is required. No Pinecone needed — we use Chroma (in-memory) throughout.

#### 23.3-Semantic-Highlight-HHEM-RAG/semantic-highlight-hhem-rag
#### Semantic Highlighting + HHEM RAG
##### 🌟 Key Features
- Upload documents (PDF, MD, TXT, JSON)
- Semantic Highlighting for context pruning
- HHEM validation for hallucination detection
- Comparison endpoint (with vs without optimizations)
- Metrics tracking and cost analysis
##### 📓 Notebooks
##### `semantic-highlight-hhem-rag/notebooks/rag_showcase.ipynb`
- **RAG Optimization:** Semantic Highlighting & Hallucination Detection
- **Semantic Highlighting:** Standard RAG sends ALL retrieved text to the LLM, including irrelevant sentences
- **HHEM Validation:** LLMs sometimes generate confident-sounding but wrong answers
- How This Notebook Is Organised
    - Section 1  →  Setup (imports + services)
    - Section 2  →  Sample corpus
    - Section 3  →  PART 1 — Semantic Highlighting
    - Section 4  →  PART 2 — Hallucination Detection (HHEM)
    - Section 5  →  Combined summary visualisation
    - Section 6  →  End-to-end pipeline (optional, needs Qdrant)
    - Section 7  →  Student exercises

> **Before you start:** Set the environment variable `OPENAI_API_KEY` in your shell or in a `.env` file at the project root. The semantic and HHEM sections do not require an OpenAI key — only Section 6 does.


## 18 Apr Day - 44 👉 Ultimate RAG 👉 Multimodal RAG 👉 Preoject-4 👉 AWS Deployment 👉 multi-modal-rag
## A complete pipeline for Multi-Modal RAG with GLM OCR with Ollama support.
## 24-Multi-Modal-RAG-Deployment-Using-AWS
- AWS Architecture Explainer — Multi-Modal RAG Pipeline
- multi-modal-rag - Step-by-Step AWS Deployment Guide

#### 🎯 Learning Objectives
#### `multi-modal-rag/docs/AWS_ARCHITECTURE_EXPLAINER.md`
##### AWS Architecture Explainer — Multi-Modal RAG Pipeline
1. What We Built — The Big Picture
1. The Full Architecture Diagram (Text)
1. Every AWS Service Used — and Why <br>
    3.1 Amazon ECS Fargate <br>
    3.2 Application Load Balancer (ALB) <br>
    3.3 Amazon EFS (Elastic File System) <br>
    3.4 Amazon ECR (Elastic Container Registry) <br>
    3.5 AWS Secrets Manager <br>
    3.6 AWS CloudWatch Logs <br>
    3.7 IAM Roles <br>
1. Networking Deep Dive — VPC, Subnets, Security Groups <br>
    4.1 What is a VPC? <br>
    4.2 What We Used (Default VPC) <br>
    4.3 What a Custom VPC Would Look Like (Enterprise) <br>
    4.4 Security Groups — The Virtual Firewall <br>
    4.5 awsvpc Network Mode <br>
1. The Multi-Container Sidecar Pattern
1. Storage Strategy — Why EFS Over EBS or S3
1. Security Design
1. Is This Enterprise-Grade? <br>
    - Current state: Production-ready for low-to-medium traffic <br>
1. Rough Monthly Cost Estimate
    - Fixed costs (running 24/7)
    - Variable costs (usage-dependent)
    - Rough total scenarios
    - Cost optimization levers
1. Future Enhancements — The Production Roadmap <br>
    - Phase 1: Security Hardening (Immediate) <br>
        10.1 HTTPS with a Custom Domain <br>
        10.2 API Authentication <br>
        10.3 Custom VPC with Private Subnets <br>
    - Phase 2: Reliability and Availability <br>
        10.4 Multi-AZ Deployment <br>
        10.5 Auto Scaling <br>
        10.6 Circuit Breaker on ECS Service <br>
    - Phase 3: Performance <br>
        10.7 GPU-Enabled Instances for Ollama <br>
        10.8 Model Caching — Pre-warm on Startup <br>
        10.9 Async Ingestion with SQS <br>
    - Phase 4: Observability <br>
        10.10 CloudWatch Dashboard <br>
        10.11 Distributed Tracing with AWS X-Ray <br>
        10.12 Alerting <br>
    - Phase 5: Data and ML Enhancements <br>
        10.13 S3 for Document Storage <br>
        10.14 Qdrant Collection Versioning <br>
        10.15 Fine-Tuned Embedding Model <br>
        10.16 Streaming Responses <br>
    - Phase 6: CI/CD and GitOps <br>
        10.17 GitHub Actions Pipeline <br>
        10.18 Infrastructure as Code (Terraform or AWS CDK) <br>
1. Key Architectural Trade-offs Made
1. Glossary for Students

#### `multi-modal-rag/docs/STEP_BY_STEP_DEPLOY.md`
##### Step-by-Step AWS Deployment Guide
1. Prerequisites
    1.1 AWS CLI v2
    1.2 Docker
    1.3 jq
    1.4 GitHub CLI (for setting secrets later)
1. IAM — Create Admin User
    2.1 Create the user
    2.2 Attach permissions
    2.3 Generate access keys
    2.4 Configure AWS CLI
    2.5 Verify
    2.6 Export profile for the session
1. Shell Variables
    3.1 Retrieve your default VPC and subnets
    3.2 Export all variables
    3.3 Verify
1. Security Groups
    Two security groups are needed: <br>
    - **ALB SG** — faces the internet, accepts port 80
    - **ECS SG** — faces the ALB only, accepts port 8000
1. ECR Repositories
1. ECS Cluster
1. EFS — Persistent Storage
    EFS provides two persistent volumes that survive deployments: <br>
    - `/qdrant/storage` — Qdrant vector database data
    - `/root/.ollama` — Ollama model weights (downloaded once, reused forever)
    > Save FS_ID, QDRANT_AP, and OLLAMA_AP — required for task definitions in Phase 12.
1. Secrets Manager
1. IAM — CI/CD Bot User
1. IAM — ECS Task Execution Role
1. CloudWatch Log Groups
1. ECS Task Definitions <br>
    12.1 App Task Definition <br>
        - **app** — FastAPI backend (port 8000)
        - **qdrant** — vector database sidecar (port 6333, EFS-backed)
        - **ollama** — local LLM / OCR engine (port 11434, EFS-backed, essential: true)
1. Application Load Balancer
1. ECS Services
1. Ollama Model Bootstrap
1. GitHub Actions Secrets
    - The CI/CD pipeline needs these secrets set in GitHub: **Repository → Settings → Secrets and variables → Actions**
1. Verify Deployment
1. Troubleshooting Common Deployment Issues
    - A — Task fails to start: AccessDeniedException on Secrets Manager
    - B — ALB health checks timing out: `Target.Timeout`
    - C — Qdrant NFS warning on EFS (not a fatal error)
1. CI/CD Flow Reference
1. Rollback Procedure
1. Cost Overview — What This Infrastructure Charges Per Month
1. How to Stop the Infrastructure (Save Money, Keep Data)
    - Step 1 — Scale the ECS service to zero tasks
    - Step 2 — Delete the ALB listener and ALB (stops the $16/month fixed charge)
1. How to Restart the Infrastructure
    - Step 1 — Recreate the ALB
    - Step 2 — Set the ALB idle timeout to 300 seconds
    - Step 3 — Recreate the listener pointing to the existing target group
    - Step 4 — Scale the ECS service back up
    - Step 5 — Get the new ALB DNS name and verify
1. How to Tear Down Everything (Full Deletion)
    - Step 1 — Stop and delete the ECS service
    - Step 2 — Delete the ALB, listener, and target group
    - Step 3 — Delete the EFS filesystem (permanent data loss)
    - Step 4 — Delete the ECS cluster
    - Step 5 — Delete ECR repositories and images
    - Step 6 — Delete CloudWatch log group
    - Step 7 — Delete Secrets Manager secret
    - Step 8 — Delete IAM role and policies
    - Step 9 — Delete Security Groups
    - Step 10 — Delete IAM users (optional)
    - Step 11 — Verify everything is gone

---

## Create the virtual environment in anaconda3 folder
```
conda create -n venv_ur_name python==3.12 -y
conda activate venv_ur_name
conda deactivate
```

## Delete the virtual environment using conda
```
conda remove -n venv_ur_name --all
```

## Create the virtual environment in current folder
```
conda create -p venv_ur_name python==3.12 -y
conda activate
conda deactivate
```

## Create a virtual environment in Python with Conda 
Refer 👉 https://gist.github.com/loic-nazaries/b18a908473935243fc23586f35d4bacc

Resources
---
[Ultimate RAG Course 👉 Notion](https://krishnaikacademy.notion.site/Ultimate-RAG-Course-2a5eba9593d08085ade8ceb2a6c6c8de) <br>

[Ultimate RAG Course 👉 Class Notes & Additional Materials](https://krishnaikacademy.notion.site/Ultimate-RAG-Course-2a5eba9593d08085ade8ceb2a6c6c8de) <br>

[Ultimate RAG Document Hub 👉 Class Notes](https://krishnaikacademy.notion.site/ultimateragcourse?v=2a5eba9593d080a885b6000c6a7b8358) <br>

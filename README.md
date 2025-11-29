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

All Classes
---
02 Nov Day - 1 Induction Session <br>
08 Nov Day - 2 Tools And Installation <br>
09 Nov Day - 3 Introduction To RAG <br>
15 Nov Day - 4 RAG Architectures & Survey <br>
16 Nov Day - 5 Exploring Langchain For RAG <br>
22 Nov Day - 6 Exploring Langchain Part-1 <br>
23 Nov Day - 7 Exploring Langchain Part-2
# ✂️ Notebook 03: Text Splitting Strategies
By the end of this notebook, you will:
1. Understand **why** text splitting is necessary for RAG
2. Master **RecursiveCharacterTextSplitter** (the recommended default)
3. Learn other splitters: Character, HTMLHeader, RecursiveJson, Token
4. Choose optimal **chunk sizes** and **overlap**
5. Compare splitters side-by-side
6. Apply the right splitter for different content types
- Embeddings and Vector Representations
    - What are Embeddings? 🤔
    - OpenAI Embeddings
    - Google Gemini Embeddings
    - Similarity Calculation
- Vector Stores
    -  FAISS (Fast & Production-Ready)
    - Chroma (Persistent & Easy)
- Retrieval Strategies
A Retriever is an interface that returns relevant documents for a query.
    - Similarity Search Retriever
    - MMR (Maximum Marginal Relevance)
    - Custom Retriever with chain
- 🚀 Notebook 07: Complete RAG Pipeline


29 Nov Day - 8 Exploring Langchain Part-3 <br>
- Vector Stores Tutorial: Qdrant & Weaviate
    - 🗄️ Part 1: Qdrant Vector Store
    - 🗄️ Part 2: Weaviate Vector Store
- RAG Evaluation with Ragas Framework 

30 Nov Day - 9 Exploring Langchain Part-4 <br>

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
[Ultimate RAG Course 👉 Software Checklist](https://krishnaikacademy.notion.site/Software-Checklist-2a5eba9593d08048927ed6fbd00f502d) <br>

[Ultimate RAG Course 👉 Class Notes & Additional Materials](https://krishnaikacademy.notion.site/Ultimate-RAG-Course-2a5eba9593d08085ade8ceb2a6c6c8de) <br>

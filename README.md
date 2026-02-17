# Retrieval-Augmented Generation (RAG)

This repository documents my end-to-end learning and implementation journey of Retrieval-Augmented Generation (RAG) systems — starting from foundational ingestion pipelines to advanced hybrid retrieval, ranking, and multi-query optimization techniques.

**Goal:** Build production-style RAG architectures with modular components and measurable improvements in retrieval quality.

---

## What I Learned & Implemented

### 1. Foundations of RAG

- Understanding the RAG architecture (Retriever + Generator)
- Difference between parametric vs non-parametric memory
- Building a basic ingestion → retrieval → generation pipeline
- One-off retrieval with LLM-based answer generation

#### Files
- ingestion_pipeline.py
- retrieval_pipeline.py
- answer_generation.py

---

### 2. Text Chunking Strategies

- Recursive Character Text Splitting
- Semantic Chunking
- Agentic Chunking strategies
- Chunk size & overlap experimentation

---

### 3. Vector Search & Embeddings

- Generating dense embeddings
- Storing vectors in a vector database
- Similarity search (cosine similarity / dot product)
- Understanding recall vs precision trade-offs

---

### 4. Retrieval Techniques

- Basic similarity retrieval
- Metadata filtering
- Multi-query retrieval
- History-aware retrieval
- Retrieval method comparison

#### Files
- retrieval_methods.py
- multi_query_retrieval.py
- history_aware_generation.py

---

### 5. Advanced Retrieval Optimization

- Hybrid search (dense + sparse retrieval)
- Multi-query query reformulation
- Cross-encoder reranking
- Reciprocal Rank Fusion (RRF)
- Multimodal retrieval integration

---

### 6. Next Steps

- Add evaluation metrics (Recall@K, MRR, NDCG)
- Deploy as API service
- Add monitoring & observability
- Introduce streaming generation
- Experiment with smaller language models (SLMs)

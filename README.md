# Advanced RAG System

Production-oriented Retrieval-Augmented Generation system.

## Current Version

v1 - Normal RAG

## Architecture

Documents
→ Chunking
→ Embeddings
→ FAISS
→ Similarity Search
→ Context
→ LLM
→ Answer

## Features

- PDF document ingestion
- Recursive text chunking
- Semantic vector search
- FAISS vector store
- LLM-based answer generation
- Grounded prompting

## Tech Stack

Python
LangChain
FAISS
Sentence Transformers
OpenAI
FastAPI (planned)

## Roadmap

[x] Normal RAG
[ ] Hybrid Search
[ ] BM25
[ ] Reciprocal Rank Fusion
[ ] Cross-Encoder Reranking
[ ] Strict Citation Enforcement
[ ] RAG Evaluation
[ ] FastAPI API
[ ] Docker
[ ] Production Deployment

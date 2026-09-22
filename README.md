# Advanced RAG System

A modular, production-oriented **Retrieval-Augmented Generation (RAG)** system built with **hybrid retrieval, reciprocal rank fusion, cross-encoder reranking, structured citations, citation validation, NLI-based entailment verification, evaluation, and monitoring**.

The system is designed to answer questions from a controlled document collection while minimizing unsupported or hallucinated responses.

---

## 🚀 Features

* 📄 PDF document ingestion
* ✂️ Configurable document chunking
* 🧠 Semantic search using FAISS
* 🔎 Keyword search using BM25
* 🔀 Hybrid retrieval using Reciprocal Rank Fusion (RRF)
* 🎯 Cross-encoder reranking
* 🤖 Gemini Flash-based answer generation
* 📌 Structured claims and citations
* ✅ Citation ID validation
* 🧠 NLI-based citation entailment verification
* 🔄 Regeneration when citation validation fails
* 📊 Retrieval and generation evaluation framework
* 📈 Basic production monitoring
* 🧩 Modular architecture

---

# 🏗️ Architecture

```text
                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │  RAG Pipeline   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Hybrid Retriever│
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ FAISS Search │          │  BM25 Search │
        │ Dense Top-K  │          │  Keyword Top-K│
        └──────┬───────┘          └──────┬───────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                    ┌──────────────┐
                    │  RRF Fusion  │
                    └──────┬───────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Cross-Encoder    │
                  │    Reranker      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Top-K Context    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Gemini Flash LLM │
                  └────────┬─────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Structured Response │
                │ Answer + Claims +    │
                │ Citations            │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Citation Validator   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ NLI Entailment Check │
                └──────────┬───────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                  Valid         Invalid
                    │             │
                    ▼             ▼
                 Answer       Regenerate
```

---

# 🔄 RAG Pipeline

## 1. Document Ingestion

PDF documents are loaded from:

```text
data/documents/
```

Each document is converted into structured text while preserving useful metadata such as:

* Source
* Page number
* Chunk ID

---

## 2. Chunking

Documents are divided into smaller chunks for efficient retrieval.

Current configuration:

```text
Chunk size:    800
Chunk overlap: 150
```

Each chunk receives a unique identifier such as:

```text
chunk_18
chunk_24
chunk_31
```

These IDs are later used for citation validation.

---

## 3. Embeddings

Semantic embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are stored in a FAISS vector index.

---

## 4. Hybrid Retrieval

The system combines two retrieval strategies.

### Dense Retrieval

FAISS performs semantic similarity search.

```text
Query
 ↓
Embedding
 ↓
FAISS
 ↓
Top 10 results
```

### Sparse Retrieval

BM25 performs keyword-based retrieval.

```text
Query
 ↓
BM25
 ↓
Top 10 results
```

---

## 5. Reciprocal Rank Fusion

The FAISS and BM25 results are combined using **Reciprocal Rank Fusion (RRF)**.

Current configuration:

```text
DENSE_TOP_K = 10
BM25_TOP_K  = 10
RRF_K       = 60
```

This combines semantic and lexical retrieval signals before reranking.

---

## 6. Cross-Encoder Reranking

The fused results are reranked using:

```text
BAAI/bge-reranker-base
```

The highest-ranked documents are passed to the generation stage.

Current configuration:

```text
RERANK_TOP_K = 5
```

This reduces irrelevant context being sent to the LLM.

---

# 🤖 Generation

The system uses Google's Gemini Flash model for answer generation.

Current model:

```text
gemini-3.8-flash
```

The LLM is instructed to:

* Use only retrieved context
* Avoid unsupported information
* Generate concise answers
* Identify factual claims
* Attach citations to claims
* Never invent citation IDs
* Return a fallback response when the context is insufficient

---

# 📌 Structured Citations

The LLM produces structured claims:

```json
{
  "claim": "PM-KISAN provides financial support to eligible farmer families.",
  "citations": ["chunk_27"]
}
```

The system then verifies that the cited chunk actually exists in the retrieved results.

---

# ✅ Citation Validation

The citation validator checks:

### 1. Citation existence

Is the cited chunk ID actually present in the retrieved context?

### 2. Citation coverage

Does every factual claim have at least one citation?

### 3. Citation support

Does the cited evidence actually support the claim?

Invalid responses are rejected rather than directly returned to the user.

---

# 🧠 NLI Entailment Verification

Citation support is verified using:

```text
cross-encoder/nli-deberta-v3-base
```

The model evaluates:

```text
Evidence
   +
Claim
   ↓
Entailment
Contradiction
Neutral
```

A claim is considered supported only when the entailment score exceeds both contradiction and neutral scores.

This provides an additional verification layer beyond simple citation ID matching.

---

# 🔄 Validation & Regeneration

The generation pipeline follows:

```text
Generate
   ↓
Validate citations
   ↓
NLI entailment
   ↓
Valid?
 ┌─┴─┐
Yes  No
 │    │
 ▼    ▼
Return  Regenerate
       │
       ▼
    Retry limit
```

This prevents unsupported answers from being returned as valid responses.

---

# 📊 Evaluation

The project includes an evaluation framework under:

```text
evaluation/
```

The evaluation dataset contains questions based on the indexed PM-AWAS and PM-KISAN documents.

The system evaluates:

### Retrieval Metrics

* Recall@5
* Precision@5
* Mean Reciprocal Rank (MRR)

### Generation / Grounding Metrics

* Citation accuracy
* Citation coverage
* Faithfulness
* Citation validation status

Run evaluation with:

```bash
python evaluation/run_evaluation.py
```

Detailed results are written to:

```text
evaluation/results.json
```

> Gemini API quota availability can affect the execution of the generation-based evaluation.

---

# 📈 Monitoring

The project includes basic request monitoring.

The monitoring layer tracks:

* Total requests
* Request latency
* Number of retrieved chunks
* Number of generated claims
* Validation status
* Validation errors

Example summary:

```text
============================================================
RAG MONITORING SUMMARY
============================================================
Total requests       : 10
Average latency      : 2.41 sec
Validation success   : 90.00%
============================================================
```

---

# 📁 Project Structure

```text
advanced-rag-system/
│
├── data/
│   └── documents/
│       ├── PM_AWAS.pdf
│       └── PM_KISAN.pdf
│
├── evaluation/
│   ├── questions.json
│   ├── evaluator.py
│   ├── evaluation.py
│   └── run_evaluation.py
│
├── scripts/
│   ├── build_index.py
│   └── view_chunks.py
│
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py
│       ├── ingestion.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── vectorstore.py
│       ├── bm25.py
│       ├── fusion.py
│       ├── reranker.py
│       ├── retriever.py
│       ├── generator.py
│       ├── citation.py
│       ├── entailment.py
│       ├── validator.py
│       ├── metrics.py
│       ├── evaluation.py
│       ├── monitoring.py
│       └── pipeline.py
│
├── vectorstore/
│
├── tests/
│
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

| Category         | Technology                      |
| ---------------- | ------------------------------- |
| Language         | Python                          |
| LLM              | Gemini 3.8 Flash                |
| Framework        | LangChain                       |
| Embeddings       | all-MiniLM-L6-v2                |
| Vector Database  | FAISS                           |
| Sparse Retrieval | BM25                            |
| Fusion           | Reciprocal Rank Fusion          |
| Reranking        | BAAI/bge-reranker-base          |
| Entailment       | NLI DeBERTa v3                  |
| Data Validation  | Pydantic                        |
| Environment      | python-dotenv                   |
| Evaluation       | Custom RAG evaluation framework |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/chaitu280/advanced-rag-system.git
cd advanced-rag-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Configuration

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
LLM_MODEL=gemini-3.8-flash
TEMPERATURE=0
```

Never commit `.env` to GitHub.

---

# 📚 Build the Index

Place your PDF documents inside:

```text
data/documents/
```

Then run:

```bash
python scripts/build_index.py
```

This creates the local retrieval indexes used by the RAG pipeline.

---

# ▶️ Run the RAG System

Start the application:

```bash
python main.py
```

Example:

```text
You: What is the annual financial benefit provided under PM-KISAN?
```

The system retrieves relevant evidence, generates the answer, validates citations, performs entailment verification, and returns the grounded response.

---

# 🔬 RAG Evolution

This project was developed incrementally:

### V1 — Basic RAG

```text
Documents → Embeddings → FAISS → LLM → Answer
```

### V2 — Hybrid Search

```text
FAISS + BM25 → RRF → LLM
```

### V3 — Reranking

```text
FAISS + BM25
      ↓
     RRF
      ↓
Cross-Encoder
      ↓
     LLM
```

### V4 — Citation Verification

```text
RAG
 ↓
Claims + Citations
 ↓
Citation Validation
 ↓
NLI Entailment
```

### V5 — Evaluation & Monitoring

```text
RAG
 ↓
Validation
 ↓
Evaluation Metrics
 ↓
Monitoring
```

---

# 🎯 Project Goals

The primary goals of this project are:

* Improve retrieval quality using hybrid search
* Reduce irrelevant context using reranking
* Reduce unsupported LLM responses
* Make generated claims traceable to source chunks
* Automatically verify citation support
* Measure retrieval and generation quality
* Provide a modular architecture suitable for experimentation and further development

---

# 🔮 Future Improvements

Potential future improvements include:

* Query rewriting
* Multi-query retrieval
* Parent-child retrieval
* Contextual compression
* Advanced evaluation with LLM-as-a-judge
* Streaming responses
* REST API using FastAPI
* Docker deployment
* Cloud deployment
* Advanced observability
* Automated evaluation pipelines

---

# 👨‍💻 Author

**Manne Lakshmi Chaitanya**

AI Engineer | Generative AI | RAG | Machine Learning

GitHub: [@chaitu280](https://github.com/chaitu280)

---

## ⭐ Project Summary

This project demonstrates a complete advanced RAG workflow:

```text
Ingestion
   ↓
Chunking
   ↓
Embeddings
   ↓
Hybrid Retrieval
   ↓
RRF Fusion
   ↓
Cross-Encoder Reranking
   ↓
Gemini Generation
   ↓
Structured Claims
   ↓
Citation Validation
   ↓
NLI Entailment
   ↓
Evaluation
   ↓
Monitoring
```

The emphasis is on **retrieval quality, grounding, citation reliability, and measurable RAG performance**, rather than simply connecting an LLM to a vector database.

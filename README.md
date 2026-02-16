# Multi-Agentic RAG AI Research Scientist

An **agent-driven RAG system** that dynamically searches, retrieves, and synthesizes research papers from arXiv on-demand. Built with **Groq API** for fast LLM inference.

## 🎯 Key Features

- **Multi-Agentic Architecture**: 7 specialized agents work together in a pipeline
- **Dynamic Retrieval**: Papers fetched on-demand, not pre-indexed
- **Grounded Synthesis**: All claims verified against evidence
- **Hallucination Detection**: LLM-based safety verification
- **Production-Ready**: FastAPI backend + Streamlit frontend
- **Comprehensive Evaluation**: Automated benchmark suite

## 🏗️ System Architecture

```
┌─────────────────────┐
│     User Query      │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Planner Agent     │ ← Query decomposition & search strategy
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│    Search Agent     │ ← arXiv API integration
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Ingestion Agent   │ ← PDF download & text extraction
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Ephemeral Vector DB │ ← FAISS (session-scoped)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Retrieval Agent   │ ← Dense semantic retrieval
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Reranking Agent   │ ← LLM-based relevance scoring
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Reasoning Agent    │ ← Multi-document synthesis
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Safety Agent      │ ← Citation & hallucination checks
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Final Response    │ ← Structured, cited answer
└─────────────────────┘
```
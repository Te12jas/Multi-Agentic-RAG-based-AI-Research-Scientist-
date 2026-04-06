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

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd "Agentic RAG AI Research Scientist"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_key_here
```

Get your free Groq API key at: https://console.groq.com

### 3. Run the Application

**Option A: Streamlit Frontend (Recommended)**
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501 in your browser.

**Option B: FastAPI Backend**
```bash
uvicorn api:app --reload --port 8000
```
API docs at http://localhost:8000/docs

## 📁 Project Structure

```
Agentic RAG AI Research Scientist/
├── agents/                  # Agent modules
│   ├── planner.py          # Query decomposition
│   ├── search.py           # arXiv API integration
│   ├── ingestion.py        # PDF processing
│   ├── retrieval.py        # Dense retrieval
│   ├── rerank.py           # LLM reranking
│   ├── reasoning.py        # Multi-doc synthesis
│   └── safety.py           # Hallucination detection
├── utils/                   # Infrastructure
│   ├── groq_client.py      # Groq API wrapper
│   ├── embeddings.py       # Sentence embeddings
│   ├── vector_store.py     # FAISS vector store
│   ├── pdf_utils.py        # PDF extraction
│   └── monitoring.py       # SQLite/Supabase logging
├── prompts/                 # Agent prompts
│   ├── planner_prompt.txt
│   ├── rerank_prompt.txt
│   ├── reasoning_prompt.txt
│   └── safety_prompt.txt
├── evaluation/              # Benchmark suite
│   ├── test_queries.json   # 15 test queries
│   ├── expected_papers.json# Ground truth
│   └── run_eval.py         # Evaluation runner
├── orchestrator.py          # Pipeline coordinator
├── api.py                   # FastAPI backend
├── streamlit_app.py         # Streamlit frontend
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── .env.example             # Environment template
```

## 🤖 Agent Details

| Agent | Model | Purpose |
|-------|-------|---------|
| **Planner** | meta-llama/llama-4-scout-17b-16e-instruct | Query analysis & search strategy |
| **Search** | - (arXiv API) | Paper discovery & metadata |
| **Ingestion** | - (PyMuPDF) | PDF download & chunking |
| **Retrieval** | all-MiniLM-L6-v2 | Dense semantic search |
| **Rerank** | meta-llama/llama-4-scout-17b-16e-instruct | Relevance scoring |
| **Reasoning** | llama-3.3-70b-versatile | Multi-doc synthesis |
| **Safety** | meta-llama/llama-4-scout-17b-16e-instruct | Grounding verification |

## 📊 Evaluation

Run the benchmark suite:

```bash
python evaluation/run_eval.py
```

With options:
```bash
python evaluation/run_eval.py --limit 5  # Run first 5 queries
python evaluation/run_eval.py --output results.json
```

### Metrics Computed

**Retrieval Metrics:**
- Recall@K
- Mean Reciprocal Rank (MRR)
- Context Precision

**Reasoning Metrics (LLM-as-Judge):**
- Faithfulness Score (0-1)
- Unsupported Claims Count
- Hallucination Flag
- Synthesis Score (1-5)
- Comparison Score (1-5)
- Limitation Awareness Score (1-5)

## 📈 Monitoring

Set `DEV_MODE=true` in `.env` to enable the monitoring dashboard in Streamlit.

### Storage Backends

The monitoring system supports two backends:

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| **SQLite** (default) | Local development | No config needed |
| **Supabase** | Cloud deployment | Set `SUPABASE_URL` + `SUPABASE_KEY` |

> **Note**: SQLite data is lost on Streamlit Cloud restarts. Use Supabase for persistent cloud monitoring.

### Supabase Setup (for Cloud Deployment)

1. Create free project at https://supabase.com
2. Run this SQL in Supabase SQL Editor:
```sql
CREATE TABLE query_logs (
    id SERIAL PRIMARY KEY,
    query_id TEXT UNIQUE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    query_text TEXT,
    papers_fetched INTEGER,
    chunks_indexed INTEGER,
    chunks_retrieved INTEGER,
    retrieval_latency_ms FLOAT,
    total_latency_ms FLOAT,
    groq_tokens_used INTEGER,
    agent_calls JSONB,
    hallucination_flag BOOLEAN,
    success BOOLEAN,
    error_message TEXT
);
```
3. Add `SUPABASE_URL` and `SUPABASE_KEY` to your `.env`

### Logged Metrics

- Query text & ID
- Papers fetched
- Chunks indexed/retrieved
- Retrieval & total latency
- Groq token usage
- Hallucination flag
- Agent call counts

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `FAST_MODEL` | meta-llama/llama-4-scout-17b-16e-instruct | Fast model for planning/reranking |
| `REASONING_MODEL` | llama-3.3-70b-versatile | Large model for synthesis |
| `DEFAULT_PAPERS_K` | 5 | Papers to retrieve per query |
| `DEFAULT_CHUNKS_TOP_N` | 10 | Chunks for retrieval |
| `DEV_MODE` | false | Show monitoring dashboard |
| `SUPABASE_URL` | (optional) | Supabase project URL |
| `SUPABASE_KEY` | (optional) | Supabase anon/service key |

## 📝 Output Format

Every research response includes:

```markdown
## Summary
[Key findings overview]

## Key Contributions
[Bullet points per paper]

## Comparative Analysis
[Cross-paper comparison]

## Limitations
[Acknowledged limitations]

## Practical Implications
[Real-world applications]

## Citations
[Full paper references]
```

## ⚠️ Design Decisions

1. **Groq-Only**: Fast inference, free tier, no local model downloads
2. **Ephemeral Vector Store**: Session-scoped FAISS, no persistent indexing
3. **Section-Aware Chunking**: Extracts Abstract/Methods/Experiments/Limitations
4. **LLM Reranking**: Reduces noise before synthesis
5. **Safety Layer**: Explicit hallucination detection and refusal logic

## 🔒 Limitations

- Requires internet for arXiv API access
- PDF extraction may miss some formatting
- Rate limited by Groq free tier
- Only arXiv papers (no other sources)

## 📄 License

MIT License
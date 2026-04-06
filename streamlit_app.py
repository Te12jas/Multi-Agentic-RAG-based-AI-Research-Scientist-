"""
Streamlit frontend for the Agentic RAG AI Research Scientist.
Provides UI for research queries and monitoring dashboard.
"""

import streamlit as st
import requests
import json
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config

# Page Configuration
st.set_page_config(
    page_title="AI Research Scientist",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #a0a0a0 !important;
        margin-bottom: 2rem !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .paper-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🔬 AI Research Scientist</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Agentic RAG-powered research paper synthesis using Groq</p>',
        unsafe_allow_html=True
    )


def render_research_tab():
    """Render the Research tab."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Research Query")
        
        query = st.text_area(
            "Enter your research question",
            placeholder="e.g., What are the key innovations in vision transformers compared to CNNs?",
            height=100,
            label_visibility="collapsed"
        )
        
        with st.expander("⚙️ Advanced Settings"):
            papers_k = st.slider(
                "Number of papers to retrieve",
                min_value=1,
                max_value=10,
                value=config.DEFAULT_PAPERS_K
            )
            chunks_n = st.slider(
                "Number of chunks for retrieval",
                min_value=5,
                max_value=20,
                value=config.DEFAULT_CHUNKS_TOP_N
            )
            safety_check = st.checkbox("Enable safety verification", value=True)
        
        if st.button("🔍 Research", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a research question.")
                return
            
            if not config.GROQ_API_KEY:
                st.error("⚠️ GROQ_API_KEY is not configured. Please set it in your .env file.")
                return
            
            with st.spinner("🔄 Researching... This may take a minute."):
                try:
                    # Import and run orchestrator directly
                    from orchestrator import orchestrator
                    
                    result = orchestrator.run(
                        query=query,
                        papers_k=papers_k,
                        chunks_top_n=chunks_n,
                        include_safety_check=safety_check
                    )
                    
                    st.session_state.current_result = result
                    st.session_state.query_history.append({
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "success": result.get("success", False)
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return
    
    with col2:
        st.subheader("📚 Query History")
        if st.session_state.query_history:
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                status = "✅" if item["success"] else "❌"
                st.markdown(f"**{status}** {item['query'][:50]}...")
        else:
            st.info("No queries yet. Start by asking a research question!")
    
    # Display Results
    if st.session_state.current_result:
        st.divider()
        render_results(st.session_state.current_result)


def render_results(result: dict):
    """Render research results."""
    st.subheader("📊 Research Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = result.get("metrics", {})
    
    with col1:
        st.metric("Papers Found", len(result.get("papers", [])))
    with col2:
        st.metric("Chunks Used", result.get("chunks_used", 0))
    with col3:
        latency = metrics.get("total_latency_ms", 0)
        st.metric("Latency", f"{latency/1000:.1f}s")
    with col4:
        tokens = metrics.get("tokens_used", 0)
        st.metric("Tokens Used", f"{tokens:,}")
    
    # Hallucination warning
    if result.get("hallucination_detected"):
        st.warning("⚠️ Some claims may not be fully grounded in the retrieved evidence.")
    
    # Main response
    st.markdown("### 📝 Synthesis")
    st.markdown(result.get("response", "No response generated."))
    
    # Papers used
    papers = result.get("papers", [])
    if papers:
        st.markdown("### 📚 Papers Retrieved")
        for paper in papers:
            with st.expander(f"📄 {paper.get('title', 'Unknown Title')}"):
                st.markdown(f"**Authors:** {', '.join(paper.get('authors', [])[:5])}")
                st.markdown(f"**Year:** {paper.get('year', 'Unknown')}")
                st.markdown(f"**arXiv ID:** {paper.get('paper_id', 'Unknown')}")
                st.markdown(f"**Abstract:** {paper.get('abstract', 'No abstract')[:500]}...")
                if paper.get("arxiv_url"):
                    st.markdown(f"[View on arXiv]({paper['arxiv_url']})")
    
    # Search plan details
    plan = result.get("plan", {})
    if plan:
        with st.expander("🎯 Search Plan Details"):
            st.json(plan)


def render_monitoring_tab():
    """Render the Monitoring tab (DEV_MODE only)."""
    if not config.DEV_MODE:
        st.info("🔒 Monitoring is only available in DEV_MODE. Set DEV_MODE=true in .env to enable.")
        return
    
    st.subheader("📈 System Monitoring")
    
    try:
        from orchestrator import orchestrator
        
        # Get stats
        stats = orchestrator.get_monitoring_stats()
        
        # Show backend type
        backend_type = stats.get("backend", "sqlite")
        if backend_type == "supabase":
            st.success("☁️ **Monitoring Backend:** Supabase (persistent cloud storage)")
        else:
            st.info("💾 **Monitoring Backend:** SQLite (local storage - data may be lost on restart)")
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", stats.get("total_queries", 0))
        with col2:
            st.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0)/1000:.1f}s")
        with col3:
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
        with col4:
            st.metric("Hallucination Rate", f"{stats.get('hallucination_rate', 0):.1f}%")
        
        st.divider()
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Papers Fetched", f"{stats.get('avg_papers_fetched', 0):.1f}")
        with col2:
            st.metric("Avg Chunks Indexed", f"{stats.get('avg_chunks_indexed', 0):.1f}")
        with col3:
            st.metric("Total Tokens Used", f"{stats.get('total_tokens_used', 0):,}")
        
        st.divider()
        
        # Recent logs
        st.subheader("📋 Recent Query Logs")
        
        logs = orchestrator.get_recent_logs(limit=10)
        
        if logs:
            for log in logs:
                status = "✅" if log.get("success") else "❌"
                halluc = "⚠️" if log.get("hallucination_flag") else ""
                
                with st.expander(
                    f"{status} {halluc} {log.get('query_text', 'Unknown')[:60]}... "
                    f"({log.get('timestamp', '')[:19]})"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Query ID:** {log.get('query_id')}")
                        st.markdown(f"**Papers Fetched:** {log.get('papers_fetched')}")
                        st.markdown(f"**Chunks Indexed:** {log.get('chunks_indexed')}")
                    with col2:
                        st.markdown(f"**Total Latency:** {log.get('total_latency_ms', 0)/1000:.2f}s")
                        st.markdown(f"**Tokens Used:** {log.get('groq_tokens_used', 0):,}")
                        st.markdown(f"**Hallucination:** {log.get('hallucination_flag')}")
                    
                    if log.get("error_message"):
                        st.error(f"Error: {log['error_message']}")
        else:
            st.info("No query logs yet. Run some research queries first!")
            
    except Exception as e:
        st.error(f"Error loading monitoring data: {str(e)}")


def render_about_tab():
    """Render the About tab."""
    st.subheader("ℹ️ About")
    
    st.markdown("""
    ### Agentic RAG AI Research Scientist
    
    This system uses a multi-agent architecture to dynamically search, retrieve, and synthesize 
    research papers from arXiv. It's designed to be:
    
    - **Agentic**: Multiple specialized agents work together
    - **Dynamic**: Papers are fetched on-demand, not pre-indexed
    - **Grounded**: All claims are verified against retrieved evidence
    
    ### Architecture
    
    1. **Planner Agent** - Analyzes query intent and creates search strategy
    2. **Search Agent** - Fetches papers from arXiv API
    3. **Ingestion Agent** - Downloads PDFs and extracts text chunks
    4. **Retrieval Agent** - Semantic search over document chunks
    5. **Rerank Agent** - LLM-based relevance reranking
    6. **Reasoning Agent** - Multi-document synthesis
    7. **Safety Agent** - Hallucination detection and citation verification
    
    ### Technology Stack
    
    - **LLM**: Groq API (llama-3.1-8b-instant, openai/gpt-oss-120b)
    - **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
    - **Vector Store**: FAISS (ephemeral, session-scoped)
    - **Backend**: FastAPI
    - **Frontend**: Streamlit
    
    ### Configuration
    """)
    
    # Show current config
    st.json({
        "Fast Model": config.FAST_MODEL,
        "Reasoning Model": config.REASONING_MODEL,
        "Default Papers K": config.DEFAULT_PAPERS_K,
        "Default Chunks Top N": config.DEFAULT_CHUNKS_TOP_N,
        "DEV Mode": config.DEV_MODE,
        "API Configured": bool(config.GROQ_API_KEY),
        "Monitoring Backend": "Supabase" if config.is_using_supabase() else "SQLite"
    })


def main():
    """Main app entry point."""
    init_session_state()
    render_header()
    
    # Create tabs
    tabs = ["🔬 Research", "ℹ️ About"]
    if config.DEV_MODE:
        tabs.insert(1, "📈 Monitoring")
    
    selected_tabs = st.tabs(tabs)
    
    with selected_tabs[0]:
        render_research_tab()
    
    if config.DEV_MODE:
        with selected_tabs[1]:
            render_monitoring_tab()
        with selected_tabs[2]:
            render_about_tab()
    else:
        with selected_tabs[1]:
            render_about_tab()


if __name__ == "__main__":
    main()

"""
Orchestrator for coordinating the Agentic RAG pipeline.
Manages the flow between all agents and tracks metrics.
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from agents.planner import PlannerAgent
from agents.search import SearchAgent, PaperMetadata
from agents.ingestion import IngestionAgent
from agents.retrieval import RetrievalAgent
from agents.rerank import RerankAgent
from agents.reasoning import ReasoningAgent
from agents.safety import SafetyAgent
from utils.vector_store import EphemeralVectorStore
from utils.monitoring import MonitoringService, QueryLog, QueryTimer
from utils.groq_client import groq_client
from config import config


class Orchestrator:
    """Coordinates the full Agentic RAG pipeline."""
    
    def __init__(self):
        """Initialize the orchestrator with all agents."""
        self.planner = PlannerAgent()
        self.searcher = SearchAgent()
        self.reranker = RerankAgent()
        self.reasoner = ReasoningAgent()
        self.safety = SafetyAgent()
        self.monitoring = MonitoringService()
        
        # These are created per-session
        self.vector_store: Optional[EphemeralVectorStore] = None
        self.ingestion: Optional[IngestionAgent] = None
        self.retrieval: Optional[RetrievalAgent] = None
    
    def run(
        self,
        query: str,
        papers_k: int = None,
        chunks_top_n: int = None,
        include_safety_check: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a research query.
        
        Args:
            query: User's research question
            papers_k: Number of papers to retrieve (optional)
            chunks_top_n: Number of chunks for retrieval (optional)
            include_safety_check: Whether to run safety verification
            
        Returns:
            Dict with response, metadata, and metrics
        """
        query_id = str(uuid.uuid4())[:8]
        timer = QueryTimer()
        timer.start()
        
        agent_calls = {
            "planner": 0,
            "search": 0,
            "ingestion": 0,
            "retrieval": 0,
            "rerank": 0,
            "reasoning": 0,
            "safety": 0
        }
        
        total_tokens = 0
        
        try:
            # Reset session-scoped components
            self._reset_session()
            groq_client.reset_stats()
            
            # Step 1: Planning
            plan = self.planner.plan(query)
            agent_calls["planner"] = 1
            total_tokens += plan.get("tokens_used", 0)
            
            papers_k = papers_k or plan.get("papers_k", config.DEFAULT_PAPERS_K)
            search_keywords = plan.get("search_keywords", [query])
            
            print(f"[ORCHESTRATOR] Search keywords: {search_keywords}")
            
            timer.checkpoint("planning")
            
            # Step 2: Search
            papers = self.searcher.search(
                keywords=search_keywords,
                max_results=papers_k
            )
            agent_calls["search"] = 1
            
            print(f"[ORCHESTRATOR] Papers found: {len(papers)}")
            
            if not papers:
                return self._create_no_papers_response(query, query_id, timer, agent_calls)
            
            timer.checkpoint("search")
            
            # Step 3: Ingestion
            ingestion_result = self.ingestion.ingest_papers(papers)
            agent_calls["ingestion"] = 1
            
            timer.checkpoint("ingestion")
            
            # Step 4: Retrieval
            chunks_top_n = chunks_top_n or config.DEFAULT_CHUNKS_TOP_N
            
            # Retrieve using sub-questions if available
            sub_questions = plan.get("sub_questions", [query])
            if len(sub_questions) > 1:
                retrieved_chunks = self.retrieval.retrieve_by_subquestions(
                    subquestions=sub_questions,
                    top_k_per_question=chunks_top_n // len(sub_questions) + 1
                )
            else:
                retrieved_chunks = self.retrieval.retrieve(
                    query=query,
                    top_k=chunks_top_n
                )
            agent_calls["retrieval"] = 1
            
            timer.checkpoint("retrieval")
            
            # Step 5: Reranking
            rerank_result = self.reranker.rerank(
                query=query,
                chunks=retrieved_chunks,
                top_k=min(len(retrieved_chunks), 8)
            )
            agent_calls["rerank"] = 1
            total_tokens += rerank_result.get("tokens_used", 0)
            
            reranked_chunks = rerank_result.get("ranked_chunks", retrieved_chunks)
            
            timer.checkpoint("rerank")
            
            # Step 6: Reasoning
            paper_metadata = [p.to_dict() for p in papers]
            synthesis_result = self.reasoner.synthesize(
                query=query,
                evidence=reranked_chunks,
                paper_metadata=paper_metadata
            )
            agent_calls["reasoning"] = 1
            total_tokens += synthesis_result.get("tokens_used", 0)
            
            response = synthesis_result.get("response", "")
            
            timer.checkpoint("reasoning")
            
            # Step 7: Safety Check
            hallucination_flag = False
            if include_safety_check and reranked_chunks:
                safety_result = self.safety.verify(
                    query=query,
                    response=response,
                    evidence=reranked_chunks
                )
                agent_calls["safety"] = 1
                total_tokens += safety_result.get("tokens_used", 0)
                
                hallucination_flag = safety_result.get("hallucination_detected", False)
                
                # Add disclaimers if needed
                if hallucination_flag or safety_result.get("unsupported_claims"):
                    response = self.safety.add_safety_disclaimer(response, safety_result)
                
                # Check for refusal
                if self.safety.should_refuse_response(safety_result):
                    response = self.safety.get_refusal_message(query, safety_result)
            
            timer.checkpoint("safety")
            timer.stop()
            
            # Log the query
            self._log_query(
                query_id=query_id,
                query=query,
                papers_fetched=len(papers),
                chunks_indexed=ingestion_result.get("total_chunks", 0),
                chunks_retrieved=len(reranked_chunks),
                timer=timer,
                agent_calls=agent_calls,
                hallucination_flag=hallucination_flag,
                total_tokens=total_tokens,
                success=True
            )
            
            return {
                "query_id": query_id,
                "query": query,
                "response": response,
                "papers": paper_metadata,
                "chunks_used": len(reranked_chunks),
                "plan": plan,
                "citations": synthesis_result.get("citations", []),
                "hallucination_detected": hallucination_flag,
                "metrics": {
                    "total_latency_ms": timer.total_ms,
                    "retrieval_latency_ms": timer.get_checkpoint_ms("retrieval"),
                    "tokens_used": total_tokens,
                    "agent_calls": agent_calls
                },
                "success": True
            }
            
        except Exception as e:
            timer.stop()
            
            self._log_query(
                query_id=query_id,
                query=query,
                papers_fetched=0,
                chunks_indexed=0,
                chunks_retrieved=0,
                timer=timer,
                agent_calls=agent_calls,
                hallucination_flag=False,
                total_tokens=total_tokens,
                success=False,
                error=str(e)
            )
            
            return {
                "query_id": query_id,
                "query": query,
                "response": f"An error occurred while processing your query: {str(e)}",
                "papers": [],
                "chunks_used": 0,
                "success": False,
                "error": str(e),
                "metrics": {
                    "total_latency_ms": timer.total_ms,
                    "tokens_used": total_tokens,
                    "agent_calls": agent_calls
                }
            }
    
    def _reset_session(self):
        """Reset session-scoped components."""
        self.vector_store = EphemeralVectorStore()
        self.ingestion = IngestionAgent(self.vector_store)
        self.retrieval = RetrievalAgent(self.vector_store)
    
    def _create_no_papers_response(
        self,
        query: str,
        query_id: str,
        timer: QueryTimer,
        agent_calls: Dict[str, int]
    ) -> Dict[str, Any]:
        """Create response when no papers are found."""
        timer.stop()
        
        response = f"""## No Research Papers Found

I was unable to find research papers matching your query: "{query}"

### Suggestions

1. Try using different keywords or terminology
2. Broaden your search scope
3. Check for typos in technical terms

### Direct Search

You can also search arXiv directly at [arxiv.org](https://arxiv.org/search/)
"""
        
        self._log_query(
            query_id=query_id,
            query=query,
            papers_fetched=0,
            chunks_indexed=0,
            chunks_retrieved=0,
            timer=timer,
            agent_calls=agent_calls,
            hallucination_flag=False,
            total_tokens=0,
            success=True
        )
        
        return {
            "query_id": query_id,
            "query": query,
            "response": response,
            "papers": [],
            "chunks_used": 0,
            "success": True,
            "metrics": {
                "total_latency_ms": timer.total_ms,
                "tokens_used": 0,
                "agent_calls": agent_calls
            }
        }
    
    def _log_query(
        self,
        query_id: str,
        query: str,
        papers_fetched: int,
        chunks_indexed: int,
        chunks_retrieved: int,
        timer: QueryTimer,
        agent_calls: Dict[str, int],
        hallucination_flag: bool,
        total_tokens: int,
        success: bool,
        error: str = None
    ):
        """Log query to monitoring database."""
        log = QueryLog(
            query_id=query_id,
            timestamp=datetime.now().isoformat(),
            query_text=query,
            papers_fetched=papers_fetched,
            chunks_indexed=chunks_indexed,
            chunks_retrieved=chunks_retrieved,
            retrieval_latency_ms=timer.get_checkpoint_ms("retrieval"),
            total_latency_ms=timer.total_ms,
            groq_tokens_used=total_tokens,
            agent_calls=agent_calls,
            hallucination_flag=hallucination_flag,
            success=success,
            error_message=error
        )
        
        self.monitoring.log_query(log)
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return self.monitoring.get_aggregate_stats()
    
    def get_recent_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent query logs."""
        return self.monitoring.get_recent_logs(limit)


# Global orchestrator instance
orchestrator = Orchestrator()

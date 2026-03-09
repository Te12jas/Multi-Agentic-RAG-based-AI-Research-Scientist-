"""
Rerank Agent for LLM-based reranking of retrieved chunks.
Uses Groq to evaluate and reorder chunks by relevance.
"""

import os
import json
from typing import Dict, Any, List
from utils.groq_client import groq_client
from config import config


class RerankAgent:
    """Agent responsible for reranking retrieved chunks using LLM."""
    
    def __init__(self):
        """Initialize the rerank agent."""
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load prompt template from file."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "rerank_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: User's research query
            chunks: List of chunk dicts to rerank
            top_k: Number of top chunks to return
            
        Returns:
            Dict with reranked chunks and metadata
        """
        if not chunks:
            return {
                "ranked_chunks": [],
                "filtered_out": [],
                "tokens_used": 0
            }
        
        top_k = top_k or min(len(chunks), 10)
        
        # Format chunks for the prompt
        chunks_text = self._format_chunks_for_prompt(chunks)
        
        prompt = self.prompt_template.format(
            query=query,
            chunks=chunks_text
        )
        
        result = groq_client.complete_json(
            prompt=prompt,
            model=config.FAST_MODEL,
            temperature=0.1,
            max_tokens=2048
        )
        
        if result.get("parsed"):
            rerank_result = result["parsed"]
            
            # Map back to original chunks with scores
            ranked_chunks = self._map_ranked_chunks(
                rerank_result.get("ranked_chunks", []),
                chunks,
                top_k
            )
            
            return {
                "ranked_chunks": ranked_chunks,
                "filtered_out": rerank_result.get("filtered_out", []),
                "summary": rerank_result.get("summary", ""),
                "tokens_used": result.get("tokens_used", 0),
                "latency": result.get("latency", 0)
            }
        
        # Fallback: return original chunks with default scores
        return {
            "ranked_chunks": chunks[:top_k],
            "filtered_out": [],
            "summary": "",
            "tokens_used": result.get("tokens_used", 0),
            "latency": result.get("latency", 0),
            "fallback": True
        }
    
    def _format_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks as numbered list for the prompt."""
        formatted = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"][:500] if len(chunk["text"]) > 500 else chunk["text"]
            formatted.append(
                f"[{chunk['chunk_id']}]\n"
                f"Paper: {chunk['paper_title']} ({chunk['year']})\n"
                f"Section: {chunk['section']}\n"
                f"Text: {text}\n"
            )
        return "\n---\n".join(formatted)
    
    def _map_ranked_chunks(
        self,
        ranked_items: List[Dict[str, Any]],
        original_chunks: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Map ranked items back to original chunks."""
        # Create lookup by chunk_id
        chunk_lookup = {c["chunk_id"]: c for c in original_chunks}
        
        ranked_chunks = []
        for item in ranked_items[:top_k]:
            chunk_id = item.get("chunk_id")
            if chunk_id and chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id].copy()
                chunk["relevance_score"] = item.get("relevance_score", 0.5)
                chunk["rerank_reasoning"] = item.get("reasoning", "")
                ranked_chunks.append(chunk)
        
        # Sort by relevance score
        ranked_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return ranked_chunks
    
    def batch_rerank(
        self,
        query: str,
        chunk_batches: List[List[Dict[str, Any]]],
        top_k_per_batch: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank multiple batches and combine results.
        
        Args:
            query: User's research query
            chunk_batches: List of chunk batches
            top_k_per_batch: Top K to keep from each batch
            
        Returns:
            Combined and sorted list of reranked chunks
        """
        all_ranked = []
        
        for batch in chunk_batches:
            result = self.rerank(query, batch, top_k_per_batch)
            all_ranked.extend(result.get("ranked_chunks", []))
        
        # Sort by relevance score
        all_ranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return all_ranked

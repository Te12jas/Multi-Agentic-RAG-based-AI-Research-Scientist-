"""
Retrieval Agent for dense retrieval from the vector store.
Handles similarity search and chunk filtering.
"""

from typing import Dict, Any, List, Optional, Tuple
from utils.vector_store import EphemeralVectorStore, DocumentChunk
from config import config


class RetrievalAgent:
    """Agent responsible for retrieving relevant document chunks."""
    
    def __init__(self, vector_store: EphemeralVectorStore):
        """
        Initialize the retrieval agent.
        
        Args:
            vector_store: Vector store to retrieve from
        """
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        paper_filter: Optional[List[str]] = None,
        section_filter: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            paper_filter: Optional list of paper IDs to filter by
            section_filter: Optional list of sections to filter by
            min_score: Minimum similarity score threshold
            
        Returns:
            List of chunk dicts with scores
        """
        top_k = top_k or config.DEFAULT_CHUNKS_TOP_N
        
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            paper_filter=paper_filter,
            section_filter=section_filter
        )
        
        # Filter by minimum score and format results
        formatted_results = []
        for chunk, score in results:
            if score >= min_score:
                formatted_results.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "authors": chunk.authors,
                    "year": chunk.year,
                    "section": chunk.section,
                    "score": score,
                    "metadata": chunk.metadata
                })
        
        return formatted_results
    
    def retrieve_by_subquestions(
        self,
        subquestions: List[str],
        top_k_per_question: int = 5,
        deduplicate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks for multiple sub-questions.
        
        Args:
            subquestions: List of sub-questions to search
            top_k_per_question: Results per sub-question
            deduplicate: Remove duplicate chunks
            
        Returns:
            List of unique chunk dicts with scores
        """
        all_results = []
        seen_chunk_ids = set()
        
        for question in subquestions:
            results = self.retrieve(
                query=question,
                top_k=top_k_per_question
            )
            
            for result in results:
                if deduplicate:
                    if result["chunk_id"] not in seen_chunk_ids:
                        seen_chunk_ids.add(result["chunk_id"])
                        all_results.append(result)
                else:
                    all_results.append(result)
        
        # Sort by score and return
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results
    
    def retrieve_by_section(
        self,
        query: str,
        sections: List[str],
        top_k_per_section: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks organized by section.
        
        Args:
            query: Search query
            sections: List of sections to retrieve from
            top_k_per_section: Results per section
            
        Returns:
            Dict mapping section names to chunk lists
        """
        results_by_section = {}
        
        for section in sections:
            results = self.retrieve(
                query=query,
                top_k=top_k_per_section,
                section_filter=[section]
            )
            results_by_section[section] = results
        
        return results_by_section
    
    def get_context_for_reasoning(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int = 6000
    ) -> str:
        """
        Format retrieved chunks as context for reasoning.
        
        Args:
            chunks: List of chunk dicts
            max_tokens: Approximate maximum context length
            
        Returns:
            Formatted context string
        """
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            chunk_text = self._format_chunk(chunk)
            chunk_len = len(chunk_text)
            
            if current_length + chunk_len > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_len
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_chunk(self, chunk: Dict[str, Any]) -> str:
        """Format a single chunk for context."""
        authors_str = ", ".join(chunk["authors"][:3])
        if len(chunk["authors"]) > 3:
            authors_str += " et al."
        
        return f"""[Source: {chunk["paper_title"]} ({chunk["year"]})]
[Authors: {authors_str}]
[Section: {chunk["section"]}]

{chunk["text"]}"""
    
    def get_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.stats()

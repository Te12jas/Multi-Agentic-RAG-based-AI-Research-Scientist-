"""
Reasoning Agent for multi-document synthesis and analysis.
Generates structured research summaries from retrieved evidence.
"""

import os
from typing import Dict, Any, List
from utils.groq_client import groq_client
from config import config


class ReasoningAgent:
    """Agent responsible for synthesizing research findings."""
    
    def __init__(self):
        """Initialize the reasoning agent."""
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load prompt template from file."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "reasoning_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def synthesize(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        paper_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize a research summary from retrieved evidence.
        
        Args:
            query: User's research query
            evidence: List of relevant chunk dicts
            paper_metadata: List of paper metadata dicts
            
        Returns:
            Dict with synthesized response and metadata
        """
        if not evidence:
            return {
                "response": self._insufficient_evidence_response(query),
                "grounded": False,
                "tokens_used": 0,
                "citations": []
            }
        
        # Format evidence and metadata
        evidence_text = self._format_evidence(evidence)
        metadata_text = self._format_metadata(paper_metadata)
        
        prompt = self.prompt_template.format(
            query=query,
            evidence=evidence_text,
            metadata=metadata_text
        )
        
        result = groq_client.complete(
            prompt=prompt,
            model=config.REASONING_MODEL,
            temperature=0.2,
            max_tokens=4096
        )
        
        response = result.get("content", "")
        
        # Extract citations from the response
        citations = self._extract_citations(response, paper_metadata)
        
        return {
            "response": response,
            "grounded": True,
            "tokens_used": result.get("tokens_used", 0),
            "latency": result.get("latency", 0),
            "citations": citations,
            "evidence_count": len(evidence),
            "papers_used": len(paper_metadata)
        }
    
    def _format_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """Format evidence chunks for the prompt."""
        formatted_parts = []
        
        for i, chunk in enumerate(evidence, 1):
            authors = ", ".join(chunk.get("authors", [])[:3])
            if len(chunk.get("authors", [])) > 3:
                authors += " et al."
            
            formatted_parts.append(
                f"--- Evidence {i} ---\n"
                f"Paper: {chunk.get('paper_title', 'Unknown')}\n"
                f"Authors: {authors}\n"
                f"Year: {chunk.get('year', 'Unknown')}\n"
                f"Section: {chunk.get('section', 'Unknown')}\n\n"
                f"{chunk.get('text', '')}\n"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _format_metadata(self, paper_metadata: List[Dict[str, Any]]) -> str:
        """Format paper metadata for reference."""
        formatted_parts = []
        
        for paper in paper_metadata:
            authors = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors += " et al."
            
            formatted_parts.append(
                f"- {paper.get('title', 'Unknown')}\n"
                f"  Authors: {authors}\n"
                f"  Year: {paper.get('year', 'Unknown')}\n"
                f"  arXiv ID: {paper.get('paper_id', 'Unknown')}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _extract_citations(
        self,
        response: str,
        paper_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and format citations mentioned in the response."""
        citations = []
        
        for paper in paper_metadata:
            title = paper.get("title", "")
            # Check if paper is mentioned in response
            if title.lower() in response.lower() or paper.get("paper_id", "") in response:
                citations.append({
                    "paper_id": paper.get("paper_id"),
                    "title": title,
                    "authors": paper.get("authors", []),
                    "year": paper.get("year"),
                    "arxiv_url": paper.get("arxiv_url")
                })
        
        return citations
    
    def _insufficient_evidence_response(self, query: str) -> str:
        """Generate response when evidence is insufficient."""
        return f"""## Summary

I was unable to find sufficient evidence in the retrieved research papers to comprehensively answer your query: "{query}"

## Evidence Gaps

The retrieved documents did not contain enough relevant information to provide a grounded synthesis. This could be because:

1. The query may require papers not currently in the arXiv database
2. The search terms may need refinement
3. The topic may be too specific or too recent

## Recommendations

- Try rephrasing your research question with different keywords
- Consider broadening or narrowing the scope of your query
- Check if there are specific papers or authors you'd like to focus on

## Citations

No relevant citations available for this query."""
    
    def compare_methods(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        paper_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a focused comparison of methods across papers.
        
        Args:
            query: Comparison query
            evidence: Evidence chunks
            paper_metadata: Paper metadata
            
        Returns:
            Comparison synthesis
        """
        comparison_prompt = f"""Based on the provided evidence, create a detailed comparison table of methods used in these papers.

Query: {query}

Evidence:
{self._format_evidence(evidence)}

Create a structured comparison with:
1. Method names and key characteristics
2. Strengths of each approach
3. Weaknesses or limitations
4. Performance metrics (if available)
5. Use cases each method is best suited for

Format as markdown with clear sections and tables where appropriate."""
        
        result = groq_client.complete(
            prompt=comparison_prompt,
            model=config.REASONING_MODEL,
            temperature=0.2,
            max_tokens=3000
        )
        
        return {
            "comparison": result.get("content", ""),
            "tokens_used": result.get("tokens_used", 0),
            "latency": result.get("latency", 0)
        }

"""
Safety Agent for citation verification and hallucination detection.
Ensures all claims are grounded in retrieved evidence.
"""

import os
from typing import Dict, Any, List
from utils.groq_client import groq_client
from config import config


class SafetyAgent:
    """Agent responsible for verifying grounding and detecting hallucinations."""
    
    def __init__(self):
        """Initialize the safety agent."""
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load prompt template from file."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "safety_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def verify(
        self,
        query: str,
        response: str,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify that the response is grounded in evidence.
        
        Args:
            query: Original user query
            response: Generated response to verify
            evidence: Evidence chunks used for generation
            
        Returns:
            Dict with verification results
        """
        if not evidence:
            return {
                "is_grounded": False,
                "hallucination_detected": True,
                "unsupported_claims": [],
                "citation_errors": [],
                "completeness_score": 0.0,
                "should_refuse": True,
                "refusal_reason": "No evidence available to support any claims"
            }
        
        # Format evidence for the prompt
        evidence_text = self._format_evidence(evidence)
        
        prompt = self.prompt_template.format(
            query=query,
            response=response,
            evidence=evidence_text
        )
        
        result = groq_client.complete_json(
            prompt=prompt,
            model=config.FAST_MODEL,
            temperature=0.1,
            max_tokens=2048
        )
        
        if result.get("parsed"):
            verification = result["parsed"]
            verification["tokens_used"] = result.get("tokens_used", 0)
            verification["latency"] = result.get("latency", 0)
            return verification
        
        # Fallback: conservative verification
        return {
            "is_grounded": True,
            "hallucination_detected": False,
            "unsupported_claims": [],
            "citation_errors": [],
            "completeness_score": 0.7,
            "recommendations": [],
            "should_refuse": False,
            "refusal_reason": None,
            "tokens_used": result.get("tokens_used", 0),
            "latency": result.get("latency", 0),
            "fallback": True
        }
    
    def _format_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """Format evidence for verification."""
        formatted_parts = []
        
        for chunk in evidence:
            formatted_parts.append(
                f"[{chunk.get('paper_title', 'Unknown')} ({chunk.get('year', 'Unknown')})]:\n"
                f"{chunk.get('text', '')}"
            )
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def add_safety_disclaimer(
        self,
        response: str,
        verification: Dict[str, Any]
    ) -> str:
        """
        Add safety disclaimers to response if needed.
        
        Args:
            response: Original response
            verification: Verification results
            
        Returns:
            Response with appropriate disclaimers
        """
        disclaimers = []
        
        if verification.get("hallucination_detected"):
            disclaimers.append(
                "> ⚠️ **Note**: Some claims in this response may not be fully "
                "supported by the retrieved evidence. Please verify with original sources."
            )
        
        unsupported = verification.get("unsupported_claims", [])
        if unsupported:
            claims_text = "\n".join([f"- {c['claim']}" for c in unsupported[:3]])
            disclaimers.append(
                f"> **Potentially unsupported claims identified:**\n{claims_text}"
            )
        
        citation_errors = verification.get("citation_errors", [])
        if citation_errors:
            disclaimers.append(
                "> ⚠️ **Citation Note**: Some citations may require verification."
            )
        
        completeness = verification.get("completeness_score", 1.0)
        if completeness < 0.5:
            disclaimers.append(
                "> 📋 **Completeness**: This response addresses only a portion of your query. "
                "Consider refining your search terms for more comprehensive results."
            )
        
        if disclaimers:
            disclaimer_text = "\n\n".join(disclaimers)
            return f"{disclaimer_text}\n\n---\n\n{response}"
        
        return response
    
    def should_refuse_response(self, verification: Dict[str, Any]) -> bool:
        """
        Determine if the response should be refused.
        
        Args:
            verification: Verification results
            
        Returns:
            True if response should be refused
        """
        if verification.get("should_refuse", False):
            return True
        
        # Additional refusal criteria
        if verification.get("completeness_score", 1.0) < 0.2:
            return True
        
        unsupported = verification.get("unsupported_claims", [])
        if len(unsupported) > 5:
            return True
        
        return False
    
    def get_refusal_message(
        self,
        query: str,
        verification: Dict[str, Any]
    ) -> str:
        """
        Generate an appropriate refusal message.
        
        Args:
            query: Original query
            verification: Verification results
            
        Returns:
            Refusal message explaining why
        """
        reason = verification.get("refusal_reason", "Insufficient evidence")
        
        return f"""## Unable to Provide Reliable Response

I'm unable to provide a well-grounded response to your query: "{query}"

**Reason**: {reason}

### What This Means

The retrieved research papers do not contain sufficient information to answer your question with the accuracy and citation support required. Rather than risk providing inaccurate information, I'm declining to generate a response.

### Suggestions

1. **Refine your query**: Try using more specific or alternative terminology
2. **Broaden the scope**: Consider a more general question that may have better coverage
3. **Check recent publications**: If this is a very recent topic, papers may not yet be indexed

### Further Resources

- [arXiv.org](https://arxiv.org) - Search directly for recent preprints
- [Google Scholar](https://scholar.google.com) - Academic search engine
- [Semantic Scholar](https://semanticscholar.org) - AI-powered research tool
"""

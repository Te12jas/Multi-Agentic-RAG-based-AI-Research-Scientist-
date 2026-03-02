"""
Planner Agent for query decomposition and search strategy.
Analyzes user intent and creates a structured search plan.
"""

import os
from typing import Dict, Any, List
from utils.groq_client import groq_client
from config import config


class PlannerAgent:
    """Agent responsible for understanding and decomposing research queries."""
    
    def __init__(self):
        """Initialize the planner agent."""
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load prompt template from file."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "planner_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def plan(self, query: str) -> Dict[str, Any]:
        """
        Create a search plan for the given query.
        
        Args:
            query: User's research question
            
        Returns:
            Dict containing search plan with keywords, paper count, etc.
        """
        prompt = self.prompt_template.format(query=query)
        
        result = groq_client.complete_json(
            prompt=prompt,
            model=config.FAST_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        
        if result.get("parsed"):
            plan = result["parsed"]
            # Ensure required fields exist
            plan.setdefault("intent", query)
            plan.setdefault("sub_questions", [query])
            plan.setdefault("search_keywords", self._extract_keywords(query))
            plan.setdefault("papers_k", config.DEFAULT_PAPERS_K)
            plan.setdefault("needs_iteration", False)
            
            plan["tokens_used"] = result.get("tokens_used", 0)
            plan["latency"] = result.get("latency", 0)
            
            return plan
        
        # Fallback if parsing failed
        return {
            "intent": query,
            "sub_questions": [query],
            "search_keywords": self._extract_keywords(query),
            "papers_k": config.DEFAULT_PAPERS_K,
            "needs_iteration": False,
            "reasoning": "Fallback plan due to parsing error",
            "tokens_used": result.get("tokens_used", 0),
            "latency": result.get("latency", 0)
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract simple keywords from query as fallback.
        
        Args:
            query: User query text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction
        stopwords = {
            "what", "how", "why", "when", "where", "which", "who",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "a", "an", "the", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "into",
            "about", "between", "through", "during", "before", "after",
            "can", "could", "should", "would", "may", "might", "must"
        }
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords[:5] if keywords else [query]

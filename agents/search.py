"""
Search Agent for dynamically fetching research papers from arXiv.
Uses the arXiv API to search and retrieve paper metadata.
"""

import arxiv
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from config import config


@dataclass
class PaperMetadata:
    """Metadata for a research paper."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    year: str
    pdf_url: str
    arxiv_url: str
    categories: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SearchAgent:
    """Agent responsible for searching and fetching papers from arXiv."""
    
    def __init__(self):
        """Initialize the search agent."""
        self.client = arxiv.Client()
    
    def search(
        self,
        keywords: List[str],
        max_results: int = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[PaperMetadata]:
        """
        Search arXiv for papers matching keywords.
        
        Args:
            keywords: List of search keywords
            max_results: Maximum number of papers to return
            sort_by: Sort criterion (Relevance, SubmittedDate, LastUpdatedDate)
            
        Returns:
            List of PaperMetadata objects
        """
        max_results = max_results or config.DEFAULT_PAPERS_K
        
        # Build search query
        query = self._build_query(keywords)
        
        # Create search object
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        try:
            for result in self.client.results(search):
                paper = self._parse_result(result)
                papers.append(paper)
        except Exception as e:
            print(f"Error searching arXiv: {e}")
        
        return papers
    
    def search_by_id(self, paper_ids: List[str]) -> List[PaperMetadata]:
        """
        Fetch specific papers by their arXiv IDs.
        
        Args:
            paper_ids: List of arXiv paper IDs (e.g., "2301.00001")
            
        Returns:
            List of PaperMetadata objects
        """
        search = arxiv.Search(id_list=paper_ids)
        
        papers = []
        try:
            for result in self.client.results(search):
                paper = self._parse_result(result)
                papers.append(paper)
        except Exception as e:
            print(f"Error fetching papers by ID: {e}")
        
        return papers
    
    def _build_query(self, keywords: List[str]) -> str:
        """
        Build arXiv search query from keywords.
        
        Args:
            keywords: List of search terms
            
        Returns:
            Formatted arXiv query string
        """
        # Join keywords with OR for more inclusive results
        # Use simpler query format without field specifiers for better matching
        terms = []
        for kw in keywords:
            # Remove any special characters and quotes
            kw = kw.strip().replace('"', '').replace("'", "").replace("?", "")
            if kw and len(kw) > 1:
                terms.append(kw)
        
        if not terms:
            return "machine learning"
        
        # Use OR between terms for broader results
        # Search in title and abstract for better relevance
        query_parts = []
        for term in terms[:3]:  # Limit to 3 main terms
            query_parts.append(f'ti:"{term}" OR abs:"{term}"')
        
        query = " OR ".join(query_parts)
        print(f"[SEARCH] arXiv query: {query}")
        return query
    
    def _parse_result(self, result: arxiv.Result) -> PaperMetadata:
        """
        Parse arXiv result into PaperMetadata.
        
        Args:
            result: arXiv search result
            
        Returns:
            PaperMetadata object
        """
        # Extract paper ID from entry_id
        paper_id = result.entry_id.split("/")[-1]
        
        # Get publication year
        year = str(result.published.year) if result.published else "Unknown"
        
        return PaperMetadata(
            paper_id=paper_id,
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            year=year,
            pdf_url=result.pdf_url,
            arxiv_url=result.entry_id,
            categories=list(result.categories)
        )
    
    def get_papers_metadata(self, papers: List[PaperMetadata]) -> List[Dict[str, Any]]:
        """
        Convert papers to dictionary format for downstream use.
        
        Args:
            papers: List of PaperMetadata objects
            
        Returns:
            List of paper dictionaries
        """
        return [p.to_dict() for p in papers]

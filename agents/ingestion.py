"""
Ingestion Agent for processing PDFs into structured chunks.
Downloads papers and converts them to indexed document chunks.
"""

import uuid
from typing import Dict, Any, List, Optional
from agents.search import PaperMetadata
from utils.pdf_utils import pdf_processor
from utils.vector_store import DocumentChunk, EphemeralVectorStore


class IngestionAgent:
    """Agent responsible for downloading and processing research papers."""
    
    def __init__(self, vector_store: Optional[EphemeralVectorStore] = None):
        """
        Initialize the ingestion agent.
        
        Args:
            vector_store: Optional vector store to add chunks to
        """
        self.vector_store = vector_store or EphemeralVectorStore()
    
    def ingest_paper(self, paper: PaperMetadata) -> List[DocumentChunk]:
        """
        Ingest a single paper into document chunks.
        
        Args:
            paper: PaperMetadata object with paper info
            
        Returns:
            List of DocumentChunk objects
        """
        # Download and extract text
        full_text, raw_chunks = pdf_processor.process_paper(
            url=paper.pdf_url,
            paper_id=paper.paper_id
        )
        
        if not raw_chunks:
            # Fallback to abstract if PDF processing fails
            raw_chunks = [{
                "text": paper.abstract,
                "section": "abstract"
            }]
        
        # Convert to DocumentChunk objects
        chunks = []
        for i, chunk_data in enumerate(raw_chunks):
            chunk = DocumentChunk(
                chunk_id=f"{paper.paper_id}_chunk_{i}",
                text=chunk_data["text"],
                paper_id=paper.paper_id,
                paper_title=paper.title,
                authors=paper.authors,
                year=paper.year,
                section=chunk_data.get("section", "body"),
                metadata={
                    "pdf_url": paper.pdf_url,
                    "arxiv_url": paper.arxiv_url,
                    "categories": paper.categories
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def ingest_papers(
        self,
        papers: List[PaperMetadata],
        add_to_store: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest multiple papers and optionally add to vector store.
        
        Args:
            papers: List of PaperMetadata objects
            add_to_store: Whether to add chunks to vector store
            
        Returns:
            Dict with ingestion statistics
        """
        all_chunks = []
        papers_processed = 0
        papers_failed = []
        
        for paper in papers:
            try:
                chunks = self.ingest_paper(paper)
                all_chunks.extend(chunks)
                papers_processed += 1
            except Exception as e:
                print(f"Error ingesting paper {paper.paper_id}: {e}")
                papers_failed.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "error": str(e)
                })
        
        # Add to vector store
        if add_to_store and all_chunks:
            self.vector_store.add_chunks(all_chunks)
        
        return {
            "papers_processed": papers_processed,
            "papers_failed": len(papers_failed),
            "failed_papers": papers_failed,
            "total_chunks": len(all_chunks),
            "chunks_by_section": self._count_by_section(all_chunks)
        }
    
    def _count_by_section(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Count chunks by section type."""
        counts = {}
        for chunk in chunks:
            section = chunk.section
            counts[section] = counts.get(section, 0) + 1
        return counts
    
    def ingest_from_urls(
        self,
        paper_data: List[Dict[str, Any]],
        add_to_store: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest papers from raw URL and metadata.
        
        Args:
            paper_data: List of dicts with 'pdf_url', 'paper_id', 'title', etc.
            add_to_store: Whether to add chunks to vector store
            
        Returns:
            Dict with ingestion statistics
        """
        papers = []
        for data in paper_data:
            paper = PaperMetadata(
                paper_id=data.get("paper_id", str(uuid.uuid4())[:8]),
                title=data.get("title", "Unknown"),
                authors=data.get("authors", []),
                abstract=data.get("abstract", ""),
                year=data.get("year", "Unknown"),
                pdf_url=data["pdf_url"],
                arxiv_url=data.get("arxiv_url", data["pdf_url"]),
                categories=data.get("categories", [])
            )
            papers.append(paper)
        
        return self.ingest_papers(papers, add_to_store)
    
    def get_vector_store(self) -> EphemeralVectorStore:
        """Get the vector store."""
        return self.vector_store
    
    def clear(self):
        """Clear the vector store."""
        self.vector_store.clear()

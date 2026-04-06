"""
Ephemeral vector store for session-scoped document retrieval.
Uses FAISS for efficient similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from dataclasses import dataclass, field
from utils.embeddings import embedding_service


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    chunk_id: str
    text: str
    paper_id: str
    paper_title: str
    authors: List[str]
    year: str
    section: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EphemeralVectorStore:
    """
    Session-scoped vector store for temporary document storage.
    Designed to be created fresh for each research query session.
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.is_initialized = False
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to add
        """
        if not chunks:
            return
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_documents(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Initialize index if needed
        if not self.is_initialized:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            self.is_initialized = True
        
        # Add to index
        start_idx = len(self.chunks)
        self.index.add(embeddings)
        
        # Store chunks and update mapping
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        paper_filter: Optional[List[str]] = None,
        section_filter: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            paper_filter: Optional list of paper IDs to filter by
            section_filter: Optional list of sections to filter by
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if not self.is_initialized or self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = embedding_service.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search with more results if filtering
        search_k = min(top_k * 3, self.index.ntotal) if (paper_filter or section_filter) else min(top_k, self.index.ntotal)
        
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            chunk = self.chunks[idx]
            
            # Apply filters
            if paper_filter and chunk.paper_id not in paper_filter:
                continue
            if section_filter and chunk.section not in section_filter:
                continue
            
            results.append((chunk, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None
    
    def get_papers(self) -> List[Dict[str, Any]]:
        """Get list of unique papers in the store."""
        papers = {}
        for chunk in self.chunks:
            if chunk.paper_id not in papers:
                papers[chunk.paper_id] = {
                    "paper_id": chunk.paper_id,
                    "title": chunk.paper_title,
                    "authors": chunk.authors,
                    "year": chunk.year,
                    "chunk_count": 0
                }
            papers[chunk.paper_id]["chunk_count"] += 1
        
        return list(papers.values())
    
    def clear(self):
        """Clear all stored data."""
        self.index = None
        self.chunks = []
        self.chunk_id_to_idx = {}
        self.is_initialized = False
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        papers = self.get_papers()
        return {
            "total_chunks": len(self.chunks),
            "total_papers": len(papers),
            "papers": papers
        }

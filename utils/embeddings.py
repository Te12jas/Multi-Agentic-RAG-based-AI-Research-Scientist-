"""
Embedding service for semantic search.
Uses sentence-transformers for generating text embeddings.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config


class EmbeddingService:
    """Service for generating text embeddings."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model."""
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the embedding model."""
        return EmbeddingService._model
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Numpy array of embeddings (N x embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array of embedding
        """
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            2D numpy array of embeddings
        """
        return self.embed(documents)
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


# Global instance
embedding_service = EmbeddingService()

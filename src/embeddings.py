"""
Embeddings module using sentence-transformers.
Handles embedding generation for queries and documents.
"""

import time
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Wrapper for sentence-transformers models with latency tracking."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default: all-MiniLM-L6-v2 (fast, 384 dims)
                       Alternative: all-mpnet-base-v2 (better quality, 768 dims)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_text(self, text: Union[str, List[str]], show_progress: bool = False) -> tuple:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single text string or list of text strings
            show_progress: Whether to show progress bar for batch processing
            
        Returns:
            tuple: (embeddings as numpy array, latency in milliseconds)
        """
        start_time = time.time()
        
        if isinstance(text, str):
            text = [text]
            
        embeddings = self.model.encode(
            text,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return embeddings, latency_ms
    
    def embed_query(self, query: str) -> tuple:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text string
            
        Returns:
            tuple: (embedding as numpy array, latency in milliseconds)
        """
        return self.embed_text(query, show_progress=False)
    
    def embed_documents(self, documents: List[str], batch_size: int = 32) -> tuple:
        """
        Generate embeddings for multiple documents with batching.
        
        Args:
            documents: List of document text strings
            batch_size: Number of documents to process in each batch
            
        Returns:
            tuple: (embeddings as numpy array, latency in milliseconds)
        """
        start_time = time.time()
        
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return embeddings, latency_ms
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim

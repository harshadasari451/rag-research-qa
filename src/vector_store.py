"""
Vector store module using ChromaDB.
Handles document storage and similarity search.
"""

import time
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB vector store for document embeddings with retrieval."""
    
    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_directory: str = "./data/chroma_db",
        embedding_dimension: int = 384
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_dimension: Dimension of embeddings (384 for MiniLM, 768 for mpnet)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> float:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document text strings
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique IDs for each document
            
        Returns:
            float: Latency in milliseconds
        """
        start_time = time.time()
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        latency_ms = (time.time() - start_time) * 1000
        return latency_ms
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return (top-k)
            filter_dict: Optional metadata filters
            
        Returns:
            tuple: (results dictionary, latency in milliseconds)
        """
        start_time = time.time()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return results, latency_ms
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": self.persist_directory
        }

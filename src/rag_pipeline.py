"""
RAG Pipeline - End-to-end retrieval augmented generation workflow.
Orchestrates embedding, retrieval, and answer generation.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.metrics import MetricsTracker


class RAGPipeline:
    """Complete RAG pipeline for question answering over research papers."""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "research_papers",
        persist_directory: str = "./data/chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model_name: Name of sentence-transformers model
            collection_name: ChromaDB collection name
            persist_directory: Directory for ChromaDB persistence
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        # Initialize components
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_dimension=self.embedding_model.get_embedding_dimension()
        )
        self.metrics_tracker = MetricsTracker()
        
    def index_documents(self, pdf_directory: str) -> Dict[str, Any]:
        """
        Index all PDF documents from a directory.
        
        Args:
            pdf_directory: Path to directory containing PDF files
            
        Returns:
            Dictionary with indexing statistics
        """
        start_time = time.time()
        
        # Load PDFs
        print(f"Loading PDFs from {pdf_directory}...")
        documents = self.document_processor.load_pdfs_from_directory(pdf_directory)
        
        if not documents:
            return {
                'num_documents': 0,
                'num_chunks': 0,
                'indexing_time_ms': 0,
                'error': 'No PDF files found'
            }
        
        print(f"Loaded {len(documents)} documents")
        
        # Process and chunk documents
        print("Chunking documents...")
        chunks = self.document_processor.process_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings, embed_latency = self.embedding_model.embed_documents(texts)
        print(f"Generated embeddings in {embed_latency:.2f}ms")
        
        # Prepare data for vector store
        ids = [f"{chunk['metadata']['filename']}_{chunk['metadata']['chunk_id']}" 
               for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add to vector store
        print("Adding to vector store...")
        add_latency = self.vector_store.add_documents(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return {
            'num_documents': len(documents),
            'num_chunks': len(chunks),
            'embedding_latency_ms': embed_latency,
            'add_latency_ms': add_latency,
            'indexing_time_ms': total_time_ms
        }
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer, sources, and metrics
        """
        total_start = time.time()
        
        # 1. Generate query embedding
        query_embedding, embedding_latency = self.embedding_model.embed_query(question)
        
        # 2. Retrieve relevant chunks
        results, retrieval_latency = self.vector_store.query(
            query_embedding=query_embedding[0].tolist(),
            n_results=top_k
        )
        
        # 3. Generate answer (extractive approach)
        generation_start = time.time()
        answer, sources = self._generate_answer(results, question)
        generation_latency = (time.time() - generation_start) * 1000
        
        # Calculate total latency
        total_latency = (time.time() - total_start) * 1000
        
        # Record metrics
        metrics = {
            'embedding_latency_ms': embedding_latency,
            'retrieval_latency_ms': retrieval_latency,
            'generation_latency_ms': generation_latency,
            'total_latency_ms': total_latency,
            'num_retrieved': len(results['documents'][0]) if results['documents'] else 0
        }
        self.metrics_tracker.record_query(metrics)
        
        response = {
            'answer': answer,
            'metrics': metrics
        }
        
        if return_sources:
            response['sources'] = sources
        
        return response
    
    def _generate_answer(
        self,
        retrieval_results: Dict[str, Any],
        question: str
    ) -> tuple:
        """
        Generate answer from retrieved chunks (extractive approach).
        
        Args:
            retrieval_results: Results from vector store query
            question: Original question
            
        Returns:
            tuple: (answer string, list of source dictionaries)
        """
        # Extract retrieved documents and metadata
        documents = retrieval_results.get('documents', [[]])[0]
        metadatas = retrieval_results.get('metadatas', [[]])[0]
        distances = retrieval_results.get('distances', [[]])[0]
        
        if not documents:
            return "I couldn't find any relevant information to answer your question.", []
        
        # Prepare sources with similarity scores
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            similarity_score = 1 - dist  # Convert distance to similarity
            sources.append({
                'text': doc,
                'metadata': meta,
                'similarity_score': similarity_score,
                'rank': i + 1
            })
        
        # Create answer by combining top chunks
        # For a simple extractive approach, we use the most relevant chunks
        context = "\n\n".join([
            f"[Source {i+1}]: {doc}" 
            for i, doc in enumerate(documents[:3])  # Top 3 chunks
        ])
        
        answer = f"""Based on the retrieved documents:

{context}

**Summary**: The most relevant information from the research papers suggests that the answer relates to the content shown above. For more detailed information, please refer to the specific source documents listed."""
        
        return answer, sources
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the indexed documents."""
        return self.vector_store.get_collection_info()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics_tracker.get_summary()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics_tracker.reset()
    
    def reset_index(self):
        """Reset the vector store index."""
        self.vector_store.reset_collection()

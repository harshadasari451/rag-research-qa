"""
Basic tests for the RAG pipeline components.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings import EmbeddingModel
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.metrics import MetricsTracker
from src.rag_pipeline import RAGPipeline


def test_embedding_model():
    """Test embedding model initialization and embedding generation."""
    print("\n=== Testing Embedding Model ===")
    
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    # Test single text embedding
    text = "This is a test sentence."
    embeddings, latency = model.embed_text(text)
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == model.get_embedding_dimension()
    assert latency > 0
    print(f"✓ Single text embedding: shape={embeddings.shape}, latency={latency:.2f}ms")
    
    # Test batch embedding
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings, latency = model.embed_text(texts)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == model.get_embedding_dimension()
    print(f"✓ Batch text embedding: shape={embeddings.shape}, latency={latency:.2f}ms")
    
    print("✓ Embedding model tests passed!")


def test_document_processor():
    """Test document processing and chunking."""
    print("\n=== Testing Document Processor ===")
    
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    # Test text chunking
    text = "This is a test. " * 50  # Create a longer text
    chunks = processor.chunk_text(text)
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)
    print(f"✓ Text chunking: created {len(chunks)} chunks")
    
    # Test metadata in chunks
    metadata = {'filename': 'test.pdf', 'page': 1}
    chunks = processor.chunk_text(text, metadata)
    
    assert all('filename' in chunk['metadata'] for chunk in chunks)
    assert all(chunk['metadata']['filename'] == 'test.pdf' for chunk in chunks)
    print(f"✓ Metadata preserved in chunks")
    
    print("✓ Document processor tests passed!")


def test_vector_store():
    """Test vector store operations."""
    print("\n=== Testing Vector Store ===")
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=tmpdir,
            embedding_dimension=384
        )
        
        # Test adding documents
        documents = ["First document", "Second document", "Third document"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        metadatas = [{"id": i} for i in range(3)]
        ids = [f"doc_{i}" for i in range(3)]
        
        latency = store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        assert latency > 0
        print(f"✓ Added documents: latency={latency:.2f}ms")
        
        # Test collection count
        count = store.get_collection_count()
        assert count == 3
        print(f"✓ Collection count: {count}")
        
        # Test querying
        query_embedding = [0.15] * 384
        results, latency = store.query(query_embedding, n_results=2)
        
        assert len(results['documents'][0]) <= 2
        assert latency > 0
        print(f"✓ Query: returned {len(results['documents'][0])} results, latency={latency:.2f}ms")
        
        print("✓ Vector store tests passed!")


def test_metrics_tracker():
    """Test metrics tracking."""
    print("\n=== Testing Metrics Tracker ===")
    
    tracker = MetricsTracker()
    
    # Record some metrics
    tracker.record_query({
        'embedding_latency_ms': 10.5,
        'retrieval_latency_ms': 5.2,
        'total_latency_ms': 20.0
    })
    
    tracker.record_query({
        'embedding_latency_ms': 12.0,
        'retrieval_latency_ms': 6.0,
        'total_latency_ms': 22.0
    })
    
    # Test query count
    assert tracker.get_query_count() == 2
    print(f"✓ Query count: {tracker.get_query_count()}")
    
    # Test memory usage
    memory = tracker.get_memory_usage_mb()
    assert memory > 0
    print(f"✓ Memory usage: {memory:.2f} MB")
    
    # Test aggregate stats
    stats = tracker.get_aggregate_stats()
    assert 'total_latency_ms' in stats
    assert 'mean' in stats['total_latency_ms']
    print(f"✓ Aggregate stats calculated")
    
    # Test formatted display
    formatted = tracker.format_metrics_for_display()
    assert 'Query Count' in formatted
    print(f"✓ Formatted metrics: {len(formatted)} fields")
    
    print("✓ Metrics tracker tests passed!")


def test_rag_pipeline_initialization():
    """Test RAG pipeline initialization."""
    print("\n=== Testing RAG Pipeline Initialization ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = RAGPipeline(
            embedding_model_name="all-MiniLM-L6-v2",
            persist_directory=tmpdir
        )
        
        assert pipeline.embedding_model is not None
        assert pipeline.document_processor is not None
        assert pipeline.vector_store is not None
        assert pipeline.metrics_tracker is not None
        print("✓ All pipeline components initialized")
        
        # Test collection info
        info = pipeline.get_collection_info()
        assert 'count' in info
        print(f"✓ Collection info: {info}")
        
        print("✓ RAG pipeline initialization tests passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running RAG Pipeline Tests")
    print("=" * 50)
    
    try:
        test_embedding_model()
        test_document_processor()
        test_vector_store()
        test_metrics_tracker()
        test_rag_pipeline_initialization()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

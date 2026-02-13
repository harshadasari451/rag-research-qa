"""
Performance metrics tracking module.
Tracks latency, throughput, memory usage, and other performance indicators.
"""

import time
import psutil
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np


class MetricsTracker:
    """Tracks and aggregates performance metrics for the RAG system."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.query_metrics = []
        self.start_time = time.time()
        self.process = psutil.Process()
        
    def record_query(self, metrics: Dict[str, float]):
        """
        Record metrics for a single query.
        
        Args:
            metrics: Dictionary containing metric values
                    (embedding_latency_ms, retrieval_latency_ms, 
                     generation_latency_ms, total_latency_ms, etc.)
        """
        metrics['timestamp'] = time.time()
        self.query_metrics.append(metrics)
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_query_count(self) -> int:
        """Get total number of queries processed."""
        return len(self.query_metrics)
    
    def get_throughput(self) -> float:
        """
        Calculate throughput (queries per second).
        
        Returns:
            Queries per second
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        return len(self.query_metrics) / elapsed_time
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """
        Calculate aggregate statistics across all queries.
        
        Returns:
            Dictionary with mean, median, p95, p99 for each metric
        """
        if not self.query_metrics:
            return {}
        
        stats = {}
        
        # Get all metric keys (excluding timestamp)
        metric_keys = [k for k in self.query_metrics[0].keys() if k != 'timestamp']
        
        for key in metric_keys:
            values = [m[key] for m in self.query_metrics if key in m]
            
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'std': np.std(values)
                }
        
        return stats
    
    def get_latest_query_metrics(self) -> Dict[str, float]:
        """Get metrics from the most recent query."""
        if not self.query_metrics:
            return {}
        return self.query_metrics[-1]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all metrics.
        
        Returns:
            Dictionary with current metrics and aggregate statistics
        """
        return {
            'query_count': self.get_query_count(),
            'throughput_qps': self.get_throughput(),
            'memory_usage_mb': self.get_memory_usage_mb(),
            'latest_query': self.get_latest_query_metrics(),
            'aggregate_stats': self.get_aggregate_stats()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.query_metrics = []
        self.start_time = time.time()
    
    def format_metrics_for_display(self) -> Dict[str, str]:
        """
        Format metrics for display in UI.
        
        Returns:
            Dictionary with formatted metric strings
        """
        summary = self.get_summary()
        
        formatted = {
            'Query Count': str(summary['query_count']),
            'Throughput': f"{summary['throughput_qps']:.2f} queries/sec",
            'Memory Usage': f"{summary['memory_usage_mb']:.2f} MB"
        }
        
        # Add latest query metrics if available
        latest = summary.get('latest_query', {})
        if latest:
            if 'embedding_latency_ms' in latest:
                formatted['Embedding Latency'] = f"{latest['embedding_latency_ms']:.2f} ms"
            if 'retrieval_latency_ms' in latest:
                formatted['Retrieval Latency'] = f"{latest['retrieval_latency_ms']:.2f} ms"
            if 'generation_latency_ms' in latest:
                formatted['Generation Latency'] = f"{latest['generation_latency_ms']:.2f} ms"
            if 'total_latency_ms' in latest:
                formatted['Total Latency'] = f"{latest['total_latency_ms']:.2f} ms"
        
        # Add aggregate stats if available
        agg_stats = summary.get('aggregate_stats', {})
        if 'total_latency_ms' in agg_stats:
            stats = agg_stats['total_latency_ms']
            formatted['Avg Total Latency'] = f"{stats['mean']:.2f} ms"
            formatted['P95 Total Latency'] = f"{stats['p95']:.2f} ms"
        
        return formatted

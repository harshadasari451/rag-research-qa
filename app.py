"""
Streamlit App for RAG Research Q&A System.
A user-friendly interface for chatting with research papers.
"""

import streamlit as st
import os
from pathlib import Path
import time
from src.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="RAG Research Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'config' not in st.session_state:
    st.session_state.config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'top_k': 5,
        'chunk_size': 512,
        'chunk_overlap': 50
    }

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")

# Model selection
model_options = {
    'all-MiniLM-L6-v2 (Fast, 384d)': 'all-MiniLM-L6-v2',
    'all-mpnet-base-v2 (Better, 768d)': 'all-mpnet-base-v2'
}
selected_model = st.sidebar.selectbox(
    "Embedding Model",
    options=list(model_options.keys()),
    index=0
)
st.session_state.config['embedding_model'] = model_options[selected_model]

# Retrieval settings
st.sidebar.subheader("Retrieval Settings")
st.session_state.config['top_k'] = st.sidebar.slider(
    "Top-K Chunks",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of relevant chunks to retrieve"
)

st.session_state.config['chunk_size'] = st.sidebar.slider(
    "Chunk Size",
    min_value=256,
    max_value=1024,
    value=512,
    step=128,
    help="Size of text chunks (characters)"
)

st.session_state.config['chunk_overlap'] = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=200,
    value=50,
    step=25,
    help="Overlap between chunks (characters)"
)

# Initialize/Reinitialize pipeline button
if st.sidebar.button("🔄 Initialize/Reinitialize Pipeline", type="primary"):
    with st.spinner("Initializing RAG pipeline..."):
        st.session_state.rag_pipeline = RAGPipeline(
            embedding_model_name=st.session_state.config['embedding_model'],
            chunk_size=st.session_state.config['chunk_size'],
            chunk_overlap=st.session_state.config['chunk_overlap']
        )
        st.sidebar.success("✅ Pipeline initialized!")

# Sidebar - Metrics
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performance Metrics")

if st.session_state.rag_pipeline:
    metrics = st.session_state.rag_pipeline.metrics_tracker.format_metrics_for_display()
    
    for metric_name, metric_value in metrics.items():
        st.sidebar.metric(metric_name, metric_value)
    
    # Collection info
    collection_info = st.session_state.rag_pipeline.get_collection_info()
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Index Information")
    st.sidebar.metric("Documents Indexed", collection_info['count'])
    
    if st.sidebar.button("Reset Metrics"):
        st.session_state.rag_pipeline.reset_metrics()
        st.sidebar.success("Metrics reset!")
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">📚 RAG Research Q&A System</h1>', unsafe_allow_html=True)
st.markdown("### Ask questions about your research papers using AI-powered search")

# Check if pipeline is initialized
if st.session_state.rag_pipeline is None:
    st.info("👈 Please initialize the RAG pipeline using the sidebar button first!")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📄 Document Management", "💬 Chat Interface", "📈 Detailed Metrics"])

# Tab 1: Document Management
with tab1:
    st.header("Document Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Research Papers")
        
        # Show current papers directory
        papers_dir = "./data/papers"
        os.makedirs(papers_dir, exist_ok=True)
        
        existing_pdfs = list(Path(papers_dir).glob("*.pdf"))
        
        if existing_pdfs:
            st.success(f"Found {len(existing_pdfs)} PDF(s) in `{papers_dir}`:")
            for pdf in existing_pdfs:
                st.write(f"- {pdf.name}")
        else:
            st.warning(f"No PDFs found in `{papers_dir}`. Please add PDF files to this directory.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload research papers in PDF format"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(papers_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name}")
        
        # Index documents button
        if st.button("🔍 Index Documents", type="primary", use_container_width=True):
            if not existing_pdfs and not uploaded_files:
                st.error("Please add PDF files before indexing!")
            else:
                with st.spinner("Indexing documents... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading and processing PDFs...")
                    progress_bar.progress(25)
                    
                    indexing_stats = st.session_state.rag_pipeline.index_documents(papers_dir)
                    
                    progress_bar.progress(100)
                    status_text.text("Indexing complete!")
                    
                    if 'error' in indexing_stats:
                        st.error(f"Error: {indexing_stats['error']}")
                    else:
                        st.session_state.indexed = True
                        st.success("✅ Documents indexed successfully!")
                        
                        # Display indexing stats
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Documents", indexing_stats['num_documents'])
                        with col_b:
                            st.metric("Chunks", indexing_stats['num_chunks'])
                        with col_c:
                            st.metric("Total Time", f"{indexing_stats['indexing_time_ms']:.0f} ms")
    
    with col2:
        st.subheader("⚠️ Manage Index")
        
        if st.button("🗑️ Reset Index", type="secondary", use_container_width=True):
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.reset_index()
                st.session_state.indexed = False
                st.session_state.chat_history = []
                st.success("Index reset successfully!")
                st.rerun()

# Tab 2: Chat Interface
with tab2:
    st.header("Chat with Your Research Papers")
    
    if not st.session_state.indexed:
        st.warning("⚠️ Please index documents first in the 'Document Management' tab!")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Question {i+1}:** {chat['question']}")
            st.markdown(f"**Answer:**\n{chat['answer']}")
            
            # Show metrics for this query
            metrics = chat.get('metrics', {})
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.caption(f"⚡ Embed: {metrics.get('embedding_latency_ms', 0):.1f}ms")
                with col2:
                    st.caption(f"🔍 Retrieve: {metrics.get('retrieval_latency_ms', 0):.1f}ms")
                with col3:
                    st.caption(f"✍️ Generate: {metrics.get('generation_latency_ms', 0):.1f}ms")
                with col4:
                    st.caption(f"🕐 Total: {metrics.get('total_latency_ms', 0):.1f}ms")
            
            # Show sources (collapsible)
            if 'sources' in chat and chat['sources']:
                with st.expander(f"📑 View {len(chat['sources'])} Retrieved Sources"):
                    for j, source in enumerate(chat['sources'][:3]):  # Show top 3
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {j+1}</strong> (Similarity: {source['similarity_score']:.3f})<br>
                            <strong>File:</strong> {source['metadata'].get('filename', 'N/A')}<br>
                            <strong>Chunk:</strong> {source['metadata'].get('chunk_id', 'N/A')}<br>
                            <em>{source['text'][:300]}...</em>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Question input
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question:",
            placeholder="What is the main conclusion of the paper?",
            disabled=not st.session_state.indexed
        )
        submit_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
        
        if submit_button and question:
            with st.spinner("Searching and generating answer..."):
                response = st.session_state.rag_pipeline.query(
                    question=question,
                    top_k=st.session_state.config['top_k'],
                    return_sources=True
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response['answer'],
                    'sources': response.get('sources', []),
                    'metrics': response['metrics']
                })
                
                st.rerun()

# Tab 3: Detailed Metrics
with tab3:
    st.header("📈 Detailed Performance Metrics")
    
    if st.session_state.rag_pipeline:
        metrics_summary = st.session_state.rag_pipeline.get_metrics()
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", metrics_summary['query_count'])
        with col2:
            st.metric("Throughput", f"{metrics_summary['throughput_qps']:.2f} q/s")
        with col3:
            st.metric("Memory Usage", f"{metrics_summary['memory_usage_mb']:.2f} MB")
        
        # Latest query metrics
        if metrics_summary['latest_query']:
            st.subheader("Latest Query Breakdown")
            latest = metrics_summary['latest_query']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Embedding", f"{latest.get('embedding_latency_ms', 0):.2f} ms")
            with col2:
                st.metric("Retrieval", f"{latest.get('retrieval_latency_ms', 0):.2f} ms")
            with col3:
                st.metric("Generation", f"{latest.get('generation_latency_ms', 0):.2f} ms")
            with col4:
                st.metric("Total", f"{latest.get('total_latency_ms', 0):.2f} ms")
        
        # Aggregate statistics
        if metrics_summary['aggregate_stats']:
            st.subheader("Aggregate Statistics")
            agg_stats = metrics_summary['aggregate_stats']
            
            for metric_name, stats in agg_stats.items():
                st.write(f"**{metric_name}**")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.caption(f"Mean: {stats['mean']:.2f}")
                with col2:
                    st.caption(f"Median: {stats['median']:.2f}")
                with col3:
                    st.caption(f"P95: {stats['p95']:.2f}")
                with col4:
                    st.caption(f"P99: {stats['p99']:.2f}")
                with col5:
                    st.caption(f"Std: {stats['std']:.2f}")
    else:
        st.info("Initialize the pipeline to see metrics")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>RAG Research Q&A System | Built with Sentence Transformers + ChromaDB + Streamlit</p>
    <p>Free and Open Source - No API Keys Required!</p>
</div>
""", unsafe_allow_html=True)

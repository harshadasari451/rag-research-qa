# 📚 RAG Research Q&A System

A **Retrieval Augmented Generation (RAG) system** for chatting with research papers. This project provides an intuitive interface to upload PDF research papers and ask questions about them using AI-powered semantic search.

## ✨ Features

- 🤖 **Sentence Transformers**: Uses open-source embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2)
- 🗄️ **ChromaDB**: Persistent vector storage for efficient similarity search
- ⚡ **Performance Metrics**: Real-time tracking of latency, throughput, and memory usage
- 🎨 **Streamlit UI**: Clean and intuitive web interface
- 📊 **Source Attribution**: Shows which document chunks were used to generate answers
- ⚙️ **Configurable**: Adjust chunk size, overlap, top-k retrieval, and model selection

## 🏗️ Architecture

The system consists of several key components:

1. **Document Processor** (`src/document_processor.py`): Loads PDFs and splits them into overlapping chunks
2. **Embedding Model** (`src/embeddings.py`): Generates vector embeddings using sentence-transformers
3. **Vector Store** (`src/vector_store.py`): Stores and retrieves embeddings using ChromaDB
4. **RAG Pipeline** (`src/rag_pipeline.py`): Orchestrates the end-to-end question-answering workflow
5. **Metrics Tracker** (`src/metrics.py`): Monitors and reports performance metrics
6. **Streamlit App** (`app.py`): Provides the user interface

### RAG Workflow

```
User Question → Embed Query → Search Vector DB → Retrieve Top-K Chunks → Generate Answer → Display with Sources
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harshadasari451/rag-research-qa.git
   cd rag-research-qa
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Prepare Your Research Papers

Place your PDF research papers in the `data/papers/` directory:

```bash
mkdir -p data/papers
# Copy your PDF files to data/papers/
cp /path/to/your/papers/*.pdf data/papers/
```

### 2. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Using the Application

#### Initialize the Pipeline

1. Click "🔄 Initialize/Reinitialize Pipeline" in the sidebar
2. Adjust configuration settings if needed:
   - **Embedding Model**: Choose between fast (MiniLM) or better quality (mpnet)
   - **Top-K Chunks**: Number of relevant chunks to retrieve (1-10)
   - **Chunk Size**: Size of text chunks in characters (256-1024)
   - **Chunk Overlap**: Overlap between chunks (0-200)

#### Index Documents

1. Go to the "📄 Document Management" tab
2. Upload PDFs or use existing files in `data/papers/`
3. Click "🔍 Index Documents"
4. Wait for indexing to complete (shows progress)

#### Ask Questions

1. Go to the "💬 Chat Interface" tab
2. Type your question in the input box
3. Click "Ask" or press Enter
4. View the answer along with:
   - Source documents with similarity scores
   - Performance metrics (embedding, retrieval, generation latency)

#### View Metrics

1. Check the sidebar for real-time performance metrics
2. Go to "📈 Detailed Metrics" tab for comprehensive statistics:
   - Query count and throughput
   - Memory usage
   - Aggregate statistics (mean, median, P95, P99)

## 📊 Performance Metrics

The system tracks the following metrics:

| Metric | Description |
|--------|-------------|
| **Embedding Latency** | Time to generate query embedding (ms) |
| **Retrieval Latency** | Time to search ChromaDB (ms) |
| **Generation Latency** | Time to generate answer (ms) |
| **Total Latency** | End-to-end query time (ms) |
| **Throughput** | Queries processed per second |
| **Memory Usage** | Current memory consumption (MB) |
| **Documents Indexed** | Number of chunks in vector database |

### Example Performance

On a typical system with `all-MiniLM-L6-v2`:

- **Embedding Latency**: 10-50ms per query
- **Retrieval Latency**: 5-20ms
- **Generation Latency**: 1-5ms
- **Total Latency**: 20-80ms per query
- **Memory Usage**: ~500-1000MB (depending on indexed documents)

## 🧪 Testing

Run the test suite to verify the installation:

```bash
python tests/test_pipeline.py
```

This will test:
- ✅ Embedding generation
- ✅ Document processing and chunking
- ✅ Vector store operations
- ✅ Metrics tracking
- ✅ RAG pipeline initialization

## 📁 Project Structure

```
rag-research-qa/
├── app.py                      # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── embeddings.py          # Sentence-transformers wrapper
│   ├── vector_store.py        # ChromaDB operations
│   ├── document_processor.py  # PDF loading and chunking
│   ├── rag_pipeline.py        # End-to-end RAG logic
│   └── metrics.py             # Performance tracking utilities
├── data/
│   ├── papers/                # Store PDF research papers (git-ignored)
│   └── chroma_db/             # ChromaDB persistent storage (git-ignored)
├── tests/
│   └── test_pipeline.py       # Basic tests for RAG pipeline
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🔧 Configuration Options

### Embedding Models

- **all-MiniLM-L6-v2** (Default)
  - Fast inference
  - 384-dimensional embeddings
  - Good for most use cases

- **all-mpnet-base-v2**
  - Better quality
  - 768-dimensional embeddings
  - Slower but more accurate

### Chunking Strategy

- **Chunk Size**: Controls how much text is in each chunk
  - Smaller chunks: More precise retrieval, more chunks to search
  - Larger chunks: More context, fewer chunks to search

- **Chunk Overlap**: Ensures context isn't lost at chunk boundaries
  - Higher overlap: Better context preservation, more storage
  - Lower overlap: Less redundancy, faster indexing

### Retrieval Parameters

- **Top-K**: Number of chunks to retrieve
  - Higher K: More context, slower generation
  - Lower K: Faster, but might miss relevant info

## 🐛 Troubleshooting

### Issue: "No PDFs found"
**Solution**: Make sure PDF files are in the `data/papers/` directory and have `.pdf` extension.

### Issue: "Out of memory during indexing"
**Solution**: 
- Process fewer documents at once
- Use a smaller embedding model (all-MiniLM-L6-v2)
- Reduce chunk size or increase chunk overlap

### Issue: "ChromaDB connection errors"
**Solution**: 
- Ensure `data/chroma_db/` directory exists and is writable
- Try resetting the index using the "Reset Index" button

### Issue: "Slow embedding generation"
**Solution**:
- Use GPU if available (install torch with CUDA support)
- Switch to a smaller model (all-MiniLM-L6-v2)
- Reduce the number of documents being indexed

### Issue: "Answers are not relevant"
**Solution**:
- Increase top-K to retrieve more chunks
- Try a better embedding model (all-mpnet-base-v2)
- Adjust chunk size for better context
- Ensure PDFs contain extractable text (not scanned images)

## 🔮 Future Improvements

Potential enhancements for this project:

1. **Advanced Generation**: Integrate a local LLM (e.g., Llama, Mistral) for better answer generation
2. **OCR Support**: Add OCR for scanned PDFs using pytesseract
3. **Multi-modal**: Support images and tables from papers
4. **Query Expansion**: Use query rewriting for better retrieval
5. **Re-ranking**: Add a cross-encoder for re-ranking retrieved chunks
6. **Batch Processing**: Support batch queries for benchmarking
7. **Export Features**: Export chat history and metrics
8. **Advanced Filters**: Filter by paper, date, or custom metadata
9. **Semantic Caching**: Cache embeddings for repeated queries
10. **Docker Support**: Containerize the application

## 📚 Learning Resources

To learn more about RAG systems:

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Papers and Tutorials](https://github.com/tongni1975/Awesome-RAG)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the UI framework
- The open-source ML community

---

**Built with ❤️ for the research community**

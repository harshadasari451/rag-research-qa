# 📖 Usage Guide - RAG Research Q&A System

## Quick Start

### Option 1: Using the run script (Recommended)

```bash
chmod +x run.sh
./run.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories
- Launch the Streamlit app

### Option 2: Manual setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/papers data/chroma_db

# Run the app
streamlit run app.py
```

## Step-by-Step Usage

### 1. Prepare Your Research Papers

Add PDF files to the `data/papers/` directory:

```bash
cp ~/Downloads/my_research_paper.pdf data/papers/
```

The system supports:
- ✅ Standard PDF files with extractable text
- ✅ Multiple papers (processes all PDFs in the folder)
- ⚠️ Large PDFs (may take longer to process)

### 2. Launch the Application

```bash
streamlit run app.py
```

The app opens at: `http://localhost:8501`

### 3. Initialize the RAG Pipeline

In the sidebar:

1. Select your preferred **Embedding Model**:
   - **all-MiniLM-L6-v2** (Default): Fast, good quality, 384 dimensions
   - **all-mpnet-base-v2**: Better quality, slower, 768 dimensions

2. Adjust **Retrieval Settings** (optional):
   - **Top-K Chunks**: 5 (recommended range: 3-7)
   - **Chunk Size**: 512 characters (range: 256-1024)
   - **Chunk Overlap**: 50 characters (range: 0-200)

3. Click **"🔄 Initialize/Reinitialize Pipeline"**

Wait for confirmation: ✅ Pipeline initialized!

### 4. Index Your Documents

Go to the **"📄 Document Management"** tab:

1. Verify your PDFs are listed
2. Upload additional PDFs if needed (using the file uploader)
3. Click **"🔍 Index Documents"**

This process will:
- Load all PDFs from `data/papers/`
- Extract and chunk the text
- Generate embeddings for each chunk
- Store everything in ChromaDB

**Time estimates**:
- 1 paper (~10 pages): 10-30 seconds
- 5 papers (~50 pages): 1-2 minutes
- 20 papers (~200 pages): 5-10 minutes

### 5. Ask Questions

Go to the **"💬 Chat Interface"** tab:

1. Type your question in the input box
2. Click **"Ask"** or press Enter
3. View the answer along with:
   - Source documents (which chunks were used)
   - Similarity scores
   - Performance metrics

**Example questions**:
- "What is the main contribution of this paper?"
- "Explain the methodology used in the experiments"
- "What are the key findings?"
- "What future work is suggested?"
- "Compare the approaches discussed in these papers"

### 6. View Retrieved Sources

For each answer, expand **"📑 View Retrieved Sources"** to see:
- Top 3 most relevant chunks
- Similarity scores (0-1, higher is better)
- Source file name and chunk ID
- Snippet of the retrieved text

### 7. Monitor Performance

**Real-time metrics** (sidebar):
- Query Count
- Throughput (queries/second)
- Memory Usage (MB)
- Embedding/Retrieval/Generation Latency
- Total Latency

**Detailed metrics** (📈 Detailed Metrics tab):
- Latest query breakdown
- Aggregate statistics (mean, median, P95, P99)
- Historical performance data

### 8. Adjust Configuration

You can adjust settings at any time:

**When to increase Top-K**:
- Answers are too narrow or missing context
- You want more comprehensive responses

**When to decrease Top-K**:
- Answers are too long or off-topic
- You want faster responses

**When to increase Chunk Size**:
- You need more context per chunk
- Papers have long, complex paragraphs

**When to decrease Chunk Size**:
- You want more precise retrieval
- Papers have short, distinct sections

**When to increase Overlap**:
- Important information spans chunk boundaries
- You're getting incomplete answers

**When to use all-mpnet-base-v2**:
- Quality is more important than speed
- You have a GPU available
- Queries are complex

## Advanced Usage

### Batch Processing

To process multiple queries:

1. Ask questions one by one in the chat interface
2. View chat history (scrolls automatically)
3. Check aggregate metrics in the Detailed Metrics tab

### Resetting the Index

If you:
- Add new papers
- Change chunk settings
- Want to start fresh

Then:
1. Go to Document Management tab
2. Click **"🗑️ Reset Index"**
3. Re-index documents

### Comparing Models

To compare embedding models:

1. Index documents with model A (e.g., MiniLM)
2. Ask questions and note metrics
3. Click "Reset Index"
4. Change to model B (e.g., mpnet)
5. Re-index and ask the same questions
6. Compare quality and performance

### Managing Large Document Sets

For 50+ papers:
- Use MiniLM model (faster)
- Process in batches (10-20 papers at a time)
- Monitor memory usage
- Consider increasing chunk size to reduce total chunks

### Optimizing Performance

**For faster queries**:
- Use MiniLM model
- Reduce Top-K to 3
- Ensure GPU is available (install torch with CUDA)

**For better quality**:
- Use mpnet model
- Increase Top-K to 7
- Reduce chunk size for precision

**For lower memory**:
- Use MiniLM model (384d vs 768d)
- Increase chunk size (fewer total chunks)
- Process fewer documents at once

## Troubleshooting

### "Out of memory during indexing"
- Process fewer PDFs at once
- Use MiniLM model
- Increase chunk size
- Close other applications

### "Slow embedding generation"
- Install torch with CUDA (if you have GPU)
- Use MiniLM model
- Reduce number of documents

### "No relevant results found"
- Increase Top-K
- Check if PDFs contain extractable text
- Try different embedding model
- Adjust chunk size

### "Answers are too vague"
- Increase Top-K
- Reduce chunk size
- Try mpnet model

### "Answers are off-topic"
- Decrease Top-K
- Increase chunk size
- Check retrieved sources

## Tips and Best Practices

1. **Start Small**: Index 2-3 papers first to test
2. **Experiment**: Try different settings to see what works
3. **Check Sources**: Always verify the retrieved chunks
4. **Ask Specific Questions**: More specific = better results
5. **Monitor Metrics**: Watch for performance issues
6. **Save Chat History**: Copy important Q&As (no export yet)
7. **Update Regularly**: Reset and re-index when adding papers

## Getting Help

If you encounter issues:
1. Check the main README.md
2. Review the Troubleshooting section
3. Check console output for errors
4. Verify PDF files are valid
5. Try resetting the index
6. Start with a fresh installation

## Next Steps

After getting comfortable:
1. Experiment with different models
2. Try various chunk sizes
3. Test with different types of papers
4. Monitor and optimize performance
5. Consider contributing improvements

Happy researching! 📚🤖

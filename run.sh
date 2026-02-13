#!/bin/bash

# Quick Start Script for RAG Research Q&A System
# This script helps set up and run the application

echo "=========================================="
echo "RAG Research Q&A System - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/papers
mkdir -p data/chroma_db

# Check for PDFs
PDF_COUNT=$(ls -1 data/papers/*.pdf 2>/dev/null | wc -l)
echo ""
echo "Found $PDF_COUNT PDF(s) in data/papers/"

if [ $PDF_COUNT -eq 0 ]; then
    echo ""
    echo "⚠️  No PDFs found!"
    echo "Please add some PDF research papers to data/papers/ before running."
    echo ""
fi

# Run the app
echo ""
echo "=========================================="
echo "Starting Streamlit application..."
echo "=========================================="
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "To stop the app, press Ctrl+C"
echo ""

streamlit run app.py

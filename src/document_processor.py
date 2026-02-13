"""
Document processor module for PDF loading and text chunking.
"""

import os
from typing import List, Dict, Any
import PyPDF2
from pathlib import Path


class DocumentProcessor:
    """Handles PDF loading, text extraction, and intelligent chunking."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size for each text chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Load a PDF file and extract text from all pages.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with 'text', 'num_pages', and 'filename'
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                text_by_page = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_by_page.append(text)
                
                full_text = "\n\n".join(text_by_page)
                
                return {
                    'text': full_text,
                    'text_by_page': text_by_page,
                    'num_pages': num_pages,
                    'filename': os.path.basename(pdf_path)
                }
        except Exception as e:
            raise Exception(f"Error loading PDF {pdf_path}: {str(e)}")
    
    def load_pdfs_from_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load all PDF files from a directory.
        
        Args:
            directory: Path to directory containing PDF files
            
        Returns:
            List of document dictionaries
        """
        pdf_files = list(Path(directory).glob("*.pdf"))
        documents = []
        
        for pdf_path in pdf_files:
            try:
                doc = self.load_pdf(str(pdf_path))
                documents.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {pdf_path}: {str(e)}")
        
        return documents
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # Only break if we're past halfway
                    end = start + break_point + 1
                    chunk_text = text[start:end]
            
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta['chunk_id'] = chunk_id
            chunk_meta['start_char'] = start
            chunk_meta['end_char'] = end
            
            chunks.append({
                'text': chunk_text.strip(),
                'metadata': chunk_meta
            })
            
            chunk_id += 1
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a document: chunk it and prepare for embedding.
        
        Args:
            doc: Document dictionary from load_pdf
            
        Returns:
            List of chunks with metadata
        """
        base_metadata = {
            'filename': doc['filename'],
            'num_pages': doc['num_pages']
        }
        
        chunks = self.chunk_text(doc['text'], base_metadata)
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks

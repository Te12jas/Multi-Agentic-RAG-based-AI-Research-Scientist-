"""
PDF processing utilities for downloading and extracting text from arXiv papers.
"""

import re
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import requests
import fitz  # PyMuPDF
from config import config


class PDFProcessor:
    """Handles PDF downloading and text extraction."""
    
    # Common section headers in academic papers
    SECTION_PATTERNS = [
        (r"^abstract\s*$", "abstract"),
        (r"^1\.?\s*introduction", "introduction"),
        (r"^2\.?\s*(?:related\s+work|background)", "related_work"),
        (r"^\d+\.?\s*(?:method|approach|methodology)", "methods"),
        (r"^\d+\.?\s*(?:experiment|evaluation|results)", "experiments"),
        (r"^\d+\.?\s*(?:discussion)", "discussion"),
        (r"^\d+\.?\s*(?:conclusion|summary)", "conclusion"),
        (r"^\d+\.?\s*(?:limitation)", "limitations"),
        (r"^(?:references|bibliography)", "references"),
        (r"^(?:appendix|supplementary)", "appendix"),
    ]
    
    def __init__(self):
        """Initialize PDF processor."""
        self.temp_dir = tempfile.mkdtemp()
    
    def download_pdf(self, url: str, paper_id: str) -> Optional[str]:
        """
        Download PDF from URL.
        
        Args:
            url: PDF URL (typically arXiv)
            paper_id: Paper identifier for filename
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        try:
            # Handle arXiv URLs
            if "arxiv.org" in url:
                # Convert abstract URL to PDF URL if needed
                if "/abs/" in url:
                    url = url.replace("/abs/", "/pdf/") + ".pdf"
                elif not url.endswith(".pdf"):
                    url = url + ".pdf"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to temp file
            pdf_path = os.path.join(self.temp_dir, f"{paper_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            
            return pdf_path
            
        except Exception as e:
            print(f"Error downloading PDF {url}: {e}")
            return None
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page in doc:
                text_parts.append(page.get_text())
            
            doc.close()
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def identify_section(self, text: str) -> str:
        """
        Identify the section type from text.
        
        Args:
            text: Text snippet (typically first line of chunk)
            
        Returns:
            Section name or 'body'
        """
        text_lower = text.lower().strip()
        
        for pattern, section_name in self.SECTION_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return section_name
        
        return "body"
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with section awareness.
        
        Args:
            text: Full document text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dicts with 'text' and 'section'
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        # Split into paragraphs first
        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_section = "abstract"  # Default start
        
        for para in paragraphs:
            # Check if this paragraph starts a new section
            first_line = para.split("\n")[0] if para else ""
            detected_section = self.identify_section(first_line)
            if detected_section != "body":
                current_section = detected_section
            
            # Skip references and appendix for now
            if current_section in ["references", "appendix"]:
                continue
            
            # Add paragraph to current chunk
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": current_section
                    })
                
                # Start new chunk with overlap
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "section": current_section
            })
        
        return chunks
    
    def process_paper(
        self,
        url: str,
        paper_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Download and process a paper into chunks.
        
        Args:
            url: PDF URL
            paper_id: Paper identifier
            
        Returns:
            Tuple of (full_text, list of chunks)
        """
        # Download PDF
        pdf_path = self.download_pdf(url, paper_id)
        if not pdf_path:
            return "", []
        
        # Extract text
        full_text = self.extract_text(pdf_path)
        if not full_text:
            return "", []
        
        # Chunk text
        chunks = self.chunk_text(full_text)
        
        # Clean up
        try:
            os.remove(pdf_path)
        except:
            pass
        
        return full_text, chunks
    
    def cleanup(self):
        """Clean up temporary directory."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass


# Global processor instance
pdf_processor = PDFProcessor()

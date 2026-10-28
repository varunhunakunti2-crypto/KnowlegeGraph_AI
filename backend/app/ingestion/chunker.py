"""Text chunking with configurable size and overlap."""
from typing import List, Dict
import re


class TextChunker:
    """Chunks text preserving paragraph boundaries where practical."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        # A zero/negative step would hang or raise in the split loop below.
        chunk_size = max(64, int(chunk_size))
        chunk_overlap = max(0, int(chunk_overlap))
        self.chunk_size = chunk_size
        # Keep the split step at least half the window so overlap cannot
        # explode the number of tiny chunks (or make the step zero).
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
    
    def chunk_text(self, text: str, page_number: int) -> List[Dict]:
        """
        Chunk text with overlap, preserving paragraph boundaries.
        Never combines text from unrelated documents.
        """
        if not text or not text.strip():
            return []
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [{
                "chunk_index": 0,
                "chunk_text": text.strip(),
                "page_number": page_number,
                "char_count": len(text.strip())
            }]
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk_parts = []
        current_length = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If paragraph alone exceeds chunk size, split it
            if para_length > self.chunk_size and not current_chunk_parts:
                # Split long paragraph
                for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                    chunk_text = para[i:i + self.chunk_size]
                    if chunk_text.strip():
                        chunks.append({
                            "chunk_index": chunk_index,
                            "chunk_text": chunk_text.strip(),
                            "page_number": page_number,
                            "char_count": len(chunk_text.strip())
                        })
                        chunk_index += 1
                continue
            
            # If adding this paragraph would exceed chunk size
            if current_length + para_length > self.chunk_size and current_chunk_parts:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk_parts)
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "page_number": page_number,
                    "char_count": len(chunk_text)
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk_parts:
                    overlap_text = current_chunk_parts[-1]
                    if len(overlap_text) > self.chunk_overlap:
                        overlap_text = overlap_text[-self.chunk_overlap:]
                    current_chunk_parts = [overlap_text] if overlap_text.strip() else []
                    current_length = len(overlap_text)
                else:
                    current_chunk_parts = []
                    current_length = 0
            
            current_chunk_parts.append(para)
            current_length += para_length + 2  # Account for \n\n
        
        # Add remaining chunk
        if current_chunk_parts:
            chunk_text = '\n\n'.join(current_chunk_parts)
            if chunk_text.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "page_number": page_number,
                    "char_count": len(chunk_text)
                })
        
        return chunks
    
    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """Chunk multiple pages maintaining page mapping."""
        all_chunks = []
        global_chunk_index = 0
        
        for page in pages:
            page_chunks = self.chunk_text(page["text"], page["page_number"])
            for chunk in page_chunks:
                chunk["chunk_index"] = global_chunk_index
                all_chunks.append(chunk)
                global_chunk_index += 1
        
        return all_chunks

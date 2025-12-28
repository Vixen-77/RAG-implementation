
import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .pdf_processor import detect_vehicle_model


CHILD_CHUNK_SIZE = 2400  # ~600 tokens
CHILD_CHUNK_OVERLAP = 400  # ~100 tokens


def create_parent_chunks(pages: List[Dict[str, Any]], filename: str, file_hash: str) -> List[Document]:
    """Create large parent chunks (full sections) for context retrieval.
    
    These parent chunks store complete sections that provide full context
    when child chunks are retrieved during vector search.
    """
    parents = []
    parent_id_counter = 0
    
    full_text = "\n\n".join([p["text"] for p in pages])
    vehicle_model = detect_vehicle_model(filename)
    sections = _split_by_headers(full_text, pages)
    
    for section_title, section_text, section_code, page_numbers in sections:
        if len(section_text.strip()) < 100:
            continue
        
        parent_id = f"{file_hash[:8]}_parent_{parent_id_counter}"
        parent_id_counter += 1
        
        parent_doc = Document(
            page_content=section_text,
            metadata={
                "parent_id": parent_id,
                "type": "parent",
                "source_file": filename,  # Document Name
                "file_hash": file_hash,
                "chapter": section_title,  # Chapter header
                "section_title": section_title,
                "section_code": section_code,
                "page_numbers": page_numbers,  # Page Number(s)
                "vehicle_model": vehicle_model,
                "char_count": len(section_text)
            }
        )
        parents.append(parent_doc)
    
    print(f"[INFO] Created {len(parents)} parent chunks (with component_type metadata)")
    return parents

def _split_by_headers(text: str, pages: List[Dict[str, Any]] = None) -> List[Tuple[str, str, str, str]]:
    """Split text by markdown-style headers and track page numbers.
    
    Returns: List of (section_title, section_text, section_code, page_numbers)
    """
    header_pattern = r'^(\d{1,2}\.?\s*)?([A-Z][a-zA-Z\s\-\&]{3,})$'
    
    # Build a mapping of content to page numbers
    content_to_page = {}
    if pages:
        for page in pages:
            page_num = page.get("page_num", 0)
            page_text = page.get("text", "")
            for line in page_text.split('\n'):
                line_stripped = line.strip()
                if line_stripped:
                    content_to_page[line_stripped] = page_num
    
    lines = text.split('\n')
    sections = []
    current_title = "General"
    current_section = []
    current_code = "unknown"
    current_pages = set()
    last_header = ""
    
    for line in lines:
        line_stripped = line.strip()

        if line_stripped.isdigit():
            continue
        
        # Track page number for this line
        if line_stripped in content_to_page:
            current_pages.add(content_to_page[line_stripped])
        
        match = re.match(header_pattern, line_stripped)
        if match and len(line_stripped) < 60: 
            new_title = match.group(2).strip()
            if new_title == last_header:
                continue 
            if current_section:
                page_str = ",".join(map(str, sorted(current_pages))) if current_pages else "unknown"
                sections.append((current_title, "\n".join(current_section), current_code, page_str))
            current_code = match.group(1).strip() if match.group(1) else "unknown"
            current_title = new_title
            current_section = []
            current_pages = set()
            last_header = new_title
        else:
            if line_stripped:
                current_section.append(line)
    
    if current_section:
        page_str = ",".join(map(str, sorted(current_pages))) if current_pages else "unknown"
        sections.append((current_title, "\n".join(current_section), current_code, page_str))
    
    return sections

def create_child_chunks(parent_docs: List[Document]) -> List[Document]:
    
    children = []
    
    # Recursive splitting with markdown-aware separators
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,  # ~600 tokens
        chunk_overlap=CHILD_CHUNK_OVERLAP,  # ~100 tokens
        separators=[
            "\n## ",  # Markdown H2 headers
            "\n### ",  # Markdown H3 headers  
            "\n#### ",  # Markdown H4 headers
            "\n\n",  # Double newlines (paragraphs)
            "\n",  # Single newlines
            ". ",  # Sentences
            "; ",  # Semi-colon separated
            ", ",  # Comma separated
            " ",  # Words
            ""  
        ]
    )
    
    for parent in parent_docs:
        parent_id = parent.metadata["parent_id"]
        child_texts = child_splitter.split_text(parent.page_content)
        
        for child_idx, child_text in enumerate(child_texts):
            child_doc = Document(
                page_content=child_text,
                metadata={
                    **parent.metadata,  
                    "parent_id": parent_id,
                    "type": "child",
                    "child_index": child_idx,
                    "chunk_id": f"{parent_id}_child_{child_idx}",
                    "char_count": len(child_text),
                    "approx_tokens": len(child_text) // 4  
                }
            )
            children.append(child_doc)
    
    avg_size = sum(len(c.page_content) for c in children) / len(children) if children else 0
    print(f"[INFO] Created {len(children)} child chunks (avg: {avg_size:.0f} chars, ~{avg_size/4:.0f} tokens)")
    return children

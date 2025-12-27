
import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .pdf_processor import detect_vehicle_model

def create_parent_chunks(pages: List[Dict[str, Any]], filename: str, file_hash: str) -> List[Document]:
    parents = []
    parent_id_counter = 0
    
    full_text = "\n\n".join([p["text"] for p in pages])
    vehicle_model = detect_vehicle_model(filename)
    sections = _split_by_headers(full_text)
    
    for section_title, section_text, section_code in sections:
        if len(section_text.strip()) < 100:
            continue
        
        parent_id = f"{file_hash[:8]}_parent_{parent_id_counter}"
        parent_id_counter += 1
        
        parent_doc = Document(
            page_content=section_text,
            metadata={
                "parent_id": parent_id,
                "type": "parent",
                "source_file": filename,
                "file_hash": file_hash,
                "section_title": section_title,
                "section_code": section_code,
                "vehicle_model": vehicle_model,
                "char_count": len(section_text)
            }
        )
        parents.append(parent_doc)
    
    print(f"[INFO] Created {len(parents)} parent chunks")
    return parents

def _split_by_headers(text: str) -> List[Tuple[str, str, str]]:
    header_pattern = r'^([0-9]{1,2}[A-Z]?\s+)?([A-Z][A-Z\s]{10,})$'
    lines = text.split('\n')
    sections = []
    current_title = "General"
    current_section = []
    current_code = "unknown"
    
    for line in lines:
        line_stripped = line.strip()
        match = re.match(header_pattern, line_stripped)
        
        if match and len(line_stripped) < 100:
            if current_section:
                sections.append((current_title, "\n".join(current_section), current_code))
            
            code_part = match.group(1) or ""
            current_code = code_part.strip() if code_part else "unknown"
            current_title = line_stripped
            current_section = []
        else:
            if line_stripped:
                current_section.append(line)
    
    if current_section:
        sections.append((current_title, "\n".join(current_section), current_code))
    
    return sections

def create_child_chunks(parent_docs: List[Document]) -> List[Document]:
    children = []
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
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
                    "char_count": len(child_text)
                }
            )
            children.append(child_doc)
    
    print(f"[INFO] Created {len(children)} child chunks")
    return children

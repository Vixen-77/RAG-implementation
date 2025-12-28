
import fitz  
from typing import List, Dict, Any

def extract_text_pages(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        page_dict = page.get_text("dict", sort=True)
        blocks = page_dict.get("blocks", [])
        
        page_text = []
        for block in blocks:
            if "lines" not in block:
                continue
            
            block_text = _extract_block_text(block)
            if len(block_text) > 20:  # Filter noise
                page_text.append(block_text)
        
        pages.append({
            "page_num": page_num + 1,
            "text": "\n\n".join(page_text),
            "blocks": len(page_text)
        })
    
    doc.close()
    print(f"[INFO] Extracted {len(pages)} pages")
    return pages

def _extract_block_text(block: Dict) -> str:
    spans_text = [
        span["text"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    return " ".join(spans_text).strip()

def detect_vehicle_model(filename: str) -> str:
    filename_lower = filename.lower()
    if "duster" in filename_lower:
        return "Dacia Duster"
    elif "logan" in filename_lower:
        return "Dacia Logan"
    elif "sandero" in filename_lower:
        return "Dacia Sandero"
    return "Dacia General"



import os
import hashlib
from typing import Dict, Any, List
from langchain_core.documents import Document

# Sub-modules
from .pdf_processor import extract_text_pages
from .chunking import create_parent_chunks, create_child_chunks
from .vision import process_images
from services.storage.vector import vector_db
from services.storage.document import get_docstore

class MultimodalIngestionPipeline:
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.docstore = get_docstore()
        self.vectorstore = vector_db
        print(f"[INFO] Pipeline initialized ({persist_dir})")
    
    def ingest_pdf(self, pdf_path: str, force: bool = False) -> Dict[str, Any]:
        filename = os.path.basename(pdf_path)
        print(f"\n[INFO] Processing: {filename}")
        
        file_hash = self._compute_file_hash(pdf_path)
        if not force and self._is_document_indexed(file_hash):
            print(f"[INFO] Skipping {filename} (Already Indexed)")
            return {"status": "skipped", "reason": "duplicate_hash"}
        
        try:
            text_pages = extract_text_pages(pdf_path)
            
            parent_docs = create_parent_chunks(text_pages, filename, file_hash)
            
            child_docs = create_child_chunks(parent_docs)
            
            image_docs = process_images(pdf_path, filename, file_hash)
            
            self._store_documents(parent_docs, child_docs, image_docs)
            
            print(f"[INFO] Ingestion complete (P:{len(parent_docs)}, C:{len(child_docs)}, I:{len(image_docs)})")
            
            return {
                "status": "success",
                "parents": len(parent_docs),
                "children": len(child_docs),
                "images_captioned": len(image_docs)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _store_documents(self, parents: List[Document], children: List[Document], images: List[Document]):
        for parent in parents:
            self.docstore.add_document(parent.metadata["parent_id"], parent)
        
        all_children = children + images
        if all_children:
            self.vectorstore.add_documents(all_children)

    def get_stats(self) -> Dict[str, Any]:
        try:
            return {
                "children_count": self.vectorstore._collection.count(),
                "parents_count": len(self.docstore),
                "persist_dir": self.persist_dir
            }
        except Exception:
            return {"children_count": 0, "parents_count": 0}

    def _compute_file_hash(self, file_path: str) -> str:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for b in iter(lambda: f.read(4096), b""):
                sha.update(b)
        return sha.hexdigest()
    
    def _is_document_indexed(self, file_hash: str) -> bool:
        try:
            res = self.vectorstore.get(where={"file_hash": file_hash}, limit=1)
            return bool(res and res['ids'])
        except:
            return False
import os
import hashlib
from typing import Dict, Any, List
from langchain_core.documents import Document

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
            print(f"[DEBUG] Extracted {len(text_pages)} text pages")
            
            parent_docs = create_parent_chunks(text_pages, filename, file_hash)
            print(f"[DEBUG] Created {len(parent_docs) if parent_docs else 0} parent docs")
            
            child_docs = create_child_chunks(parent_docs)
            print(f"[DEBUG] child_docs type: {type(child_docs)}, count: {len(child_docs) if child_docs else 'None'}")
            
            image_docs = process_images(pdf_path, filename, file_hash)
            print(f"[DEBUG] image_docs type: {type(image_docs)}, count: {len(image_docs) if image_docs else 'None'}")
            
            # Ensure we have lists
            if child_docs is None:
                child_docs = []
                print("[WARNING] child_docs was None, using empty list")
            if image_docs is None:
                image_docs = []
                print("[WARNING] image_docs was None, using empty list")
            
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
            BATCH_SIZE = 5000  
            
            for i in range(0, len(all_children), BATCH_SIZE):
                batch = all_children[i:i + BATCH_SIZE]
                print(f"[INFO] Adding batch {i//BATCH_SIZE + 1} ({len(batch)} documents)")
                self.vectorstore.add_documents(batch)
            
            print(f"[INFO] Successfully added {len(all_children)} documents to vector store")

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
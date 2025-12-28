
import os
import json
from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class ParentDocumentStore:
   
    
    _instance: Optional['ParentDocumentStore'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_dir: str = "./chroma_parent_child"):
        if not self._initialized:
            self.persist_dir = persist_dir
            self.persist_path = os.path.join(persist_dir, "docstore.json")
            self.store: Dict[str, Document] = {}
            
            os.makedirs(self.persist_dir, exist_ok=True)
            
            self._load()
            
            self._initialized = True
            print(f" [DocStore] Parent document store initialized ({len(self.store)} documents)")
    
    def _load(self):
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        self.store[doc_id] = Document(
                            page_content=doc_data.get('page_content', ''),
                            metadata=doc_data.get('metadata', {})
                        )
                print(f"    [DocStore] Loaded {len(self.store)} parents from {self.persist_path}")
            except Exception as e:
                print(f"    [DocStore] Failed to load persistence file: {e}")
    
    def _save(self):
        try:
            data = {
                doc_id: {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc_id, doc in self.store.items()
            }
            
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"    [DocStore] Failed to save persistence file: {e}")
    
    def add_document(self, doc_id: str, document: Document):
        
        self.store[doc_id] = document
        self._save()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        
        return self.store.get(doc_id)
    
    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        
        return [self.store[doc_id] for doc_id in doc_ids if doc_id in self.store]
    
    def delete_by_file_hash(self, file_hash: str) -> int:
        
        to_delete = [
            doc_id for doc_id, doc in self.store.items()
            if doc.metadata.get("file_hash") == file_hash
        ]
        
        for doc_id in to_delete:
            del self.store[doc_id]
        
        if to_delete:
            self._save()
        
        return len(to_delete)
    
    def clear(self):
        count = len(self.store)
        self.store.clear()
        self._save()
        print(f"  [DocStore] Cleared {count} parent documents")
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "total_parents": len(self.store),
            "total_chars": sum(len(doc.page_content) for doc in self.store.values()),
            "avg_chars": sum(len(doc.page_content) for doc in self.store.values()) // len(self.store) if self.store else 0,
            "persist_path": self.persist_path
        }
    
    def __len__(self):
        return len(self.store)
    
    def __repr__(self):
        return f"<ParentDocumentStore: {len(self.store)} documents (persisted at {self.persist_path})>"


_docstore = ParentDocumentStore()


def get_docstore() -> ParentDocumentStore:
    
    return _docstore
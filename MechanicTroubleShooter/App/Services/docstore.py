"""
Shared Parent Document Store
=============================

This module provides a singleton document store that can be shared
between the ingestion pipeline and the RAG retrieval system.

The docstore holds full parent chunks in memory and can be accessed
by both the ingest and rag modules.
"""

import os
import json
from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class ParentDocumentStore:
    """
    In-memory document store for parent chunks with JSON persistence.
    
    This is a singleton pattern - there's only one instance shared
    across the application.
    """
    
    _instance: Optional['ParentDocumentStore'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_dir: str = "./chroma_parent_child"):
        # Only initialize once (singleton pattern)
        if not self._initialized:
            self.persist_dir = persist_dir
            self.persist_path = os.path.join(persist_dir, "docstore.json")
            self.store: Dict[str, Document] = {}
            
            # Ensure directory exists
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Load existing data
            self._load()
            
            self._initialized = True
            print(f"📚 [DocStore] Parent document store initialized ({len(self.store)} documents)")
    
    def _load(self):
        """Load documents from JSON file."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        # reconstruct Document objects
                        self.store[doc_id] = Document(
                            page_content=doc_data.get('page_content', ''),
                            metadata=doc_data.get('metadata', {})
                        )
                print(f"   📂 [DocStore] Loaded {len(self.store)} parents from {self.persist_path}")
            except Exception as e:
                print(f"   ⚠️ [DocStore] Failed to load persistence file: {e}")
    
    def _save(self):
        """Save documents to JSON file."""
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
            print(f"   ❌ [DocStore] Failed to save persistence file: {e}")
    
    def add_document(self, doc_id: str, document: Document):
        """
        Store a parent document by ID.
        
        Args:
            doc_id: Unique identifier for the parent document
            document: LangChain Document object containing the parent content
        """
        self.store[doc_id] = document
        self._save()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a parent document by ID.
        
        Args:
            doc_id: Parent document identifier
            
        Returns:
            Document object if found, None otherwise
        """
        return self.store.get(doc_id)
    
    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """
        Retrieve multiple parent documents.
        
        Args:
            doc_ids: List of parent document identifiers
            
        Returns:
            List of Document objects (only found documents)
        """
        return [self.store[doc_id] for doc_id in doc_ids if doc_id in self.store]
    
    def delete_by_file_hash(self, file_hash: str) -> int:
        """
        Delete all parent documents from a specific file.
        
        Args:
            file_hash: Hash of the source file
            
        Returns:
            Number of documents deleted
        """
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
        """Clear all stored documents."""
        count = len(self.store)
        self.store.clear()
        self._save()
        print(f"🗑️  [DocStore] Cleared {count} parent documents")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about stored documents."""
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


# Global singleton instance
_docstore = ParentDocumentStore()


def get_docstore() -> ParentDocumentStore:
    """
    Get the global document store instance.
    
    This ensures both ingest.py and rag.py use the same store.
    
    Returns:
        Shared ParentDocumentStore instance
    """
    return _docstore
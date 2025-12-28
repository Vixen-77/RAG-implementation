from typing import List, Dict, Any, Set
from langchain_core.documents import Document

from services.storage.vector import vector_db
from services.retrieval.reranker import rerank_results
from services.llm.client import generate_chat_answer
from services.storage.document import get_docstore


class ParentChildRAG:
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.vectorstore = vector_db
        self.docstore = get_docstore()
        
        count = self.vectorstore._collection.count()
        print(f"[RAG] Initialized. Children: {count}. Parents: {len(self.docstore)}")
    
    def query(self, user_question: str, k: int = 10, child_k: int = 50) -> Dict[str, Any]:
        print(f"\n[QUERY] {user_question}")
        
        is_visual = self._is_visual_query(user_question)
        search_k = (child_k * 3) + 10 if is_visual else (child_k * 3)
        
        child_docs = self._search_children(user_question, k=search_k, include_images=is_visual)
        
        if not child_docs:
            return self._no_results_response()
            
        print(f"[RAG] Found {len(child_docs)} children from vector search")
        
        unique_docs = self._deduplicate_aggressively(child_docs)
        print(f"[RAG] After deduplication: {len(unique_docs)} unique children")
        
        candidates = unique_docs[:child_k]
        print(f"[RAG] Passing {len(candidates)} unique candidates to reranker")
        
        try:
            reranked = rerank_results(user_question, candidates, top_k=k)
        except Exception as e:
            print(f"[WARN] Rerank failed: {e}")
            reranked = candidates[:k]
        
        print(f"[RAG] Final sources after reranking: {len(reranked)}")
        
        # 5. Generate
        context = self._build_context(reranked)
        answer = generate_chat_answer(context, user_question, {"strategy": "child_direct"})
        
        print(f"[RAG] Answer generated ({len(answer)} chars)")
        
        return {
            "answer": answer,
            "sources": reranked,
            "num_sources": len(reranked),
            "context_chars": len(context),
            "formatted_sources": self._format_sources(reranked)
        }
    
    def _deduplicate_aggressively(self, docs: List[Document]) -> List[Document]:
       
        if not docs:
            return []
        
        unique_docs = []
        seen_content: Set[str] = set()
        seen_parents: Set[str] = set()
        similar_contents: List[str] = []
        
        for doc in docs:
            content = doc.page_content.strip()
            parent_id = doc.metadata.get("parent_id")
            
            # Skip if exact duplicate
            if content in seen_content:
                continue
            
            is_similar = False
            if len(content) < 100:  # Only check similarity for short chunks
                for existing in similar_contents:
                    if self._is_too_similar(content, existing):
                        is_similar = True
                        break
            
            if is_similar:
                continue
            

            if parent_id:
                parent_count = sum(1 for d in unique_docs if d.metadata.get("parent_id") == parent_id)
                if parent_count >= 2:  
                    continue
            
            seen_content.add(content)
            if len(content) < 100:
                similar_contents.append(content)
            if parent_id:
                seen_parents.add(parent_id)
            
            unique_docs.append(doc)
        
        removed = len(docs) - len(unique_docs)
        if removed > 0:
            
        
         return unique_docs
    
    def _is_too_similar(self, content1: str, content2: str, threshold: float = 0.85) -> bool:
        if content1 == content2:
            return True
        
        if len(content1) < 50 or len(content2) < 50:
            longer = content1 if len(content1) > len(content2) else content2
            shorter = content2 if len(content1) > len(content2) else content1
            return shorter in longer
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        smaller_set = min(len(words1), len(words2))
        
        similarity = intersection / smaller_set if smaller_set > 0 else 0
        return similarity >= threshold
    
    def _is_visual_query(self, query: str) -> bool:
        keywords = ["show", "diagram", "picture", "image", "photo", "location", "look like", "see"]
        return any(k in query.lower() for k in keywords)

    def _search_children(self, query: str, k: int = 20, include_images: bool = False) -> List[Document]:
        filter_dict = {"type": "child"} if not include_images else None
        try:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
    
    def _build_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            src = f"[Source {i+1}, {doc.metadata.get('section_title', 'Unknown')}]"
            parts.append(f"{src}\n{doc.page_content}")
        return "\n\n".join(parts)
    
    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        return [{
            "source_num": i+1,
            "title": d.metadata.get("section_title"),
            "file": d.metadata.get("source_file"),
            "preview": d.page_content[:200]
        } for i, d in enumerate(docs)]
    
    def _no_results_response(self):
        return {"answer": "No relevant info found.", "sources": []}
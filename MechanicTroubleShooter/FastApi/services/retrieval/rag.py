"""
3-Stage Retrieval Pipeline for Car Manual RAG

Stage 1: Hybrid Search (Vector + BM25 with RRF fusion)
Stage 2: Parent-Child Context Retrieval
Stage 3: Cross-Encoder Reranking
"""

from typing import List, Dict, Any, Set, Optional
from langchain_core.documents import Document

from services.storage.vector import vector_db
from services.retrieval.reranker import rerank_results
from services.retrieval.hybrid_search import hybrid_search
from services.llm.client import generate_chat_answer
from services.storage.document import get_docstore


class ParentChildRAG:
    """
    3-Stage Retrieval Pipeline:
    
    1. Hybrid Search: Combines vector (semantic) + BM25 (keyword) search
       - Handles both "how do I fix overheating?" and "DF025 fault code"
       
    2. Parent-Child Retrieval: Uses small child chunks for search accuracy,
       then retrieves parent sections for complete context
       
    3. Reranking: Cross-encoder reorders results by true relevance
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.vectorstore = vector_db
        self.docstore = get_docstore()
        
        count = self.vectorstore._collection.count()
        print(f"[RAG] 3-Stage Pipeline Initialized. Children: {count}. Parents: {len(self.docstore)}")
    
    def query(self, user_question: str, k: int = 10, child_k: int = 50, 
              use_parent_context: bool = True) -> Dict[str, Any]:
        """
        Execute 3-stage retrieval pipeline.
        
        Args:
            user_question: The user's query
            k: Number of final results to return
            child_k: Number of candidates for reranking
            use_parent_context: Whether to fetch parent context (Stage 2)
        """
        print(f"\n[QUERY] {user_question}")
        print("[RAG] Stage 1: Hybrid Search (Vector + BM25)")
        
        is_visual = self._is_visual_query(user_question)
        
        # Stage 1: Hybrid Search (Vector + BM25 with RRF fusion)
        child_docs = hybrid_search(
            user_question, 
            k=child_k * 2,  # Get extra for deduplication
            include_images=is_visual
        )
        
        if not child_docs:
            return self._no_results_response()
            
        print(f"[RAG] Hybrid search returned {len(child_docs)} children")
        
        # Deduplicate
        unique_docs = self._deduplicate_aggressively(child_docs)
        print(f"[RAG] After deduplication: {len(unique_docs)} unique children")
        
        candidates = unique_docs[:child_k]
        
        # Stage 3: Reranking (before parent retrieval for efficiency)
        print(f"[RAG] Stage 3: Reranking {len(candidates)} candidates")
        try:
            reranked = rerank_results(user_question, candidates, top_k=k)
        except Exception as e:
            print(f"[WARN] Rerank failed: {e}")
            reranked = candidates[:k]
        
        print(f"[RAG] Top {len(reranked)} results after reranking")
        
        # Stage 2: Parent Context Retrieval
        if use_parent_context:
            print("[RAG] Stage 2: Retrieving parent context")
            context = self._build_context_with_parents(reranked)
        else:
            context = self._build_context(reranked)
        
        # Generate answer
        answer = generate_chat_answer(context, user_question, {"strategy": "3_stage_rag"})
        
        print(f"[RAG] Answer generated ({len(answer)} chars)")
        
        return {
            "answer": answer,
            "sources": reranked,
            "num_sources": len(reranked),
            "context_chars": len(context),
            "formatted_sources": self._format_sources(reranked),
            "pipeline": "3-stage (hybrid + parent + rerank)"
        }
    
    def _build_context_with_parents(self, child_docs: List[Document]) -> str:
        """
        Build context by fetching parent documents for complete sections.
        
        When a child chunk like "Tighten to 20Nm" is found, this retrieves
        the full parent section to ensure no safety steps are missed.
        """
        seen_parents: Set[str] = set()
        context_parts = []
        
        for i, child in enumerate(child_docs):
            parent_id = child.metadata.get("parent_id")
            section_title = child.metadata.get("section_title", "Unknown")
            
            if parent_id and parent_id not in seen_parents:
                parent_doc = self.docstore.get_document(parent_id)
                
                if parent_doc:
                    seen_parents.add(parent_id)
                    src = f"[Source {i+1}: {section_title} (Full Section)]"
                    context_parts.append(f"{src}\n{parent_doc.page_content}")
                    print(f"    [Parent] Retrieved: {section_title} ({len(parent_doc.page_content)} chars)")
                else:
                    src = f"[Source {i+1}: {section_title}]"
                    context_parts.append(f"{src}\n{child.page_content}")
            elif parent_id in seen_parents:
                continue
            else:
                src = f"[Source {i+1}: {section_title}]"
                context_parts.append(f"{src}\n{child.page_content}")
        
        context = "\n\n".join(context_parts)
        print(f"[RAG] Built context: {len(context)} chars from {len(seen_parents)} parent sections")
        return context
    
    def _deduplicate_aggressively(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        
        unique_docs = []
        seen_content: Set[str] = set()
        seen_normalized: Set[str] = set()  
        
        for doc in docs:
            content = doc.page_content.strip()
            
            normalized = ' '.join(content.lower().split())
            
            if content in seen_content:
                continue
            
            if normalized in seen_normalized:
                continue
            
            is_similar = False
            for existing_doc in unique_docs:
                if self._is_too_similar(content, existing_doc.page_content):
                    is_similar = True
                    break
            
            if is_similar:
                continue
                        
            seen_content.add(content)
            seen_normalized.add(normalized)
            unique_docs.append(doc)
        
        removed = len(docs) - len(unique_docs)
        if removed > 0:
            print(f"[DEDUP] Removed {removed} duplicate/similar chunks")
        
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
        return {"answer": "No relevant info found.", "sources": [], "pipeline": "3-stage"}

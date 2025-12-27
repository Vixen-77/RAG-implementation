

from typing import List, Dict, Any
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
    
    def query(self, user_question: str, k: int = 3, child_k: int = 20) -> Dict[str, Any]:
        """Execute RAG pipeline."""
        print(f"\n[QUERY] {user_question}")
        
        # 1. Search
        is_visual = self._is_visual_query(user_question)
        search_k = child_k + 5 if is_visual else child_k
        child_docs = self._search_children(user_question, k=search_k, include_images=is_visual)
        
        if not child_docs:
            return self._no_results_response()
            
        print(f"[RAG] Found {len(child_docs)} children")
        
        # 2. Rerank
        try:
            reranked = rerank_results(user_question, child_docs, top_k=k)
        except Exception as e:
            print(f"[WARN] Rerank failed: {e}")
            reranked = child_docs[:k]
            
        # 3. Generate
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
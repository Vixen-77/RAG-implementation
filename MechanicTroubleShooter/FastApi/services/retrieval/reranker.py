
import time
from typing import List, Tuple
from langchain_core.documents import Document

# Lazy load the model to avoid slow startup
_reranker = None
_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker():
    
    global _reranker
    
    if _reranker is None:
        print(f"🔄 [Reranker] Loading cross-encoder model: {_model_name}...")
        start = time.time()
        
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(_model_name)
            elapsed = round(time.time() - start, 2)
        except ImportError:
            print(" [Reranker] sentence-transformers not installed!")
            return None
        except Exception as e:
            print(f"❌ [Reranker] Failed to load model: {e}")
            return None
    
    return _reranker


def rerank_results(query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
    
    if not documents:
        return []
    
    reranker = get_reranker()
    
    if reranker is None:
        print("⚠️ [Reranker] Model not available, returning original order")
        return documents[:top_k]
    
    print(f" [Reranker] Re-scoring {len(documents)} candidates...")
    start = time.time()
    
    try:
        pairs = [[query, doc.page_content] for doc in documents]
        
        scores = reranker.predict(pairs)
        
        scored_docs: List[Tuple[Document, float]] = list(zip(documents, scores))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
   
        actual_k = min(top_k, len(scored_docs))
        reranked = [doc for doc, score in scored_docs[:actual_k]]
        
        elapsed = round(time.time() - start, 3)
        
        print(f" [Reranker] Complete in {elapsed}s")
        print(f"   Top scores: {[round(s, 10) for _, s in scored_docs[:3]]}")
        
        if documents and reranked:
            original_first = documents[0].page_content[:50]
            reranked_first = reranked[0].page_content[:50]
            if original_first != reranked_first:
                print(f"    Reordering detected - top result changed")
        
        return reranked
        
    except Exception as e:
        print(f" [Reranker] Scoring failed: {e}")
        return documents[:min(top_k, len(documents))]
    
   

def rerank_with_scores(query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
    
    if not documents:
        return []
    
    reranker = get_reranker()
    
    if reranker is None:
        # Return with zero scores if model not available
        return [(doc, 0.0) for doc in documents[:top_k]]
    
    try:
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
        
    except Exception as e:
        print(f" [Reranker] Failed: {e}")
        return [(doc, 0.0) for doc in documents[:top_k]]


def filter_low_relevance(documents: List[Document], scores: List[float], 
                         threshold: float = 0.1) -> List[Document]:
    
    filtered = [doc for doc, score in zip(documents, scores) if score >= threshold]
    
    if len(filtered) < len(documents):
        removed = len(documents) - len(filtered)
        print(f"   🗑️ [Reranker] Filtered {removed} low-relevance chunks (threshold: {threshold})")
    
    return filtered




    
  
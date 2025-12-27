
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


def rerank_results(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    
    if not documents:
        return []
    
    reranker = get_reranker()
    
    if reranker is None:
        print("⚠️ [Reranker] Model not available, returning original order")
        return documents[:top_k]
    
    print(f" [Reranker] Re-scoring {len(documents)} candidates...")
    start = time.time()
    
    try:
        # Create query-document pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Combine documents with scores
        scored_docs: List[Tuple[Document, float]] = list(zip(documents, scores))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, score in scored_docs[:top_k]]
        
        elapsed = round(time.time() - start, 3)
        
        print(f" [Reranker] Complete in {elapsed}s")
        print(f"   Top scores: {[round(s, 3) for _, s in scored_docs[:3]]}")
        
        if documents and reranked:
            original_first = documents[0].page_content[:50]
            reranked_first = reranked[0].page_content[:50]
            if original_first != reranked_first:
                print(f"    Reordering detected - top result changed")
        
        return reranked
        
    except Exception as e:
        print(f" [Reranker] Scoring failed: {e}")
        return documents[:top_k]


def rerank_with_scores(query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
    
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




    
    
    # Simulate retrieved documents
    test_docs = [
        Document(page_content="Test 9: Check fuel injector flow rate using diagnostic tool.", 
                 metadata={"section_code": "13B", "system_context": "Diesel Injection"}),
        Document(page_content="The AdBlue tank has a capacity of 17 liters and is located under the boot floor.",
                 metadata={"section_code": "19B", "system_context": "Exhaust After-treatment"}),
        Document(page_content="DPF regeneration occurs automatically when soot loading reaches 45%.",
                 metadata={"section_code": "19B", "system_context": "Exhaust After-treatment"}),
        Document(page_content="Replace the diesel fuel filter every 60,000 km.",
                 metadata={"section_code": "13B", "system_context": "Diesel Injection"}),
    ]
    
    query = "AdBlue fault and DPF blocked"
    
    print(f"\nQuery: \"{query}\"")
    print("\nBefore reranking (order from vector search):")
    for i, doc in enumerate(test_docs):
        print(f"  {i+1}. [{doc.metadata['section_code']}] {doc.page_content[:60]}...")
    
    # Rerank
    reranked = rerank_with_scores(query, test_docs, top_k=4)
    
    print("\nAfter reranking:")
    for i, (doc, score) in enumerate(reranked):
        print(f"  {i+1}. [Score: {score:.3f}] [{doc.metadata['section_code']}] {doc.page_content[:50]}...")
    
    # Verify AdBlue/DPF content ranks higher than fuel injector content
    if reranked:
        top_section = reranked[0][0].metadata.get('section_code', '')
        if top_section == '19B':
            print("\n✅ TEST PASSED: Exhaust content (19B) ranked higher than Injection (13B)")
        else:
            print("\n⚠️ TEST WARNING: Expected 19B to rank highest")


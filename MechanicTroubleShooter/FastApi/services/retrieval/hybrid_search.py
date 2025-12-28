

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

_bm25_index = None
_bm25_corpus = None
_bm25_doc_map: Dict[int, Document] = {}

BM25_PERSIST_PATH = "./chroma_db/bm25_index.pkl"


def _tokenize(text: str) -> List[str]:
    import re
    text = re.sub(r'[^\w\s\-\.]', ' ', text.lower())
    return text.split()


def _load_bm25_index():
    global _bm25_index, _bm25_corpus, _bm25_doc_map
    
    if os.path.exists(BM25_PERSIST_PATH):
        try:
            with open(BM25_PERSIST_PATH, 'rb') as f:
                data = pickle.load(f)
                _bm25_index = data.get('index')
                _bm25_corpus = data.get('corpus')
                _bm25_doc_map = data.get('doc_map', {})
                print(f"[BM25] Loaded index with {len(_bm25_doc_map)} documents")
                return True
        except Exception as e:
            print(f"[BM25] Failed to load index: {e}")
    return False


def _save_bm25_index():
    """Persist BM25 index to disk."""
    global _bm25_index, _bm25_corpus, _bm25_doc_map
    
    try:
        os.makedirs(os.path.dirname(BM25_PERSIST_PATH), exist_ok=True)
        with open(BM25_PERSIST_PATH, 'wb') as f:
            pickle.dump({
                'index': _bm25_index,
                'corpus': _bm25_corpus,
                'doc_map': _bm25_doc_map
            }, f)
        print(f"[BM25] Saved index with {len(_bm25_doc_map)} documents")
    except Exception as e:
        print(f"[BM25] Failed to save index: {e}")


def rebuild_bm25_index(documents: List[Document] = None):
    global _bm25_index, _bm25_corpus, _bm25_doc_map
    
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[BM25] rank_bm25 not installed. Run: pip install rank_bm25")
        return False
    
    if documents is None:
        from services.storage.vector import vector_db
        try:
            results = vector_db.get(where={"type": "child"})
            if results and results.get('documents'):
                documents = []
                for i, content in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results.get('metadatas') else {}
                    documents.append(Document(page_content=content, metadata=metadata))
                print(f"[BM25] Fetched {len(documents)} child documents from vector store")
        except Exception as e:
            print(f"[BM25] Failed to fetch documents: {e}")
            return False
    
    if not documents:
        print("[BM25] No documents to index")
        return False
    
    print(f"[BM25] Building index for {len(documents)} documents...")
    
    _bm25_corpus = []
    _bm25_doc_map = {}
    
    for idx, doc in enumerate(documents):
        tokens = _tokenize(doc.page_content)
        _bm25_corpus.append(tokens)
        _bm25_doc_map[idx] = doc
    
    _bm25_index = BM25Okapi(_bm25_corpus)
    
    _save_bm25_index()
    print(f"[BM25] Index built successfully")
    return True


def bm25_search(query: str, k: int = 20) -> List[Tuple[Document, float]]:
    """
    Perform BM25 keyword search.
    Returns list of (document, score) tuples.
    """
    global _bm25_index, _bm25_doc_map
    
    # Lazy load index
    if _bm25_index is None:
        if not _load_bm25_index():
            print("[BM25] No index available. Run rebuild_bm25_index() first.")
            return []
    
    query_tokens = _tokenize(query)
    
    try:
        scores = _bm25_index.get_scores(query_tokens)
        
        # Get top k indices
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_indices[:k]
        
        results = []
        for idx, score in top_k:
            if idx in _bm25_doc_map and score > 0:
                results.append((_bm25_doc_map[idx], score))
        
        return results
        
    except Exception as e:
        print(f"[BM25] Search failed: {e}")
        return []


def reciprocal_rank_fusion(
    vector_results: List[Document],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> List[Document]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank)) for each result list
    
    Args:
        vector_results: Documents from vector search
        bm25_results: (Document, score) tuples from BM25
        k: RRF constant (default 60, standard value)
        vector_weight: Weight for vector search results
        bm25_weight: Weight for BM25 results
    
    Returns:
        Fused and reordered list of documents
    """
    doc_scores: Dict[str, Tuple[Document, float]] = {}
    
    # Score vector results
    for rank, doc in enumerate(vector_results):
        doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
        rrf_score = vector_weight * (1.0 / (k + rank + 1))
        
        if doc_id in doc_scores:
            doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + rrf_score)
        else:
            doc_scores[doc_id] = (doc, rrf_score)
    
    # Score BM25 results
    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
        rrf_score = bm25_weight * (1.0 / (k + rank + 1))
        
        if doc_id in doc_scores:
            doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + rrf_score)
        else:
            doc_scores[doc_id] = (doc, rrf_score)
    
    # Sort by combined RRF score
    sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in sorted_results]


def hybrid_search(query: str, k: int = 20, include_images: bool = False) -> List[Document]:
    """
    Perform hybrid search combining vector and BM25.
    
    Stage 1 of 3-Stage Retrieval Pipeline.
    """
    from services.storage.vector import vector_db
    
    print(f"[Hybrid] Starting hybrid search for: {query[:50]}...")
    
    try:
        vector_results = vector_db.similarity_search(query, k=k*2, filter={"type": "child"})
        print(f"[Hybrid] Vector search (children): {len(vector_results)} results")
    except Exception as e:
        print(f"[Hybrid] Vector search failed: {e}")
        vector_results = []
    
    bm25_results = bm25_search(query, k=k*2)
    print(f"[Hybrid] BM25 search: {len(bm25_results)} results")
    
    if not bm25_results:
        print("[Hybrid] BM25 empty, using vector results only")
        combined = vector_results[:k]
    elif not vector_results:
        print("[Hybrid] Vector empty, using BM25 results only")
        combined = [doc for doc, _ in bm25_results[:k]]
    else:
        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        print(f"[Hybrid] RRF fusion: {len(fused)} unique documents")
        combined = fused[:k]
    
    if include_images and combined:
        relevant_pages = set()
        relevant_files = set()
        for doc in combined[:10]:  
            page_numbers_str = doc.metadata.get('page_numbers', '')
            source = doc.metadata.get('source_file')
            if page_numbers_str and source:
                relevant_files.add(source)
                for page_str in page_numbers_str.split(','):
                    try:
                        relevant_pages.add(int(page_str.strip()))
                    except (ValueError, AttributeError):
                        pass
        
        print(f"[Hybrid] Relevant files: {relevant_files}")
        print(f"[Hybrid] Relevant pages from text: {sorted(relevant_pages)[:10]}{'...' if len(relevant_pages) > 10 else ''}")
        
        page_matched_images = []
        if relevant_pages:
            try:
                all_images = vector_db.get(
                    where={"type": "image"},
                    limit=500
                )
                
                if all_images and all_images.get('metadatas'):
                    for i, meta in enumerate(all_images['metadatas']):
                        img_page = meta.get('page')  
                        img_source = meta.get('source_file')
                        if img_page in relevant_pages and img_source in relevant_files:
                            from langchain_core.documents import Document as LangDoc
                            doc = LangDoc(
                                page_content=all_images['documents'][i],
                                metadata=meta
                            )
                            page_matched_images.append(doc)
                
                print(f"[Hybrid] Found {len(page_matched_images)} images from relevant pages")
                    
            except Exception as e:
                print(f"[Hybrid] Image page matching failed: {e}")
        
        for img in page_matched_images[:5]:  
            combined.insert(0, img)
    
    image_count = sum(1 for d in combined if d.metadata.get('type') == 'image')
    print(f"[Hybrid] Final: {len(combined)} docs ({image_count} images)")
    
    return combined



def get_bm25_stats() -> Dict[str, Any]:
    global _bm25_index, _bm25_doc_map
    
    if _bm25_index is None:
        _load_bm25_index()
    
    return {
        "indexed": _bm25_index is not None,
        "num_documents": len(_bm25_doc_map) if _bm25_doc_map else 0,
        "persist_path": BM25_PERSIST_PATH
    }

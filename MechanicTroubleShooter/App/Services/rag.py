"""
Parent-Child RAG Retrieval System
==================================

This module implements the retrieval side of Parent-Child chunking:

1. SEARCH: User query → Vector search on children (small, searchable chunks)
2. FETCH: Get parent_ids from matched children
3. RETRIEVE: Fetch full parent documents from document store
4. RERANK: Use cross-encoder to rerank parents by relevance
5. GENERATE: Send full parent context to LLM for answer generation

Why this works:
- Children are optimized for semantic search (small, focused)
- Parents provide complete context (warnings, steps, diagrams)
- LLM gets full context without truncation artifacts
"""

from typing import List, Dict, Any
from langchain_core.documents import Document

from Services.vectorStore import vector_db
from Services.reranker import rerank_results
from Services.llm_service import generate_chat_answer
from Services.docstore import get_docstore


class ParentChildRAG:
    """
    Parent-Child Retrieval-Augmented Generation system.
    
    This class handles querying the Parent-Child architecture:
    1. Search children in vector DB
    2. Fetch corresponding parents from document store
    3. Rerank parents for relevance
    4. Generate answer with full context
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize RAG system.
        
        Args:
            persist_dir: Path to ChromaDB persistence directory
        
        Note: Uses shared docstore singleton that is populated by ingestion pipeline.
        """
        self.persist_dir = persist_dir
        self.vectorstore = vector_db
        
        # Use shared document store (populated by ingestion pipeline)
        self.docstore = get_docstore()
        
        # Check if documents are indexed
        try:
            count = self.vectorstore._collection.count()
            if count == 0:
                raise FileNotFoundError("No documents indexed. Please ingest PDFs first.")
            print(f"✅ [RAG] Initialized with {count} child chunks")
            print(f"   📚 Parent store has {len(self.docstore)} parents")
        except Exception as e:
            print(f"⚠️  [RAG] Warning: {e}")
    
    def query(
        self, 
        user_question: str, 
        k: int = 3,
        child_k: int = 20
    ) -> Dict[str, Any]:
        """
        Main query method using Parent-Child retrieval.
        
        Pipeline:
        1. Search for top child_k children (cast wide net)
        2. Extract unique parent_ids
        3. Fetch full parent documents
        4. Rerank parents by relevance
        5. Generate answer using top k parents
        
        Args:
            user_question: User's query
            k: Number of parent chunks to use for generation
            child_k: Number of children to retrieve initially
            
        Returns:
            Dict with answer, sources, metadata
        """
        print(f"\n{'='*70}")
        print(f"🔍 Query: {user_question}")
        print(f"{'='*70}")
        
        # Step 1: Detect Intent & Search
        is_visual = self._is_visual_query(user_question)
        print(f"📍 Step 1: Search (Visual Intent: {is_visual})")
        
        # Increase k for visual queries to ensure we get a mix of text/images
        search_k = child_k + 5 if is_visual else child_k
        
        child_docs = self._search_children(
            user_question, 
            k=search_k, 
            include_images=is_visual
        )
        
        if not child_docs:
            return self._no_results_response()
        
        print(f"   ✓ Found {len(child_docs)} children")
        
        # NOTE: Skipping parent fetching as per user request (use chunks directly)
        source_docs = child_docs
        
        # Step 4: Rerank chunks by relevance
        print(f"🎯 Step 2: Reranking chunks for relevance...")
        try:
            reranked_docs = rerank_results(user_question, source_docs, top_k=k)
            print(f"   ✓ Reranked to top {len(reranked_docs)}")
        except Exception as e:
            print(f"   ⚠️  Reranking skipped: {e}")
            reranked_docs = source_docs[:k]
        
        # Step 5: Build context and generate answer
        print(f"💬 Step 3: Generating answer...")
        context = self._build_context(reranked_docs)
        
        answer = generate_chat_answer(
            context=context,
            user_question=user_question,
            routing_info={"strategy": "child_chunk_direct"}
        )
        
        # Format response
        print(f"✅ Answer generated ({len(answer)} chars)")
        print(f"{'='*70}\n")
        
        return {
            "answer": answer,
            "sources": reranked_docs,
            "num_sources": len(reranked_docs),
            "context_chars": len(context),
            "formatted_sources": self._format_sources(reranked_docs)
        }
    
    def _is_visual_query(self, query: str) -> bool:
        """Check if query implies visual intent (images/diagrams)"""
        visual_keywords = [
            "show", "diagram", "picture", "image", "photo", 
            "where is", "location", "locate", "look like", "see"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visual_keywords)

    def _search_children(self, query: str, k: int = 20, include_images: bool = False) -> List[Document]:
        """
        Search for child chunks in vector database.
        
        Args:
            query: User's search query
            k: Number of children to retrieve
            include_images: Whether to include image captions in search results
            
        Returns:
            List of child Document objects
        """
        try:
            # If visual query, search BOTH text and images (no type filter or explicit list)
            # ChromaDB filter syntax for "OR" is complex/limited in some versions.
            # Simplified approach: 
            # - If include_images=False -> filter={"type": "child"}
            # - If include_images=True -> filter=None (search everything) or filter={"$or": [{"type": "child"}, {"type": "image"}]}
            
            filter_dict = {"type": "child"}
            if include_images:
                # Search everything (parents are in docstore, so vector store only has children + images)
                filter_dict = None 
                
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def _extract_parent_ids(self, child_docs: List[Document]) -> List[str]:
        """
        Extract unique parent IDs from child documents.
        
        Args:
            child_docs: List of child Document objects
            
        Returns:
            List of unique parent_id strings
        """
        parent_ids = set()
        
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                parent_ids.add(parent_id)
        
        return list(parent_ids)
    
    def _fetch_parents(self, parent_ids: List[str]) -> List[Document]:
        """
        Fetch full parent documents from document store.
        
        Args:
            parent_ids: List of parent IDs to fetch
            
        Returns:
            List of parent Document objects
        """
        parents = []
        
        for parent_id in parent_ids:
            parent = self.docstore.get_document(parent_id)
            if parent:
                parents.append(parent)
            else:
                print(f"   ⚠️  Parent {parent_id} not found in store")
        
        return parents
    
    def _build_context(self, parent_docs: List[Document]) -> str:
        """
        Build context string from parent documents with source tracking.
        
        Args:
            parent_docs: List of parent Document objects
            
        Returns:
            Formatted context string with citations
        """
        context_parts = []
        
        for i, doc in enumerate(parent_docs):
            metadata = doc.metadata
            
            # Build source header
            source_info = (
                f"[Source {i+1}, "
                f"Section {metadata.get('section_code', 'N/A')}, "
                f"Title: {metadata.get('section_title', 'N/A')}]"
            )
            
            # Add content
            context_parts.append(f"{source_info}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _format_sources(self, parent_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Format parent documents into readable source metadata.
        
        Args:
            parent_docs: List of parent Document objects
            
        Returns:
            List of formatted source dicts
        """
        formatted = []
        
        for i, doc in enumerate(parent_docs):
            meta = doc.metadata
            
            formatted.append({
                "source_num": i + 1,
                "section_title": meta.get("section_title", "Unknown"),
                "section_code": meta.get("section_code", "N/A"),
                "system": meta.get("system_context", "General"),
                "vehicle": meta.get("vehicle_model", "Unknown"),
                "file": meta.get("source_file", "Unknown"),
                "image_path": meta.get("image_path", None), # For frontend display
                "type": meta.get("type", "text"),
                "char_count": len(doc.page_content),
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return formatted
    
    def _no_results_response(self) -> Dict[str, Any]:
        """Return response when no results found"""
        return {
            "answer": (
                "I couldn't find relevant information in the vehicle manual. "
                "Please try rephrasing your question or provide more details about "
                "the specific system or component you're asking about."
            ),
            "sources": [],
            "num_sources": 0,
            "context_chars": 0,
            "formatted_sources": []
        }
    
    def search_with_details(
        self,
        query: str,
        k: int = 3,
        child_k: int = 20
    ) -> Dict[str, Any]:
        """
        Enhanced query method that returns detailed retrieval information.
        
        Useful for debugging and understanding the retrieval process.
        
        Returns:
            Dict with answer, sources, and detailed retrieval metrics
        """
        # Run standard query
        result = self.query(query, k=k, child_k=child_k)
        
        # Add detailed metrics
        children = self._search_children(query, k=child_k)
        parent_ids = self._extract_parent_ids(children)
        
        result["retrieval_details"] = {
            "children_retrieved": len(children),
            "unique_parents_found": len(parent_ids),
            "parents_used": len(result["sources"]),
            "average_parent_length": sum(
                len(doc.page_content) for doc in result["sources"]
            ) // len(result["sources"]) if result["sources"] else 0
        }
        
        return result


def test_parent_child_rag():
    """
    Test function to verify Parent-Child RAG is working.
    """
    print("\n" + "="*70)
    print("🧪 PARENT-CHILD RAG SYSTEM TEST")
    print("="*70)
    
    try:
        # Initialize system
        rag = ParentChildRAG()
        
        # Test query
        test_query = "What should I do if my DPF warning light is on?"
        print(f"\n📝 Test Query: {test_query}")
        
        # Run query with details
        result = rag.search_with_details(test_query, k=2, child_k=10)
        
        print("\n📊 Results:")
        print(f"   Answer length: {len(result['answer'])} chars")
        print(f"   Sources used: {result['num_sources']}")
        print(f"   Context size: {result['context_chars']} chars")
        
        if result.get("retrieval_details"):
            details = result["retrieval_details"]
            print("\n🔍 Retrieval Details:")
            print(f"   Children retrieved: {details['children_retrieved']}")
            # print(f"   Unique parents: {details.get('unique_parents_found', 'N/A')}")
            # print(f"   Parents used: {details['parents_used']}")
            print(f"   Avg source length: {details['average_parent_length']} chars")
        
        print("\n✅ TEST PASSED: Parent-Child RAG is operational")
        
        return result
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_parent_child_rag()
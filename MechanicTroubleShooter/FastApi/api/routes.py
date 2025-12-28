
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List, Dict, Any
import os
import shutil

from core.config import DATA_FOLDER, CHROMA_PERSIST_DIR
from schemas.models import ChatRequest, ChatResponse, IngestResponse, StatsResponse
from services import get_collection_stats, clear_database, get_docstore, ParentChildRAG

router = APIRouter()

def get_rag_system(request: Request):
    if not hasattr(request.app.state, "rag_system"): return None
    return request.app.state.rag_system

def get_ingestion_pipeline(request: Request):
    if not hasattr(request.app.state, "ingestion_pipeline"): return None
    return request.app.state.ingestion_pipeline

@router.get("/")
def read_root(request: Request):
    """Health check endpoint."""
    pipe = get_ingestion_pipeline(request)
    rag = get_rag_system(request)
    ready = rag is not None
    
    stats = pipe.get_stats() if pipe else {"children_count": 0}
    doc_stats = get_docstore().get_stats()
    
    return {
        "status": "System operational" if ready else "System initializing",
        "service": "Mecanic-IA Parent-Child RAG",
        "rag_ready": ready,
        "database_stats": {**stats, "docstore": doc_stats.get("total_parents", 0)}
    }


@router.post("/ingest", response_model=IngestResponse)
async def upload_manual(request: Request, file: UploadFile = File(...), force_reingest: bool = False):
    pipe = get_ingestion_pipeline(request)
    if not pipe: raise HTTPException(503, "Ingestion unavailable")
    
    if not file.filename.endswith('.pdf'): raise HTTPException(400, "PDF only")
    
    path = os.path.join(DATA_FOLDER, file.filename)
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    
    print(f"[API] Saved: {path}")
    res = pipe.ingest_pdf(path, force=force_reingest)
    
    if res.get("status") == "success":
        request.app.state.rag_system = ParentChildRAG(persist_dir=CHROMA_PERSIST_DIR)
    
    return IngestResponse(
        message="Ingestion complete" if res["status"]=="success" else "Skipped",
        filename=file.filename,
        status=res.get("status"),
        parents_processed=res.get("parents", 0),
        children_processed=res.get("children", 0)
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_req: ChatRequest):
    """Chat endpoint."""
    rag = get_rag_system(request)
    if not rag: raise HTTPException(503, "System not ready")
    
    print(f"[API] Chat: {chat_req.query[:50]}...")
    try:
        res = rag.query(chat_req.query, k=10)
        return ChatResponse(
            answer=res.get("answer"),
            sources=[{"content": d.page_content[:200], "meta": d.metadata} for d in res.get("sources", [])],
            num_sources=res.get("num_sources", 0)
        )
    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        raise HTTPException(500, str(e))


@router.post("/chat/stream")
async def chat_stream(request: Request, chat_req: ChatRequest):
    """Streaming chat endpoint with agentic routing."""
    from sse_starlette.sse import EventSourceResponse
    from services import get_conversation_store, stream_chat_answer
    from services.llm.router import (
        route_query, QueryRoute, 
        generate_direct_answer, generate_clarification_request, generate_out_of_scope_response
    )
    import json
    
    rag = get_rag_system(request)
    if not rag: 
        raise HTTPException(503, "System not ready")
    
    conv_store = get_conversation_store()
    conv_id = conv_store.get_or_create(chat_req.conversation_id)
    history = conv_store.get_history(conv_id) or []
    
    print(f"[API] Stream chat: {chat_req.query[:50]}... (conv: {conv_id[:8]})")
    

    conv_store.add_message(conv_id, "user", chat_req.query)
    

    routing = route_query(chat_req.query, history)
    decision = routing["decision"]
    
    # Force RAG for visual queries (images, diagrams, etc.)
    visual_keywords = ["show", "diagram", "picture", "image", "photo", "location", "look like", "see", "where is"]
    is_visual = any(k in chat_req.query.lower() for k in visual_keywords)
    if is_visual and decision == QueryRoute.DIRECT_ANSWER:
        print(f"[ROUTER] Visual query detected, overriding to RAG_NEEDED")
        decision = QueryRoute.RAG_NEEDED

    if decision == QueryRoute.DIRECT_ANSWER:
        async def direct_response():
            yield {"event": "metadata", "data": json.dumps({
                "conversation_id": conv_id, "num_sources": 0, "route": "direct"
            })}
            answer = generate_direct_answer(chat_req.query, history)
            yield {"event": "token", "data": answer}
            conv_store.add_message(conv_id, "assistant", answer)
            yield {"event": "done", "data": ""}
        return EventSourceResponse(direct_response())
    
    if decision == QueryRoute.CLARIFICATION_NEEDED:
        async def clarification_response():
            yield {"event": "metadata", "data": json.dumps({
                "conversation_id": conv_id, "num_sources": 0, "route": "clarification"
            })}
            answer = generate_clarification_request(chat_req.query)
            yield {"event": "token", "data": answer}
            conv_store.add_message(conv_id, "assistant", answer)
            yield {"event": "done", "data": ""}
        return EventSourceResponse(clarification_response())
    
    if decision == QueryRoute.OUT_OF_SCOPE:
        async def out_of_scope_response():
            yield {"event": "metadata", "data": json.dumps({
                "conversation_id": conv_id, "num_sources": 0, "route": "out_of_scope"
            })}
            answer = generate_out_of_scope_response(chat_req.query)
            yield {"event": "token", "data": answer}
            conv_store.add_message(conv_id, "assistant", answer)
            yield {"event": "done", "data": ""}
        return EventSourceResponse(out_of_scope_response())
    
    # === RAG_NEEDED: Proceed with retrieval ===
    query_to_use = routing.get("reformulated_query", chat_req.query)
    
    try:
        from services.retrieval.hybrid_search import hybrid_search
        from services.retrieval.reranker import rerank_results
        
        # Check visual intent on ORIGINAL query (before reformulation strips keywords)
        is_visual = rag._is_visual_query(chat_req.query)
        print(f"[API] is_visual={is_visual} for query: {chat_req.query[:50]}...")
        
        # Stage 1: Hybrid Search (Vector + BM25)
        child_docs = hybrid_search(query_to_use, k=chat_req.k * 3, include_images=is_visual)
        
        # Track images after hybrid search
        img_count_hybrid = sum(1 for d in child_docs if d.metadata.get('type') == 'image')
        print(f"[API] After hybrid: {len(child_docs)} docs ({img_count_hybrid} images)")
        
        if not child_docs:
            async def no_results():
                yield {"event": "metadata", "data": json.dumps({"conversation_id": conv_id, "num_sources": 0, "route": "rag"})}
                yield {"event": "token", "data": "No relevant information found in the manual."}
                yield {"event": "done", "data": ""}
            return EventSourceResponse(no_results())
        
        # Deduplicate
        unique_docs = rag._deduplicate_aggressively(child_docs)
        
        # Track images after dedup
        img_count_dedup = sum(1 for d in unique_docs if d.metadata.get('type') == 'image')
        print(f"[API] After dedup: {len(unique_docs)} docs ({img_count_dedup} images)")
        
        # Stage 3: Reranking
        try:
            from services.retrieval.reranker import rerank_with_scores
            from services.llm.client import evaluate_image_relevance
            
            scored_docs = rerank_with_scores(query_to_use, unique_docs[:50], top_k=chat_req.k + 10)
            
            # Use LLM to evaluate image relevance instead of score threshold
            filtered_docs = []
            for doc, score in scored_docs:
                if doc.metadata.get('type') == 'image':
                    # Ask LLM if this image is relevant to the query
                    is_relevant = evaluate_image_relevance(chat_req.query, doc.page_content)
                    if is_relevant:
                        filtered_docs.append(doc)
                        print(f"[API] Image KEPT by LLM: {doc.page_content[:50]}...")
                    else:
                        print(f"[API] Image FILTERED by LLM: {doc.page_content[:50]}...")
                else:
                    filtered_docs.append(doc)
            
            reranked = filtered_docs[:chat_req.k]
        except Exception as e:
            print(f"[API] Reranking with LLM filter failed: {e}, using basic rerank")
            reranked = rerank_results(query_to_use, unique_docs[:50], top_k=chat_req.k)
        
        # Track images after rerank
        img_count_rerank = sum(1 for d in reranked if d.metadata.get('type') == 'image')
        print(f"[API] After rerank: {len(reranked)} docs ({img_count_rerank} images)")
        
        # Stage 2: Build context with parent retrieval
        context = rag._build_context_with_parents(reranked)
        sources = [{"content": d.page_content[:200], "meta": d.metadata} for d in reranked]
        
        # Log image sources for debugging
        image_sources = [s for s in sources if s["meta"].get("type") == "image"]
        print(f"[API] Sending {len(sources)} sources ({len(image_sources)} images) to frontend")
        
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))
    
    async def event_generator():
        yield {
            "event": "metadata", 
            "data": json.dumps({
                "conversation_id": conv_id,
                "num_sources": len(sources),
                "sources": sources,
                "route": "rag"
            })
        }
        
        full_response = []
        for token in stream_chat_answer(context, chat_req.query, history):
            full_response.append(token)
            yield {"event": "token", "data": token}
        
        final_answer = "".join(full_response)
        conv_store.add_message(conv_id, "assistant", final_answer)
        yield {"event": "done", "data": ""}
    
    return EventSourceResponse(event_generator())


@router.get("/stats", response_model=StatsResponse)
def get_stats(request: Request):
    """System stats."""
    pipe = get_ingestion_pipeline(request)
    rag = get_rag_system(request)
    if not pipe: return StatsResponse(children_count=0, parents_count=0, docstore_parents=0, persist_dir="", system_ready=False, avg_parent_chars=0)
    
    s = pipe.get_stats()
    ds = get_docstore().get_stats()
    return StatsResponse(
        children_count=s['children_count'],
        parents_count=s['parents_count'],
        docstore_parents=ds['total_parents'],
        persist_dir=s['persist_dir'],
        system_ready=rag is not None,
        avg_parent_chars=ds['avg_chars']
    )

@router.delete("/reset")
def reset_database(request: Request, confirm: bool = False):
    """Reset database."""
    if not confirm: raise HTTPException(400, "Confirm required")
    
    clear_database()
    get_docstore().clear()
    request.app.state.rag_system = None
    print("[WARN] Database reset")
    return {"message": "Database cleared"}

@router.get("/health")
def health_check(request: Request):
    """Health check."""
    pipe = get_ingestion_pipeline(request)
    rag = get_rag_system(request)
    return {
        "status": "healthy" if pipe and rag else "degraded",
        "components": {"pipeline": pipe is not None, "rag": rag is not None}
    }


from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List, Dict, Any
import os
import shutil

from core.config import DATA_FOLDER, CHROMA_PERSIST_DIR, DEMO_QUERIES
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

@router.get("/demos")
def list_demos():
    """List demo scenarios."""
    return {
        "demos": [{"id": k, "desc": v["description"], "link": f"/demo/{k}"} for k,v in DEMO_QUERIES.items()]
    }

@router.get("/demo/{demo_id}")
async def run_demo(demo_id: str, request: Request):
    """Run demo scenario."""
    rag = get_rag_system(request)
    if demo_id not in DEMO_QUERIES or not rag:
        raise HTTPException(503, "Demo unavailable or system not ready")
    
    demo = DEMO_QUERIES[demo_id]
    print(f"[DEMO] Running: {demo['description']}")
    
    try:
        res = rag.query(demo["query"], k=3)
        return {
            "answer": res.get("answer"),
            "sources": [{"content": d.page_content[:200], "meta": d.metadata} for d in res.get("sources", [])],
            "demo_id": demo_id
        }
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        raise HTTPException(500, str(e))

@router.post("/ingest", response_model=IngestResponse)
async def upload_manual(request: Request, file: UploadFile = File(...), force_reingest: bool = False):
    """Upload PDF."""
    pipe = get_ingestion_pipeline(request)
    if not pipe: raise HTTPException(503, "Ingestion unavailable")
    
    if not file.filename.endswith('.pdf'): raise HTTPException(400, "PDF only")
    
    path = os.path.join(DATA_FOLDER, file.filename)
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    
    print(f"[API] Saved: {path}")
    res = pipe.ingest_pdf(path, force=force_reingest)
    
    if res.get("status") == "success":
        # Re-init RAG
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
        res = rag.query(chat_req.query, k=chat_req.k)
        return ChatResponse(
            answer=res.get("answer"),
            sources=[{"content": d.page_content[:200], "meta": d.metadata} for d in res.get("sources", [])],
            num_sources=res.get("num_sources", 0)
        )
    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        raise HTTPException(500, str(e))

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

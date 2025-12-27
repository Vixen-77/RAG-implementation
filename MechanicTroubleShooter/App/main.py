"""
FastAPI Server for Parent-Child RAG System
===========================================

Endpoints:
- GET  /              Health check
- GET  /demos         List demo scenarios
- GET  /demo/{id}     Run pre-configured demo
- POST /ingest        Upload and process PDF
- POST /chat          Query the RAG system
- GET  /stats         Get database statistics
- DELETE /reset       Clear database (requires confirm=true)

Updated for Parent-Child chunking architecture.
"""

import os
import shutil
import glob
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import Parent-Child RAG components
from Services.ingest import MultimodalIngestionPipeline
from Services.rag import ParentChildRAG
from Services.vectorStore import get_collection_stats, clear_database
from Services.docstore import get_docstore

# Configuration
DATA_FOLDER = "Data"
CHROMA_PERSIST_DIR = "./chroma_parent_child"

# Demo queries for testing
DEMO_QUERIES = {
    "engine_smoke": {
        "query": "I see smoke coming from under the hood, what should I do?",
        "description": "Fault: Smoke from engine compartment"
    },
    "won't_start": {
        "query": "I turn the key and the engine doesn't start, but the lights work. What could be wrong?",
        "description": "Fault: Engine fails to start (starter motor works)"
    },
    "rapid_blinking": {
        "query": "My turn signal is blinking much faster than usual. What does that mean?",
        "description": "Fault: Indicator light flashing frequency"
    },
    "steering_vibration": {
        "query": "The steering wheel shakes when I drive fast. Is something broken?",
        "description": "Fault: Vibration while driving"
    },
    "brake_pedal_soft": {
        "query": "My brake pedal feels spongy and goes all the way to the floor. Is it safe to drive?",
        "description": "Fault: Loss of braking pressure"
    },
    "oil_pressure_light": {
        "query": "The oil pressure warning light just came on while I was driving. Should I stop?",
        "description": "Fault: Oil pressure warning"
    },
    "white_smoke_exhaust": {
        "query": "There is white smoke coming from the exhaust but the car runs fine. Is this normal?",
        "description": "Fault: Exhaust smoke (DPF regeneration)"
    },
    "coolant_boiling": {
        "query": "The coolant in the reservoir is boiling. What is the cause?",
        "description": "Fault: Cooling system malfunction"
    },
    "dpf_blocked": {
        "query": "My DPF warning light is on and the engine feels sluggish. What should I do?",
        "description": "Fault: Diesel Particulate Filter blockage"
    },
    "adblue_fault": {
        "query": "I have an AdBlue system fault warning. How do I check and refill AdBlue?",
        "description": "Fault: AdBlue system malfunction"
    },
    "fuse_location": {
        "query": "Where is the fuse for the cooling fan located and what is its amperage?",
        "description": "Info: Fuse box layout query"
    }
}

# Global instances (initialized in lifespan)
ingestion_pipeline: Optional[MultimodalIngestionPipeline] = None
rag_system: Optional[ParentChildRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management with auto-ingestion.
    """
    global ingestion_pipeline, rag_system
    
    print("\n" + "="*70)
    print("🚀 MECANIC-IA PARENT-CHILD RAG SYSTEM STARTING")
    print("="*70)

    # Create data folder if missing
    if not os.path.exists(DATA_FOLDER):
        print(f"⚠️  [Startup] Folder '{DATA_FOLDER}' missing. Creating it.")
        os.makedirs(DATA_FOLDER)

    # Initialize ingestion pipeline
    print("\n🔧 [Startup] Initializing ingestion pipeline...")
    try:
        ingestion_pipeline = MultimodalIngestionPipeline(persist_dir=CHROMA_PERSIST_DIR)
        print("✅ [Startup] Ingestion pipeline ready")
    except Exception as e:
        print(f"❌ [Startup] Failed to initialize ingestion pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Show current database state
    print("\n📊 [Startup] Checking existing database...")
    stats = ingestion_pipeline.get_stats()
    print(f"   Current children: {stats['children_count']}")
    print(f"   Current parents: {stats['parents_count']}")

    # Auto-ingest PDFs from Data folder
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    
    new_files_count = 0
    total_children_added = 0
    processed_count = 0
    skipped_count = 0

    if not pdf_files:
        print(f"\nℹ️  [Startup] No PDFs found in '{DATA_FOLDER}' folder.")
        print(f"💡 [Tip] Place your vehicle manuals in '{DATA_FOLDER}' or use /ingest endpoint")
    else:
        print(f"\n📚 [Startup] Found {len(pdf_files)} manual(s) in '{DATA_FOLDER}':")
        for pdf in pdf_files:
            print(f"   • {os.path.basename(pdf)}")
        
        print(f"\n⏳ [Startup] Beginning auto-ingestion check...\n")

        for pdf_path in pdf_files:
            try:
                print(f"{'─'*70}")
                print(f"📄 Checking: {os.path.basename(pdf_path)}")
                print(f"{'─'*70}")
                
                # Ingest using Parent-Child pipeline
                result = ingestion_pipeline.ingest_pdf(pdf_path, force=False)
                
                status = result.get("status", "unknown")
                
                if status == "skipped":
                    skipped_count += 1
                    print(f"   ⏭️  Skipped (Already indexed)")
                elif status == "success":
                    children_added = result.get("children", 0)
                    parents_added = result.get("parents", 0)
                    images_captioned = result.get("images_captioned", 0)
                    
                    new_files_count += 1
                    total_children_added += children_added
                    processed_count += 1
                    
                    print(f"   ✅ Indexed {parents_added} parents, {children_added} children")
                    print(f"   🖼️  Captioned {images_captioned} images")
                else:
                    skipped_count += 1
                    print(f"   ⚠️  Processing completed with status: {status}")
                    
            except Exception as e:
                print(f"❌ [Startup] Failed to process {os.path.basename(pdf_path)}: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*70}")
        print(f"🎉 AUTO-INGESTION COMPLETE")
        print(f"📊 Summary:")
        print(f"   • New documents processed: {processed_count}")
        print(f"   • Already indexed (skipped): {skipped_count}")
        print(f"   • New child chunks added: {total_children_added}")
        print(f"{'='*70}")

    # Initialize RAG system
    print("\n🔧 [Startup] Initializing RAG query system...")
    try:
        rag_system = ParentChildRAG(persist_dir=CHROMA_PERSIST_DIR)
        print("✅ [Startup] RAG system ready")
    except FileNotFoundError:
        print("⚠️  [Startup] No indexed documents found. RAG queries will fail until documents are ingested.")
        rag_system = None
    except Exception as e:
        print(f"❌ [Startup] Failed to initialize RAG system: {e}")
        import traceback
        traceback.print_exc()
        rag_system = None

    # Show final collection stats
    print("\n📊 [Startup] Final Database State:")
    final_stats = ingestion_pipeline.get_stats()
    docstore_stats = get_docstore().get_stats()
    print(f"   Total parents: {final_stats['parents_count']} (docstore: {docstore_stats['total_parents']})")
    print(f"   Total children: {final_stats['children_count']}")
    print(f"   Avg parent size: {docstore_stats['avg_chars']} chars")

    print(f"\n✅ [Startup] API is ready!")
    print(f"📖 Documentation: http://localhost:8000/docs")
    print(f"🎬 Demo scenarios: http://localhost:8000/demos")
    print(f"{'='*70}\n")
    
    yield
    
    # Shutdown logic
    print("\n" + "="*70)
    print("🛑 MECANIC-IA SHUTTING DOWN")
    print("="*70 + "\n")


# ========== FastAPI App Setup ==========
app = FastAPI(
    title="Mecanic-IA Parent-Child RAG API",
    description="Multimodal RAG system for automotive technical assistance using Parent-Child retrieval architecture",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ========== Data Models ==========
class ChatRequest(BaseModel):
    query: str
    k: Optional[int] = 3  # Number of parent chunks to retrieve

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    num_sources: int = 0
    context_chars: Optional[int] = 0
    formatted_sources: Optional[List[Dict[str, Any]]] = []

class IngestResponse(BaseModel):
    message: str
    filename: str
    status: str
    parents_processed: int = 0
    children_processed: int = 0
    images_captioned: int = 0
    already_indexed: bool = False

class StatsResponse(BaseModel):
    children_count: int
    parents_count: int
    docstore_parents: int
    persist_dir: str
    system_ready: bool
    avg_parent_chars: int


# ========== Endpoints ==========

@app.get("/")
def read_root():
    """Health check endpoint"""
    if ingestion_pipeline:
        stats = ingestion_pipeline.get_stats()
        docstore_stats = get_docstore().get_stats()
        system_ready = rag_system is not None
    else:
        stats = {"children_count": 0, "parents_count": 0}
        docstore_stats = {"total_parents": 0}
        system_ready = False
    
    return {
        "status": "✅ System operational" if system_ready else "⚠️  System initializing",
        "service": "Mecanic-IA Parent-Child RAG",
        "version": "2.0.0",
        "architecture": "Parent-Child Retrieval",
        "documentation": "/docs",
        "demos": "/demos",
        "database_stats": {
            **stats,
            "docstore_parents": docstore_stats.get("total_parents", 0)
        },
        "rag_ready": system_ready
    }


@app.get("/demos")
def list_demos():
    """List available demo scenarios for testing"""
    return {
        "available_demos": [
            {
                "id": key, 
                "description": val["description"], 
                "query": val["query"],
                "endpoint": f"/demo/{key}"
            }
            for key, val in DEMO_QUERIES.items()
        ],
        "usage": "Visit /demo/{demo_id} to run a pre-configured scenario",
        "total_demos": len(DEMO_QUERIES)
    }


@app.get("/demo/{demo_id}")
async def run_demo(demo_id: str):
    """
    Run pre-scripted demo endpoint for testing.
    
    Usage: GET /demo/dpf_blocked
    """
    if demo_id not in DEMO_QUERIES:
        raise HTTPException(
            status_code=404, 
            detail=f"Demo '{demo_id}' not found. Available: {list(DEMO_QUERIES.keys())}"
        )
    
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please ingest documents first."
        )
    
    demo = DEMO_QUERIES[demo_id]
    print(f"\n🎬 [DEMO MODE] Running scenario: {demo['description']}")
    
    try:
        # Query using Parent-Child RAG
        result = rag_system.query(demo["query"], k=3)
        
        # Convert Document objects to serializable dicts
        serializable_sources = []
        for doc in result.get('sources', []):
            serializable_sources.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "full_length": len(doc.page_content)
            })
        
        response = {
            "answer": result.get("answer", ""),
            "sources": serializable_sources,
            "num_sources": result.get("num_sources", 0),
            "context_chars": result.get("context_chars", 0),
            "formatted_sources": result.get("formatted_sources", []),
            "demo_mode": True,
            "demo_id": demo_id,
            "demo_description": demo["description"]
        }
        
        return response
        
    except Exception as e:
        print(f"❌ [Demo] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def upload_manual(file: UploadFile = File(...), force_reingest: bool = False):
    """
    Upload and ingest a new vehicle manual PDF.
    
    Args:
        file: PDF file to upload
        force_reingest: Set to True to re-ingest even if already processed
    """
    if not ingestion_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Ingestion pipeline not initialized"
        )
    
    print(f"\n📥 [API] Received upload: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    try:
        # Save uploaded file
        file_location = os.path.join(DATA_FOLDER, file.filename)
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✅ [API] File saved to {file_location}")
        
        # Ingest the manual using Parent-Child pipeline
        result = ingestion_pipeline.ingest_pdf(file_location, force=force_reingest)
        
        status = result.get("status", "unknown")
        
        if status == "skipped":
            return IngestResponse(
                message="⏭️  Manual already indexed (use force_reingest=true to re-process)",
                filename=file.filename,
                status="skipped",
                parents_processed=0,
                children_processed=0,
                images_captioned=0,
                already_indexed=True
            )
        
        # Reinitialize RAG system to pick up new documents
        global rag_system
        try:
            rag_system = ParentChildRAG(persist_dir=CHROMA_PERSIST_DIR)
            print("🔄 [API] RAG system reinitialized with new documents")
        except Exception as e:
            print(f"⚠️  [API] Failed to reinitialize RAG system: {e}")
        
        return IngestResponse(
            message="✅ Ingestion successful",
            filename=file.filename,
            status="success",
            parents_processed=result.get("parents", 0),
            children_processed=result.get("children", 0),
            images_captioned=result.get("images_captioned", 0),
            already_indexed=False
        )
        
    except Exception as e:
        print(f"❌ [API] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint - query the vehicle manual using Parent-Child RAG.
    """
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please ingest documents first using /ingest endpoint."
        )
    
    print(f"\n💬 [API] Chat request received")
    
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Query must be at least 3 characters long"
        )
    
    try:
        # Query using Parent-Child RAG
        result = rag_system.query(request.query, k=request.k)
        
        # Convert Document objects to serializable dicts
        serializable_sources = []
        for doc in result.get('sources', []):
            serializable_sources.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "full_length": len(doc.page_content)
            })
        
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=serializable_sources,
            num_sources=result.get("num_sources", 0),
            context_chars=result.get("context_chars", 0),
            formatted_sources=result.get("formatted_sources", [])
        )
        
    except Exception as e:
        print(f"❌ [API] Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get vector database and RAG system statistics"""
    if not ingestion_pipeline:
        return StatsResponse(
            children_count=0,
            parents_count=0,
            docstore_parents=0,
            persist_dir=CHROMA_PERSIST_DIR,
            system_ready=False,
            avg_parent_chars=0
        )
    
    stats = ingestion_pipeline.get_stats()
    docstore_stats = get_docstore().get_stats()
    
    return StatsResponse(
        children_count=stats['children_count'],
        parents_count=stats['parents_count'],
        docstore_parents=docstore_stats['total_parents'],
        persist_dir=stats['persist_dir'],
        system_ready=rag_system is not None,
        avg_parent_chars=docstore_stats['avg_chars']
    )


@app.delete("/reset")
def reset_database(confirm: bool = False):
    """
    ⚠️  WARNING: Clear all data from vector database and document store.
    Requires confirm=true parameter for safety.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Must set confirm=true to reset database. This will delete all indexed documents!"
        )
    
    if not ingestion_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Ingestion pipeline not initialized"
        )
    
    try:
        # Get counts before clearing
        stats_before = ingestion_pipeline.get_stats()
        docstore_stats_before = get_docstore().get_stats()
        
        # Clear ChromaDB
        clear_database()
        print("🗑️  [Reset] Vector database cleared")
        
        # Clear docstore
        get_docstore().clear()
        print("🗑️  [Reset] Document store cleared")
        
        # Reinitialize components
        global rag_system
        rag_system = None
        
        print("✅ [Reset] Database reset complete")
        
        return {
            "message": "✅ Database cleared successfully",
            "warning": "All documents have been removed. Re-ingest PDFs to rebuild the database.",
            "parents_removed": docstore_stats_before['total_parents'],
            "children_removed": stats_before['children_count']
        }
    except Exception as e:
        print(f"❌ [Reset] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Detailed health check for monitoring"""
    health_status = {
        "status": "healthy",
        "components": {
            "ingestion_pipeline": ingestion_pipeline is not None,
            "rag_system": rag_system is not None,
            "vector_db": False,
            "doc_store": False
        }
    }
    
    if ingestion_pipeline:
        try:
            stats = ingestion_pipeline.get_stats()
            docstore_stats = get_docstore().get_stats()
            health_status["components"]["vector_db"] = stats['children_count'] > 0
            health_status["components"]["doc_store"] = docstore_stats['total_parents'] > 0
            health_status["database_stats"] = {
                **stats,
                "docstore_parents": docstore_stats['total_parents'],
                "avg_parent_chars": docstore_stats['avg_chars']
            }
        except:
            pass
    
    # Overall status
    all_healthy = all(health_status["components"].values())
    health_status["status"] = "healthy" if all_healthy else "degraded"
    
    return health_status


# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
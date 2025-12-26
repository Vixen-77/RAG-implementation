import os
import shutil
import glob
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import your services
from Services.rag_services import ingest_manual_multimodal, generate_answer_multimodal
from Services.vectorStore import get_collection_stats, clear_database

DATA_FOLDER = "Data"
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
    }
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ✅ Enhanced startup with duplicate prevention and better error handling.
    """
    print("\n" + "="*70)
    print("🚀 MECANIC-IA SYSTEM STARTING")
    print("="*70)
    
    # Startup logic
    if not os.path.exists(DATA_FOLDER):
        print(f"⚠️  [Startup] Folder '{DATA_FOLDER}' missing. Creating it.")
        os.makedirs(DATA_FOLDER)
    
    # Show current database state
    print("\n📊 [Startup] Checking existing database...")
    stats = get_collection_stats()
    
    # Auto-ingest PDFs from Data folder
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print(f"\nℹ️  [Startup] No PDFs found in '{DATA_FOLDER}' folder.")
        print(f"💡 [Tip] Place your vehicle manuals in the '{DATA_FOLDER}' folder or use /ingest endpoint")
    else:
        print(f"\n📚 [Startup] Found {len(pdf_files)} manual(s) in '{DATA_FOLDER}':")
        for pdf in pdf_files:
            print(f"   • {os.path.basename(pdf)}")
        
        print(f"\n⏳ [Startup] Beginning auto-ingestion check...\n")
        
        total_new_chunks = 0
        processed_count = 0
        skipped_count = 0
        
        for pdf_path in pdf_files:
            try:
                print(f"{'─'*70}")
                print(f"📄 Checking: {os.path.basename(pdf_path)}")
                print(f"{'─'*70}")
                
                # ✅ KEY FIX: ingest_manual_multimodal now returns 0 if already indexed
                num_chunks = await ingest_manual_multimodal(pdf_path, force_reingest=False)
                
                if num_chunks > 0:
                    total_new_chunks += num_chunks
                    processed_count += 1
                    print(f"✅ Processed: {os.path.basename(pdf_path)} ({num_chunks} chunks)\n")
                else:
                    skipped_count += 1
                    print(f"⏭️  Skipped: {os.path.basename(pdf_path)} (already indexed)\n")
                
            except Exception as e:
                print(f"❌ [Startup] Failed to process {os.path.basename(pdf_path)}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"🎉 AUTO-INGESTION COMPLETE")
        print(f"📊 Summary:")
        print(f"   • New documents processed: {processed_count}")
        print(f"   • Already indexed (skipped): {skipped_count}")
        print(f"   • New chunks added: {total_new_chunks}")
        print(f"{'='*70}")
    
    # Show final collection stats
    print("\n📊 [Startup] Final Database State:")
    final_stats = get_collection_stats()
    
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
    title="Mecanic-IA API",
    description="Multimodal RAG system for automotive technical assistance",
    version="1.0.0",
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

class ChatResponse(BaseModel):
    answer: str
    source_type: str
    media_content: Optional[List[dict]] = None
    num_sources: Optional[int] = 0

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_processed: int
    already_indexed: bool = False


# ========== Endpoints ==========

@app.get("/")
def read_root():
    """
    Health check endpoint
    """
    stats = get_collection_stats()
    return {
        "status": "✅ System operational",
        "service": "Mecanic-IA Multimodal RAG",
        "documentation": "/docs",
        "demos": "/demos",
        "database_stats": stats
    }


@app.get("/demos")
def list_demos():
    """
    ✅ List available demo scenarios for testing
    """
    return {
        "available_demos": [
            {"id": key, "description": val["description"], "endpoint": f"/demo/{key}"}
            for key, val in DEMO_QUERIES.items()
        ],
        "usage": "Visit /demo/{demo_id} to run a pre-configured scenario"
    }


@app.get("/demo/{demo_id}")
async def run_demo(demo_id: str):
    """
    ✅ Pre-scripted demo endpoint for testing
    
    Usage: GET /demo/engine_smoke
    """
    if demo_id not in DEMO_QUERIES:
        raise HTTPException(
            status_code=404, 
            detail=f"Demo '{demo_id}' not found. Available: {list(DEMO_QUERIES.keys())}"
        )
    
    demo = DEMO_QUERIES[demo_id]
    print(f"\n🎬 [DEMO MODE] Running scenario: {demo['description']}")
    
    try:
        response = await generate_answer_multimodal(demo["query"])
        response["demo_mode"] = True
        response["demo_id"] = demo_id
        response["demo_description"] = demo["description"]
        
        return response
        
    except Exception as e:
        print(f"❌ [Demo] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def upload_manual(file: UploadFile = File(...), force_reingest: bool = False):
    """
    Upload and ingest a new vehicle manual PDF
    
    Args:
        file: PDF file to upload
        force_reingest: Set to True to re-ingest even if already processed
    """
    print(f"\n📥 [API] Received upload: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    try:
        # Save uploaded file
        file_location = f"{DATA_FOLDER}/{file.filename}"
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✅ [API] File saved to {file_location}")
        
        # Ingest the manual
        num_chunks = await ingest_manual_multimodal(file_location, force_reingest=force_reingest)
        
        if num_chunks == 0:
            return IngestResponse(
                message="⏭️ Manual already indexed (use force_reingest=true to re-process)",
                filename=file.filename,
                chunks_processed=0,
                already_indexed=True
            )
        
        return IngestResponse(
            message="✅ Ingestion successful",
            filename=file.filename,
            chunks_processed=num_chunks,
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
    Main chat endpoint - query the vehicle manual
    """
    print(f"\n💬 [API] Chat request received")
    
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Query must be at least 3 characters long"
        )
    
    try:
        response_data = await generate_answer_multimodal(request.query)
        return ChatResponse(**response_data)
        
    except Exception as e:
        print(f"❌ [API] Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/stats")
def get_stats():
    """
    Get vector database statistics
    """
    return get_collection_stats()


@app.delete("/reset")
def reset_database(confirm: bool = False):
    """
    ⚠️  WARNING: Clear all data from vector database
    Requires confirm=true parameter for safety
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Must set confirm=true to reset database. This will delete all indexed documents!"
        )
    
    try:
        clear_database()
        return {
            "message": "✅ Database cleared successfully",
            "warning": "All documents have been removed. Re-ingest PDFs to rebuild the database."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
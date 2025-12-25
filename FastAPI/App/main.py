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

# ========== Hackathon Demo Scenarios ==========
DEMO_QUERIES = {
    "overheating": {
        "query": "My vehicle is overheating, which fuse controls the cooling fan?",
        "description": "Demo: Cooling fan fuse location"
    },
    "tire_pressure": {
        "query": "What should the tire pressure be for a fully loaded vehicle?",
        "description": "Demo: Tire pressure specifications"
    },
    "towing": {
        "query": "Where are the towing points located on this vehicle?",
        "description": "Demo: Towing point locations"
    },
    "oil_change": {
        "query": "How much engine oil does this vehicle need?",
        "description": "Demo: Engine oil specifications"
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ✅ Enhanced startup with better error handling and progress tracking.
    """
    print("\n" + "="*70)
    print("🚀 MECANIC-IA SYSTEM STARTING")
    print("="*70)
    
    # Startup logic
    if not os.path.exists(DATA_FOLDER):
        print(f"⚠️  [Startup] Folder '{DATA_FOLDER}' missing. Creating it.")
        os.makedirs(DATA_FOLDER)
    
    # Auto-ingest PDFs from Data folder
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print(f"\nℹ️  [Startup] No PDFs found in '{DATA_FOLDER}' folder.")
        print(f"💡 [Tip] Place your vehicle manuals in the '{DATA_FOLDER}' folder or use /ingest endpoint")
    else:
        print(f"\n📚 [Startup] Found {len(pdf_files)} manual(s) to process:")
        for pdf in pdf_files:
            print(f"   • {os.path.basename(pdf)}")
        
        print(f"\n⏳ [Startup] Beginning auto-ingestion...\n")
        
        total_chunks = 0
        for pdf_path in pdf_files:
            try:
                print(f"{'─'*70}")
                print(f"📄 Processing: {os.path.basename(pdf_path)}")
                print(f"{'─'*70}")
                
                num_chunks = await ingest_manual_multimodal(pdf_path)
                total_chunks += num_chunks
                
                print(f"✅ Completed: {os.path.basename(pdf_path)} ({num_chunks} chunks)")
                
            except Exception as e:
                print(f"❌ [Startup] Failed to ingest {os.path.basename(pdf_path)}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"🎉 AUTO-INGESTION COMPLETE")
        print(f"📊 Total chunks indexed: {total_chunks}")
        print(f"{'='*70}")
    
    # Show collection stats
    get_collection_stats()
    
    print(f"\n✅ [Startup] API is ready!")
    print(f"📖 Documentation: http://localhost:8000/docs")
    print(f"🎮 Demo scenarios: http://localhost:8000/demos")
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
    ✅ NEW: List available demo scenarios for hackathon judges
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
    ✅ NEW: Pre-scripted demo endpoint for hackathon presentation
    
    Usage: GET /demo/overheating
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
async def upload_manual(file: UploadFile = File(...)):
    """
    Upload and ingest a new vehicle manual PDF
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
        num_chunks = await ingest_manual_multimodal(file_location)
        
        return IngestResponse(
            message="✅ Ingestion successful",
            filename=file.filename,
            chunks_processed=num_chunks
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


@app.post("/reset")
def reset_database():
    """
    ⚠️  WARNING: Clear all data from vector database
    Use for testing only!
    """
    try:
        clear_database()
        return {"message": "✅ Database cleared successfully"}
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
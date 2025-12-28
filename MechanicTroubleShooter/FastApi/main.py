import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from core.config import DATA_FOLDER, CHROMA_PERSIST_DIR
from services import MultimodalIngestionPipeline, ParentChildRAG
from api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[INFO] Starting Server...")
    app.state.ingestion_pipeline = None
    app.state.rag_system = None
    
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    try:
        app.state.ingestion_pipeline = MultimodalIngestionPipeline(persist_dir=CHROMA_PERSIST_DIR)
        print("[INFO] Pipeline ready")
    except Exception as e:
        print(f"[ERROR] Pipeline init failed: {e}")

    try:
        app.state.rag_system = ParentChildRAG(persist_dir=CHROMA_PERSIST_DIR)
        print("[INFO] RAG ready")
    except Exception as e:
        print(f"[ERROR] RAG init failed: {e}")
        app.state.rag_system = None
    
    yield
    print("[INFO] Server Shutdown")

app = FastAPI(title="Mecanic-IA API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
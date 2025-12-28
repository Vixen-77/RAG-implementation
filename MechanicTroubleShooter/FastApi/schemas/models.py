
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    query: str
    k: Optional[int] = 8
    conversation_id: Optional[str] = None  
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

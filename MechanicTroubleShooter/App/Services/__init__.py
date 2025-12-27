
"""
Mecanic-IA Services Package
===========================

Exposes the core services as a clean public API.
"""
from .ingestion.pipeline import MultimodalIngestionPipeline
from .retrieval.rag import ParentChildRAG
from .storage.document import get_docstore
from .storage.vector import vector_db, get_collection_stats, clear_database
from .llm.client import call_ollama, describe_image, generate_chat_answer
from .llm.grader import grade_context_relevance, grade_hallucination, grade_answer_question

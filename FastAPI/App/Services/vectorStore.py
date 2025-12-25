import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIRECTORY = "./chroma_db"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

print("📚 [Init] Loading Embedding Model (all-MiniLM-L6-v2)...")
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}  #  Better for similarity search
)

vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_function,
    collection_name="car_manual_rag"
)

print(" [Init] ChromaDB Connected.")
print(f" [Stats] Current collection size: {vector_db._collection.count()} documents\n")


def add_multimodal_documents(chunks, type="text"):
    """
     Enhanced storage with proper metadata handling for each type.
    
    Args:
        chunks: List of dicts with content and metadata
        type: "text", "table", or "image"
    """
    if not chunks:
        print(f"     No {type} chunks to add, skipping...")
        return

    documents = []
    print(f"💾 [VectorDB] Preparing {len(chunks)} {type} chunks...")

    for idx, chunk in enumerate(chunks):
        try:
            if type == "table":
                doc = Document(
                    page_content=chunk["sentence"],  
                    metadata={
                        "type": "table",
                        "original_html": chunk.get("html", chunk.get("original_html", "")),
                        "page": chunk.get("page", "unknown"),
                        "chunk_id": f"table_{idx}"
                    }
                )
            
            elif type == "image":
                doc = Document(
                    page_content=chunk["description"],  
                    metadata={
                        "type": "image",
                        "image_path": chunk["path"],
                        "page": chunk.get("page", "unknown"),
                        "size_kb": chunk.get("size_kb", 0),
                        "chunk_id": f"image_{idx}"
                    }
                )
            
            else:
                content = chunk.get("content") if isinstance(chunk, dict) else chunk
                
                if isinstance(content, str) and len(content.strip()) > 10:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "type": "text",
                            "page": chunk.get("page", "unknown") if isinstance(chunk, dict) else "unknown",
                            "category": chunk.get("category", "text") if isinstance(chunk, dict) else "text",
                            "chunk_id": f"text_{idx}"
                        }
                    )
                else:
                    continue  
            
            documents.append(doc)
            
        except Exception as e:
            print(f"     Failed to process chunk {idx}: {e}")
            continue

    if documents:
        try:
            vector_db.add_documents(documents)
            print(f"    Committed {len(documents)} {type} vectors to ChromaDB")
        except Exception as e:
            print(f"    Failed to add documents: {e}")
    else:
        print(f"     No valid documents to add")


def search_vector_db(query: str, k=5, filter_type=None):
    """
     Enhanced search with optional type filtering.
    
    Args:
        query: User's question
        k: Number of results to return
        filter_type: Optional filter ("text", "table", "image", or None for all)
    
    Returns:
        List of Document objects with content and metadata
    """
    try:
        if filter_type:
            # Search only specific type
            results = vector_db.similarity_search(
                query, 
                k=k,
                filter={"type": filter_type}
            )
        else:
            # Search across all types
            results = vector_db.similarity_search(query, k=k)
        
        return results
        
    except Exception as e:
        print(f" [Search] Failed: {e}")
        return []


def get_collection_stats():
    """
    Get statistics about the vector database.
    """
    try:
        total_count = vector_db._collection.count()
        
        # Count by type (approximate - requires querying)
        print(f"\n [VectorDB Stats]")
        print(f"   Total documents: {total_count}")
        print(f"   Collection name: {vector_db._collection.name}")
        print(f"   Persist directory: {PERSIST_DIRECTORY}")
        
        return {
            "total": total_count,
            "collection": vector_db._collection.name,
            "directory": PERSIST_DIRECTORY
        }
    except Exception as e:
        print(f" [Stats] Failed: {e}")
        return {}


def clear_database():
    global vector_db                   # <--- Moved to top
    """
     WARNING: Deletes all documents from the collection.
    Use for testing/debugging only.
    """
    try:
        vector_db.delete_collection()
        print(" [VectorDB] Collection cleared")

        # Recreate empty collection
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name="car_manual_rag"
        )
        print(" [VectorDB] New empty collection created")
    except Exception as e:
        print(f" [Clear] Failed: {e}")
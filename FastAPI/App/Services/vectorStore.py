import os
import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIRECTORY = "./chroma_db"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

print("📚 [Init] Loading Embedding Model (all-MiniLM-L6-v2)...")
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}  # ✅ Better for similarity search
)

vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_function,
    collection_name="car_manual_rag"
)

print("📚 [Init] ChromaDB Connected.")
print(f"📊 [Stats] Current collection size: {vector_db._collection.count()} documents\n")


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file to uniquely identify it.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"❌ [Hash] Failed to compute hash: {e}")
        return None


def is_document_indexed(file_path: str) -> bool:
    """
    Check if a document has already been indexed by checking its file hash.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        True if document is already indexed, False otherwise
    """
    file_hash = compute_file_hash(file_path)
    
    if not file_hash:
        return False
    
    try:
        # Search for documents with this file hash in metadata
        results = vector_db.get(
            where={"file_hash": file_hash},
            limit=1
        )
        
        if results and results['ids']:
            print(f"✅ [Check] Document already indexed (hash: {file_hash[:16]}...)")
            return True
        else:
            print(f"🆕 [Check] Document is new (hash: {file_hash[:16]}...)")
            return False
            
    except Exception as e:
        print(f"⚠️  [Check] Could not verify indexing status: {e}")
        return False


def add_multimodal_documents(chunks, type="text", file_path=None):
    """
    ✅ Enhanced storage with proper metadata handling and duplicate prevention.
    
    Args:
        chunks: List of dicts with content and metadata
        type: "text", "table", or "image"
        file_path: Original PDF file path (for tracking indexed documents)
    """
    if not chunks:
        print(f"   ⚠️  No {type} chunks to add, skipping...")
        return

    documents = []
    print(f"💾 [VectorDB] Preparing {len(chunks)} {type} chunks...")
    
    # Compute file hash for duplicate tracking
    file_hash = None
    if file_path:
        file_hash = compute_file_hash(file_path)

    for idx, chunk in enumerate(chunks):
        try:
            # Base metadata (common to all types)
            base_metadata = {
                "type": type,
                "chunk_id": f"{type}_{idx}"
            }
            
            # Add file hash for tracking
            if file_hash:
                base_metadata["file_hash"] = file_hash
                base_metadata["source_file"] = os.path.basename(file_path)
            
            # ========== Handle Tables ==========
            if type == "table":
                doc = Document(
                    page_content=chunk["sentence"],
                    metadata={
                        **base_metadata,
                        "original_html": chunk.get("html", chunk.get("original_html", "")),
                        "page": chunk.get("page", "unknown")
                    }
                )
            
            # ========== Handle Images ==========
            elif type == "image":
                doc = Document(
                    page_content=chunk["description"],
                    metadata={
                        **base_metadata,
                        "image_path": chunk["path"],
                        "page": chunk.get("page", "unknown"),
                        "size_kb": chunk.get("size_kb", 0)
                    }
                )
            
            # ========== Handle Text ==========
            else:
                content = chunk.get("content") if isinstance(chunk, dict) else chunk
                
                if isinstance(content, str) and len(content.strip()) > 10:
                    doc = Document(
                        page_content=content,
                        metadata={
                            **base_metadata,
                            "page": chunk.get("page", "unknown") if isinstance(chunk, dict) else "unknown",
                            "category": chunk.get("category", "text") if isinstance(chunk, dict) else "text"
                        }
                    )
                else:
                    continue  # Skip invalid text chunks
            
            documents.append(doc)
            
        except Exception as e:
            print(f"   ⚠️  Failed to process chunk {idx}: {e}")
            continue

    # ========== Batch Insert ==========
    if documents:
        try:
            vector_db.add_documents(documents)
            print(f"   ✅ Committed {len(documents)} {type} vectors to ChromaDB")
        except Exception as e:
            print(f"   ❌ Failed to add documents: {e}")
    else:
        print(f"   ⚠️  No valid documents to add")


def search_vector_db(query: str, k=5, filter_type=None):
    """
    ✅ Enhanced search with optional type filtering.
    
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
        print(f"❌ [Search] Failed: {e}")
        return []


def get_collection_stats():
    """
    Get statistics about the vector database including indexed documents.
    """
    try:
        total_count = vector_db._collection.count()
        
        # Get list of unique indexed files
        try:
            all_docs = vector_db.get()
            indexed_files = set()
            
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source_file' in metadata:
                        indexed_files.add(metadata['source_file'])
            
            print(f"\n📊 [VectorDB Stats]")
            print(f"   Total documents: {total_count}")
            print(f"   Indexed files: {len(indexed_files)}")
            if indexed_files:
                print(f"   Files: {', '.join(sorted(indexed_files))}")
            print(f"   Collection name: {vector_db._collection.name}")
            print(f"   Persist directory: {PERSIST_DIRECTORY}")
            
            return {
                "total": total_count,
                "indexed_files": list(indexed_files),
                "num_files": len(indexed_files),
                "collection": vector_db._collection.name,
                "directory": PERSIST_DIRECTORY
            }
        except Exception as e:
            print(f"⚠️  Could not retrieve indexed files: {e}")
            return {
                "total": total_count,
                "collection": vector_db._collection.name,
                "directory": PERSIST_DIRECTORY
            }
            
    except Exception as e:
        print(f"❌ [Stats] Failed: {e}")
        return {}


def clear_database():
    """
    ⚠️  WARNING: Deletes all documents from the collection.
    Use for testing/debugging only.
    """
    global vector_db
    
    try:
        vector_db.delete_collection()
        print("🗑️  [VectorDB] Collection cleared")
        
        # Recreate empty collection
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name="car_manual_rag"
        )
        print("✅ [VectorDB] New empty collection created")
        
    except Exception as e:
        print(f"❌ [Clear] Failed: {e}")


def get_indexed_documents():
    """
    Get a list of all indexed document file hashes and names.
    
    Returns:
        List of tuples (file_hash, source_file)
    """
    try:
        all_docs = vector_db.get()
        indexed_docs = {}
        
        if all_docs and 'metadatas' in all_docs:
            for metadata in all_docs['metadatas']:
                if metadata and 'file_hash' in metadata and 'source_file' in metadata:
                    file_hash = metadata['file_hash']
                    source_file = metadata['source_file']
                    indexed_docs[file_hash] = source_file
        
        return [(hash_val, filename) for hash_val, filename in indexed_docs.items()]
        
    except Exception as e:
        print(f"❌ [Get Indexed] Failed: {e}")
        return []
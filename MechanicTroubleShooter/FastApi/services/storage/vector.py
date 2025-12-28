import os
import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIRECTORY = "./chroma_db"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            print(" [GPU] CUDA detected")
            return 'cuda'
    except ImportError:
        pass
    print(" [CPU] Running embeddings on CPU")
    return 'cpu'

DEVICE = get_device()

print(" [Init] Loading Embedding Model (all-MiniLM-L6-v2)...")
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': DEVICE},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 64
    }
)

vector_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_function,
    collection_name="car_manual_rag"
)

print(" [Init] ChromaDB Connected.")
print(f" [Stats] Current collection size: {vector_db._collection.count()} documents\n")


def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f" [Hash] Failed to compute hash: {e}")
        return None


def is_document_indexed(file_path: str) -> bool:
    file_hash = compute_file_hash(file_path)
    
    if not file_hash:
        return False
    
    try:
        results = vector_db.get(
            where={"file_hash": file_hash},
            limit=1
        )
        
        if results and results['ids']:
            print(f" [Check] Document already indexed (hash: {file_hash[:16]}...)")
            return True
        else:
            print(f" [Check] Document is new (hash: {file_hash[:16]}...)")
            return False
            
    except Exception as e:
        print(f" [Check] Could not verify indexing status: {e}")
        return False


def add_multimodal_documents(chunks, file_path=None):

    if not chunks:
        print(f"    No chunks to add, skipping...")
        return

    documents = []
    print(f" [VectorDB] Preparing {len(chunks)} chunks...")
    
    file_hash = None
    if file_path:
        file_hash = compute_file_hash(file_path)

    for idx, chunk in enumerate(chunks):
        try:
            content = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            if not content or len(content.strip()) < 10:
                continue
            
            if file_hash and "file_hash" not in metadata:
                metadata["file_hash"] = file_hash
            
            if file_path and "source_file" not in metadata:
                metadata["source_file"] = os.path.basename(file_path)
            
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = f"chunk_{idx}"
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            documents.append(doc)
            
        except Exception as e:
            print(f"    Failed to process chunk {idx}: {e}")
            continue

    BATCH_SIZE = 5000
    
    if documents:
        try:
            total_docs = len(documents)
            
            if total_docs <= BATCH_SIZE:
                vector_db.add_documents(documents)
                print(f"    Committed {total_docs} vectors to ChromaDB")
            else:
                print(f"    Large batch detected ({total_docs} docs), splitting into chunks of {BATCH_SIZE}...")
                
                for i in range(0, total_docs, BATCH_SIZE):
                    batch = documents[i:i + BATCH_SIZE]
                    vector_db.add_documents(batch)
                    print(f"       Batch {i // BATCH_SIZE + 1}: Committed {len(batch)} vectors")
                
                print(f"    Total committed: {total_docs} vectors to ChromaDB")
                
        except Exception as e:
            print(f"    Failed to add documents: {e}")
            import traceback
            traceback.print_exc()
    


def search_vector_db(query: str, k=5, filter_type=None, section_codes=None):
    
    try:
        filter_conditions = []
        
        if filter_type:
            filter_conditions.append({"type": filter_type})
        
        if section_codes:
            filter_conditions.append({"section_code": {"$in": section_codes}})
        
        if len(filter_conditions) > 1:
            combined_filter = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            combined_filter = filter_conditions[0]
        else:
            combined_filter = None
        
        if combined_filter:
            results = vector_db.similarity_search(
                query, 
                k=k,
                filter=combined_filter
            )
            print(f"    [Search] Applied filter: {combined_filter}")
        else:
            results = vector_db.similarity_search(query, k=k)
        
        print(f"    [Search] Found {len(results)} results")
        return results
        
    except Exception as e:
        print(f" [Search] Failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_collection_stats():
    try:
        total_count = vector_db._collection.count()
        
        try:
            all_docs = vector_db.get()
            indexed_files = set()
            
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source_file' in metadata:
                        indexed_files.add(metadata['source_file'])
            
            print(f"\n [VectorDB Stats]")
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
            print(f" Could not retrieve indexed files: {e}")
            return {
                "total": total_count,
                "collection": vector_db._collection.name,
                "directory": PERSIST_DIRECTORY
            }
            
    except Exception as e:
        print(f" [Stats] Failed: {e}")
        return {}


def clear_database():
    global vector_db
    
    try:
        vector_db.delete_collection()
        print("🗑️ [VectorDB] Collection cleared")
        
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name="car_manual_rag"
        )
        print(" [VectorDB] New empty collection created")
        
    except Exception as e:
        print(f" [Clear] Failed: {e}")


def get_indexed_documents():
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
        print(f" [Get Indexed] Failed: {e}")
        return []


def delete_documents_by_source(source_file: str) -> dict:
    """Delete all documents from the vector database that came from a specific source file."""
    try:
        all_docs = vector_db.get(
            where={"source_file": source_file}
        )
        
        if not all_docs or not all_docs['ids']:
            print(f" [Delete] No documents found for source: {source_file}")
            return {"deleted": 0, "source_file": source_file, "status": "not_found"}
        
        ids_to_delete = all_docs['ids']
        count = len(ids_to_delete)
        
        print(f" [Delete] Found {count} documents from '{source_file}'")
        
        vector_db._collection.delete(ids=ids_to_delete)
        
        print(f" [Delete] Successfully removed {count} embeddings from '{source_file}'")
        
        return {
            "deleted": count,
            "source_file": source_file,
            "status": "success"
        }
        
    except Exception as e:
        print(f" [Delete] Failed: {e}")
        return {"deleted": 0, "source_file": source_file, "status": "error", "error": str(e)}
    

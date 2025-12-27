import chromadb
import random
import json

# CONFIGURATION
# Make sure this path is correct for your setup
DB_PATH = "./chroma_db"
COLLECTION_NAME = "car_manual_rag"

def inspect_random_chunk():
    print(f"🔌 Connecting to ChromaDB at: {DB_PATH}")
    
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        total_count = collection.count()
        print(f"✅ Collection loaded. Total Chunks: {total_count}")
    except Exception as e:
        print(f"❌ Error connecting: {e}")
        return

    if total_count == 0:
        print("⚠️ Collection is empty.")
        return

    # Pick a random index
    random_index = random.randint(0, total_count - 1)
    
    print(f"\n🎲 Picking random chunk at index: {random_index}")
    print("="*60)

    # Fetch the chunk
    # Note: Using .get() with limit=1 and offset=random_index
    try:
        result = collection.get(limit=1, offset=random_index)
        
        if not result['ids']:
            print("❌ Failed to retrieve chunk.")
            return

        # Extract Data
        chunk_id = result['ids'][0]
        # Handle cases where document/metadata might be None
        text = result['documents'][0] if result['documents'][0] else "[NO TEXT]"
        metadata = result['metadatas'][0] if result['metadatas'][0] else {}

        # DISPLAY RESULTS
        print(f"🆔 Chunk ID: {chunk_id}")
        
        print(f"\n📄 Text Content (First 300 chars):")
        print("-" * 30)
        print(text[:300].replace('\n', ' ') + "...")
        print("-" * 30)

        print(f"\n🏷️  METADATA (The Tags):")
        print("-" * 30)
        # Pretty print the metadata dictionary
        print(json.dumps(metadata, indent=4))
        print("-" * 30)

        # QUICK ANALYSIS
        section = metadata.get('section_code', 'unknown')
        system = metadata.get('system_context', 'General')
        
        print(f"\n🤖 RAG Analysis:")
        if section != 'unknown':
            print(f"✅ GOOD CHUNK: This belongs to Section {section} ({system}).")
        else:
            print(f"⚠️ GENERAL CHUNK: This is untagged or generic info (Page {metadata.get('page', '?')}).")

    except Exception as e:
        print(f"❌ Error during fetch: {e}")

if __name__ == "__main__":
    inspect_random_chunk()

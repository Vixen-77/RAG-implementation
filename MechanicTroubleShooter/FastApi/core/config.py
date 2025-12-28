import os

DATA_FOLDER = "Data"
CHROMA_PERSIST_DIR = "./chroma_parent_child" 
CHROMA_DB_DIR = "./chroma_db"

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "llama3.1"
VISION_MODEL = "llava-phi3"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


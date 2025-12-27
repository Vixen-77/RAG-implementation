
import requests
import base64
import os
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava-phi3"
CACHE_FILE = "./chroma_db/image_captions_cache.json"

def call_ollama(prompt: str, model: str, image_path: str = None) -> str:
    """Call Ollama API."""
    print(f"[LLM] Calling {model}...")
    
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
            payload["images"] = [b64]

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Ollama failed: {e}")
        return f"Error: {str(e)}"

def describe_image(image_path: str, page_num: int = None) -> str:
    file_hash = _get_file_hash(image_path)
    cache = _load_cache()
    if file_hash in cache:
        print(f"[VISION] Cache hit: {os.path.basename(image_path)}")
        return cache[file_hash]

    prompt = "Describe this automotive diagram. detailed. List components."
    res = call_ollama(prompt, model=VISION_MODEL, image_path=image_path)
    
    # Save cache
    cache[file_hash] = res
    _save_cache(cache)
    return res

def generate_chat_answer(context: str, user_question: str, routing_info: dict = None) -> str:
    

    prompt = f"""You are a Dacia workshop technician. Answer ONLY using the CONTEXT from the official manual.

CONTEXT:
{context}

QUESTION:
{user_question}

STRICT RULES:
1. Use ONLY information that is clearly present in CONTEXT.
2. Do NOT guess causes (battery, starter, fuel pump, ECU, etc.) unless those words appear in CONTEXT.
3. If CONTEXT only says to retry starting and then contact a dealer if it still fails, do NOT add extra diagnostic steps.
4. If the manual sections do NOT explain the cause, say clearly:
   "The manual sections here only say to contact an approved dealer if the engine does not start after several attempts."
5. It is better to say "the manual does not specify further causes in these sections" than to invent explanations.

ANSWER:
"""

    raw = call_ollama(prompt, model="llama3.2")
    answer = raw.strip()

    # Light post-processing: remove leading filler
    for prefix in [
        "BASED ON THE CONTEXT,",
        "ACCORDING TO THE CONTEXT,",
        "ACCORDING TO THE MANUAL,",
        "FROM THE CONTEXT,",
    ]:
        if answer.upper().startswith(prefix):
            answer = answer[len(prefix):].lstrip()

    return answer



def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def _save_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f: json.dump(cache, f)
    except: pass

def _get_file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f: h.update(f.read())
    return h.hexdigest()

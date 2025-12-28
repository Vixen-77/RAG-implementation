import requests
import base64
import os
import json
import hashlib

from core.config import OLLAMA_URL, TEXT_MODEL, VISION_MODEL

CACHE_FILE = "./chroma_db/image_captions_cache.json"


def call_ollama(prompt: str, model: str = None, image_path: str = None) -> str:
    model = model or TEXT_MODEL
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            payload["images"] = [base64.b64encode(f.read()).decode('utf-8')]

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Ollama failed: {e}")
        return f"Error: {str(e)}"


def stream_ollama(prompt: str, model: str = None):
    model = model or TEXT_MODEL
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
    except Exception as e:
        print(f"[ERROR] Streaming failed: {e}")
        yield f"Error: {str(e)}"


def describe_image(image_path: str, page_num: int = None) -> str:
    file_hash = _get_file_hash(image_path)
    cache = _load_cache()
    
    if file_hash in cache:
        print(f"[VISION] Cache hit: {os.path.basename(image_path)}")
        return cache[file_hash]

    prompt = "Describe this automotive diagram. detailed. List components."
    result = call_ollama(prompt, model=VISION_MODEL, image_path=image_path)
    
    cache[file_hash] = result
    _save_cache(cache)
    return result


def generate_chat_answer(context: str, question: str, history: list = None) -> str:
    prompt = _build_rag_prompt(context, question, history)
    return call_ollama(prompt)


def stream_chat_answer(context: str, question: str, history: list = None):
    prompt = _build_rag_prompt(context, question, history)
    yield from stream_ollama(prompt)


def _build_rag_prompt(context: str, question: str, history: list = None) -> str:
    history_text = ""
    if history:
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ])
        history_text = f"\nCONVERSATION HISTORY:\n{history_text}\n"

    return f"""You are a Dacia workshop technician. Answer ONLY using the CONTEXT from the official manual.
{history_text}
CONTEXT:
{context}

QUESTION:
{question}

RULES:
1. Use ONLY information present in CONTEXT.
2. Do NOT guess causes unless mentioned in CONTEXT.
3. If unsure, say "the manual does not specify this".
4. Reference previous conversation when relevant.

ANSWER:
"""


def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass


def _get_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

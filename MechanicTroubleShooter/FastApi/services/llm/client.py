import requests
import base64
import os
import json
import hashlib
import re

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

    prompt = """Analyze this automotive manual image and provide a structured description.

FORMAT YOUR RESPONSE EXACTLY AS:
CATEGORY: [One of: engine, transmission, brakes, suspension, electrical, dashboard, interior, exterior, body, wheels, steering, exhaust, cooling, fuel system, climate control, airbags, sensors, wiring, diagram, other]
COMPONENTS: [List the main visible components/parts, comma-separated]
DESCRIPTION: [One sentence describing what this image shows and its purpose]

Be specific about automotive systems. If it's a diagram, identify what system it documents.
If it shows the dashboard or interior, specify what controls or features are visible.
If it shows an engine or mechanical parts, name the specific components."""

    result = call_ollama(prompt, model=VISION_MODEL, image_path=image_path)
    
    formatted = _format_image_caption(result, page_num)
    
    cache[file_hash] = formatted
    _save_cache(cache)
    return formatted


def _format_image_caption(raw_result: str, page_num: int = None) -> str:
    """Format vision model output into a searchable caption."""
    lines = raw_result.strip().split('\n')
    
    category = "automotive"
    components = ""
    description = raw_result  
    
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith('category:'):
            category = line.split(':', 1)[1].strip()
        elif line_lower.startswith('components:'):
            components = line.split(':', 1)[1].strip()
        elif line_lower.startswith('description:'):
            description = line.split(':', 1)[1].strip()
    
    # Build searchable caption
    parts = []
    parts.append(f"[{category.upper()}]")
    if components:
        parts.append(f"Components: {components}.")
    parts.append(description)
    if page_num:
        parts.append(f"(Page {page_num})")
    
    return " ".join(parts)


def evaluate_image_relevance(query: str, image_caption: str) -> bool:
    """
    Ask the LLM to evaluate if an image is relevant to the user's query.
    Returns True if relevant, False otherwise.
    """
    prompt = f"""You are evaluating whether an image is relevant to a user's query.

USER QUERY: {query}

IMAGE DESCRIPTION: {image_caption}

Is this image DIRECTLY relevant and useful for answering the user's query?
Answer with ONLY "YES" or "NO".

- Answer YES only if the image shows exactly what the user is asking about
- Answer NO if the image is about a different topic or unrelated system
- Be strict: a dashboard image is NOT relevant to an engine question"""

    try:
        response = call_ollama(prompt).strip().upper()
        is_relevant = response.startswith("YES")
        print(f"[LLM] Image relevance: {response} for '{image_caption[:50]}...'")
        return is_relevant
    except Exception as e:
        print(f"[LLM] Relevance check failed: {e}")
        return False  # Default to not showing if evaluation fails


def generate_chat_answer(context: str, question: str, history: list = None) -> str:
    prompt = _build_rag_prompt(context, question, history)
    return call_ollama(prompt)


def stream_chat_answer(context: str, question: str, history: list = None):
    prompt = _build_rag_prompt(context, question, history)
    yield from stream_ollama(prompt)


def _build_rag_prompt(context: str, question: str, history: list = None) -> str:
    # Clean OCR artifacts from context at runtime
    context = _clean_text(context)
    
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
5. If CONTEXT contains image descriptions (marked as "image" sources), reference them naturally. The images WILL be displayed to the user alongside your response, so you can say "As shown in the image below" or "The diagram shows...".

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


def _clean_text(text: str) -> str:
    """
    Clean OCR/PDF tokenization artifacts at runtime.
    Fixes issues like "I 'm a D acia" -> "I'm a Dacia"
    """
    if not text:
        return text
    
    # Fix contractions with spaces: "I 'm" -> "I'm", "I 'll" -> "I'll"
    text = re.sub(r"(\w)\s+'(\w)", r"\1'\2", text)
    
    # Fix spaces around hyphens
    text = re.sub(r'\s*-\s*', '-', text)
    
    # Fix single letter followed by space then word (e.g., "D acia" -> "Dacia")
    text = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', text)
    
    # Fix common split words with lowercase
    text = re.sub(r'(\w{2,})\s+([a-z]{2,}(?:tion|ment|ing|ness|able|ible|ure|ous|ive|ect|oot|ose|age|ance|ence))\b', r'\1\2', text)
    
    # Fix broken abbreviations
    abbreviations = {
        r'O\s*B\s*D': 'OBD', r'A\s*B\s*S': 'ABS', r'E\s*S\s*P': 'ESP',
        r'E\s*C\s*U': 'ECU', r'D\s*T\s*C': 'DTC'
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    
    # Fix multiple spaces
    text = re.sub(r'  +', ' ', text)
    
    # Fix space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    return text


import requests
import base64
import os
import json
import time

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "llama3.2"
VISION_MODEL = "llava-phi3"

def call_ollama(prompt: str, model: str, image_path: str = None) -> str:
    """
    Call Ollama API with text or vision model.
    
    Args:
        prompt: The prompt to send
        model: Model name (llama3.2 or llava-phi3)
        image_path: Optional path to image for vision model
        
    Returns:
        Model's response as string
    """
    start_time = time.time()

    # Log the request
    msg = f"📡 [Ollama] Calling {model}..."
    if image_path:
        msg += f" (with Image: {os.path.basename(image_path)})"
    print(msg)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  
            "num_ctx": 4096,        
            "num_predict": 500,     
            "top_p": 0.9,
            "top_k": 20
        }
    }

    if image_path:
        if not os.path.exists(image_path):
            print(f"❌ [Ollama] Image not found: {image_path}")
            return "Error: Image missing"
        try:
            with open(image_path, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode('utf-8')
                payload["images"] = [b64]
        except Exception as e:
            print(f"❌ [Ollama] Image encoding failed: {e}")
            return f"Error: {e}"

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json().get("response", "").strip()

        # Log success and duration
        duration = round(time.time() - start_time, 2)
        print(f"✅ [Ollama] Success ({duration}s)")
        return result

    except Exception as e:
        print(f"❌ [Ollama] Failed: {e}")
        return f"Error: {str(e)}"


    return result


# --- Cache utilities ---
CACHE_FILE = "./chroma_parent_child/image_captions_cache.json"

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

def get_file_hash(file_path):
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def describe_image(image_path: str, page_num: int = None) -> str:
    """
    Generate description for a technical automotive diagram.
    Uses caching to avoid re-generating descriptions for same images.
    
    Args:
        image_path: Path to the image file
        page_num: Optional page number for context
        
    Returns:
        Detailed description of the image
    """
    # Check cache first
    try:
        file_hash = get_file_hash(image_path)
        cache = _load_cache()
        if file_hash in cache:
            print(f"   ⚡ [Vision] Cache hit for {os.path.basename(image_path)}")
            return cache[file_hash]
    except Exception as e:
        print(f"   ⚠️ [Vision] Cache check failed: {e}")
        file_hash = None

    page_info = f" from page {page_num}" if page_num else ""
    
    prompt = f"""Analyze this technical automotive diagram{page_info}. Provide a detailed description including:
1. DIAGRAM TYPE: (wiring schematic, fuse box layout, component location, assembly diagram, etc.)
2. COMPONENTS SHOWN: List all visible parts, their labels, and numbers (e.g., "Fuse #32", "Bolt position 5")
3. PURPOSE: What is this diagram used for? (e.g., "locating cooling fan fuse", "towing point access")
4. KEY INFORMATION: Any torque specs, part numbers, or measurements visible

Format your response clearly and include all visible labels/numbers.

Example: "Fuse box diagram showing engine compartment layout. Fuse #32 (30A) controls cooling fan. Fuse #15 controls headlights. Used for electrical troubleshooting."
"""
    
    result = call_ollama(prompt, model=VISION_MODEL, image_path=image_path)
    
    # Validate response quality
    if len(result) < 50 or "error" in result.lower():
        print(f"⚠️ [Vision] Low quality description, retrying...")
        # Fallback to simpler prompt
        result = call_ollama(
            "Describe this automotive technical diagram in detail. Include all visible labels and numbers.",
            model=VISION_MODEL,
            image_path=image_path
        )
    
    # Save to cache
    if file_hash:
        try:
            cache = _load_cache() # Reload in case of concurrent writes (simple race condition handling)
            cache[file_hash] = result
            _save_cache(cache)
        except Exception as e:
            print(f"   ⚠️ [Vision] Cache save failed: {e}")

    return result


def generate_chat_answer(context: str, user_question: str, routing_info: dict = None) -> str:
    """
    Generate answer with ENFORCED grounding.
    """
    
    # Build minimal, strict prompt
    prompt = f"""You are a Dacia workshop technician. Answer ONLY using the CONTEXT below.

CONTEXT:
{context}

QUESTION: {user_question}

STRICT RULES:
1. If the context does NOT contain the answer, say: "The manual does not provide information about [topic]."
2. Never invent part numbers, page numbers, or section codes.
3. Cite the source number for each fact: [Source 1], [Source 2]

ANSWER:"""
    
    response = call_ollama(prompt, model="llama3.2")
    
    # Post-process: Remove common hallucination patterns
    response = response.strip()
    
    # Remove phrases that indicate guessing
    response = response.replace("Based on general knowledge,", "")
    response = response.replace("typically", "")
    response = response.replace("usually", "")
    
    # Validate citations - remove if they look fake
    import re
    # Remove citations like "Page 1234" or "Section 99Z" (clearly fake)
    if re.search(r'Page \d{4,}', response) or re.search(r'Section \d{2}[A-Z]', response):
        print("[LLM] ⚠️ Detected suspicious citations - stripping")
        response = re.sub(r'\[Source \d+, Page \d+, Section [A-Z0-9]+\]', '', response)
    
    return response

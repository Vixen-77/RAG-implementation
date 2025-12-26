import requests
import base64
import os
import time

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "llama3.2"
VISION_MODEL = "llava-phi3"

def call_ollama(prompt: str, model: str, image_path: str = None) -> str:
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
            "temperature": 0.2,  
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


def describe_image(image_path: str, page_num: int = None) -> str:
    
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
    
    return result


def generate_chat_answer(context: str, user_question: str) -> str:
    """
    Generate answer based on retrieved context.
    Enhanced prompt for technical accuracy.
    """
    prompt = f"""You are Mecanic-IA, an expert automotive technical assistant.

CONTEXT (Retrieved from vehicle manual):
{context}

USER QUESTION: {user_question}

INSTRUCTIONS:
1. Answer ONLY using information from the CONTEXT above
2. Be specific - include exact part numbers, torque values, fuse numbers, page references
3. If context mentions a diagram or table, reference it clearly
4. If you cannot answer from the context, say "I don't have that information in the manual"
5. Keep answer concise but complete (2-4 sentences)

ANSWER:"""
    
    response = call_ollama(prompt, model=TEXT_MODEL)
    
    # Post-process: Remove common LLM artifacts
    response = response.replace("Based on the context,", "").strip()
    response = response.replace("According to the manual,", "").strip()
    
    return response
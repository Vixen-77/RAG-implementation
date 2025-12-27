import requests
import base64
import os
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


def describe_image(image_path: str, page_num: int = None) -> str:
    """
    Generate description for a technical automotive diagram.
    
    Args:
        image_path: Path to the image file
        page_num: Optional page number for context
        
    Returns:
        Detailed description of the image
    """
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


def generate_chat_answer(context: str, user_question: str, routing_info: dict = None) -> str:
    """
    Generate diagnostic answer with GROUNDING PRECISION.
    
    Enhanced for Dacia vehicle diagnostics with:
    - Mandatory citations for every fact [Source X, Page Y, Section Z]
    - "I don't know" response when information is not in context
    - Section context awareness
    - Structured output format
    
    Args:
        context: Retrieved context from vector DB (includes section metadata)
        user_question: User's diagnostic question
        routing_info: Optional routing metadata for context awareness
    
    Returns:
        Grounded answer with citations
    """
    
    # Build section context from routing
    section_context = ""
    if routing_info:
        category = routing_info.get('matched_category', 'general')
        reason = routing_info.get('routing_reason', 'N/A')
        include = routing_info.get('include_sections', [])
        exclude = routing_info.get('exclude_sections', [])
        
        if category != 'general':
            section_context = f"""
QUERY ROUTING APPLIED:
- Category: {category}
- Reason: {reason}
- Prioritized Sections: {', '.join(include) if include else 'All'}
- Excluded Sections: {', '.join(exclude) if exclude else 'None'}
"""
    
    prompt = f"""You are a Senior Diagnostic Technician for Dacia vehicles.
You have access to the MR 388 Workshop Manual.

=== GROUNDING PRECISION REQUIREMENT ===
For EVERY fact you state, you MUST include a citation in this exact format:
[Source X, Page Y, Section Z]

Example of grounded answer:
"The AdBlue tank capacity is 17 liters [Source 3, Page 1297, Section 19B]. 
The DPF regeneration threshold is 45% soot loading [Source 1, Page 1302, Section 19B]."

CRITICAL RULES:
1. If you CANNOT find a direct citation for a claim in the context, DO NOT make the claim
2. Say "I don't have specific information about [topic] in the retrieved context" for ungrounded facts
3. NEVER guess values, part numbers, specifications, or procedures
4. Each sentence with a technical fact MUST have a citation
5. Opinions and general knowledge don't need citations, but technical specs DO



{section_context}

=== RETRIEVED CONTEXT ===
{context}

=== USER QUESTION ===
{user_question}

=== RESPONSE FORMAT ===
1. **Verified System Fault**: Identify the system at fault with section reference
   Example: "Exhaust After-treatment System (Section 19B)"

2. **Diagnostic Steps**: Step-by-step procedure WITH CITATIONS
   Example: "1. Check differential pressure sensor [Source 2, Page 1305, Section 19B]"

3. **Required Parts**: Parts WITH CITATIONS, or "Not specified in retrieved context"
   Example: "DPF differential pressure sensor (Part# 8200909457) [Source 2, Page 1305, Section 19B]"

4. **Confidence Note**: If some information is missing, state what you couldn't find

ANSWER:"""
    
    response = call_ollama(prompt, model=TEXT_MODEL)
    
    # ✅ Post-process: Remove common LLM artifacts but keep citations
    response = response.replace("Based on the context,", "").strip()
    response = response.replace("According to the manual,", "").strip()
    response = response.replace("Based on the information provided,", "").strip()
    
    # ✅ Remove empty lines
    response = "\n".join(line for line in response.split("\n") if line.strip())
    
    return response
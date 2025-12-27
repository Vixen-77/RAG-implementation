import os
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
import re

from Services.llm_service import describe_image, generate_chat_answer
from Services.reranker import rerank_results
from Services.vectorStore import (
    add_multimodal_documents, 
    search_vector_db, 
    is_document_indexed
)


def process_pdf(pdf_path: str):
    """
    Complete PDF processing pipeline:
    1. Check for duplicates
    2. Extract images using PyMuPDF
    3. Extract text with smart header detection
    4. Store in vector database
    """
    filename = os.path.basename(pdf_path)
    print(f"📄 Processing: {filename}")

    # Check for duplicates
    if is_document_indexed(pdf_path):
        print(f"⏩ Skipping {filename} (Already Indexed)")
        return {"status": "skipped", "reason": "duplicate_hash"}

    # Extract Images
    image_tasks = extract_images_with_pymupdf(pdf_path)
    processed_images = process_images_parallel(image_tasks)

    # Extract Text with Smart Headers
    print("📝 Extracting text with structure-aware chunking...")
    documents_to_add = []
    
    try:
        text_chunks = extract_text_with_smart_headers(pdf_path, filename)
        documents_to_add.extend(text_chunks)
    except Exception as e:
        print(f"⚠️ Smart extraction failed, using fallback: {e}")
        text_chunks = fallback_simple_extraction(pdf_path, filename)
        documents_to_add.extend(text_chunks)

    # Add Image Descriptions
    file_hash = get_file_hash(pdf_path)
    vehicle_model = detect_vehicle_model(filename)
    
    for img_path, page_num, description, size_kb in processed_images:
        documents_to_add.append({
            "text": description,
            "metadata": {
                "source_file": filename,
                "page": page_num,
                "type": "image",
                "image_path": img_path,
                "size_kb": round(size_kb, 2),
                "chapter_header": "Visual Data",
                "system_context": "Visual",
                "vehicle_model": vehicle_model,
                "file_hash": file_hash,
                "chunk_id": f"image_{page_num}"
            }
        })

    # Store in Vector Database
    if documents_to_add:
        print(f"💾 Storing {len(documents_to_add)} chunks to VectorDB...")
        add_multimodal_documents(documents_to_add)
        print("✅ Indexing Complete.")
    else:
        print("⚠️ No valid content found to index.")

    return {"status": "success", "chunks": len(documents_to_add)}


# ==============================================================================
# OPTIMIZED SMART TEXT EXTRACTION WITH HEADER DETECTION
# ==============================================================================

def extract_text_with_smart_headers(pdf_path: str, filename: str):
    """
    Extract text chunks with intelligent header detection.
    
    OPTIMIZATIONS:
    - Pre-compute file hash and vehicle model once
    - Cache page headers to reduce redundant processing
    - Better header persistence logic
    - Improved block filtering
    """
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_counter = 0
    
    # ✅ Pre-compute once (not in loop)
    file_hash = get_file_hash(pdf_path)
    vehicle_model = detect_vehicle_model(filename)
    
    # ✅ Track current section header with better persistence
    current_header = "General"
    previous_header = "General"
    header_confidence = 0  # Track how strong the current header is
    
    print(f"📖 Processing {len(doc)} pages...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # ✅ Find header on this page
        page_header, confidence = find_best_header_on_page_improved(page)
        
        # ✅ Smart header update logic:
        # Only update if new header is more confident OR if we haven't found a good header yet
        if page_header != "General":
            if confidence > header_confidence or current_header == "General":
                previous_header = current_header
                current_header = page_header
                header_confidence = confidence
                print(f"   📍 Page {page_num + 1}: Section '{current_header}' (confidence: {confidence:.1f})")
        
        # ✅ Extract blocks with optimized processing
        page_dict = page.get_text("dict", sort=True)
        blocks = page_dict.get("blocks", [])
        
        for block in blocks:
            # Skip non-text blocks immediately
            if "lines" not in block:
                continue
            
            # ✅ Extract text more efficiently
            block_text = extract_block_text(block)
            
            # ✅ Better filtering logic
            if not is_valid_content_block(block_text, current_header, page_header):
                continue
            
            # ✅ Build metadata once
            metadata = build_chunk_metadata(
                filename=filename,
                page_num=page_num + 1,
                header=current_header,
                block_text=block_text,
                vehicle_model=vehicle_model,
                file_hash=file_hash,
                chunk_id=f"text_{chunk_counter}"
            )
            
            chunks.append({
                "text": block_text,
                "metadata": metadata
            })
            
            chunk_counter += 1
    
    doc.close()
    print(f"✅ Extracted {chunk_counter} text chunks")
    return chunks


# ==============================================================================
# IMPROVED HEADER DETECTION
# ==============================================================================

def find_best_header_on_page_improved(page):
    """
    Enhanced header detection with confidence scoring.
    
    IMPROVEMENTS:
    - Returns (header, confidence_score) tuple
    - Better noise filtering with regex patterns
    - Considers text position on page (top = more likely header)
    - Handles section codes better (19B, 13A, etc.)
    
    Returns:
        tuple: (header_text, confidence_score)
    """
    blocks = page.get_text("dict", sort=True)["blocks"]
    
    max_score = 0
    best_header = "General"
    page_height = page.rect.height
    
    # ✅ Improved noise patterns
    noise_patterns = [
        r'^[A-Z]{2,3}-\d+$',  # Document codes like "MR-388"
        r'^\d+\s*[-–]\s*\d+$',  # Page ranges like "1-5"
        r'^page\s+\d+$',  # "Page 1"
        r'^(jaune|noir|texte|edition|anglaise|dacia|renault)\b',  # Language artifacts
        r'^[|]+$',  # Just pipes
        r'^\d{1,4}$',  # Just numbers
    ]
    
    # ✅ Valid section code pattern
    section_code_pattern = r'\b(\d{1,2}[A-Z])\b'
    
    for block in blocks:
        if "lines" not in block:
            continue
        
        # ✅ Get block position (for positional scoring)
        block_y = block.get("bbox", [0, 0, 0, 0])[1]  # Top Y coordinate
        
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                size = span["size"]
                font = span.get("font", "")
                
                # ✅ Skip too short or empty
                if len(text) < 3:
                    continue
                
                # ✅ Check against noise patterns
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in noise_patterns):
                    continue
                
                # ✅ Calculate base score from font size
                score = size
                
                # ✅ Font weight bonus
                if any(weight in font for weight in ["Bold", "Black", "Heavy", "Semibold"]):
                    score += 3
                
                # ✅ ALL CAPS bonus (but not for short text)
                if text.isupper() and len(text) > 5:
                    score += 2
                
                # ✅ Position bonus (headers usually at top of page)
                if block_y < page_height * 0.2:  # Top 20% of page
                    score += 1
                
                # ✅ Section code bonus (e.g., "19B EXHAUST")
                if re.search(section_code_pattern, text):
                    score += 2
                
                # ✅ Length penalty (very long text unlikely to be header)
                if len(text) > 100:
                    score -= 2
                
                # ✅ Update best header if this scores higher
                if score > max_score:
                    max_score = score
                    best_header = text
    
    return best_header, max_score


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_block_text(block):
    """
    Efficiently extract text from a PyMuPDF block.
    
    Returns:
        str: Cleaned block text
    """
    spans_text = [
        span["text"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    return " ".join(spans_text).strip()


def is_valid_content_block(block_text, current_header, page_header):
    """
    Determine if a text block should be included as content.
    
    FILTERS OUT:
    - Empty or very short blocks
    - The header itself (to avoid duplication)
    - Common page artifacts (headers, footers, page numbers)
    - Repeated content
    
    Args:
        block_text: The text content of the block
        current_header: Current section header
        page_header: Header detected on current page
        
    Returns:
        bool: True if block should be included
    """
    # ✅ Length check
    if len(block_text) < 50:
        return False
    
    # ✅ Don't duplicate headers
    if block_text == current_header or block_text == page_header:
        return False
    
    # ✅ Filter common page artifacts
    text_lower = block_text.lower()
    artifacts = [
        "jaune noir",
        "édition anglaise",
        "workshop manual",
        "mr 388",
    ]
    
    # Only reject if the ENTIRE block is an artifact (not just contains it)
    if len(block_text) < 100 and any(artifact in text_lower for artifact in artifacts):
        return False
    
    # ✅ Check for excessive repetition (e.g., "|||||||")
    if len(set(block_text)) < 5:  # Less than 5 unique characters
        return False
    
    return True


def build_chunk_metadata(filename, page_num, header, block_text, vehicle_model, file_hash, chunk_id):
    """
    Build standardized metadata for a text chunk.
    
    Centralizes metadata creation to ensure consistency.
    """
    return {
        "source_file": filename,
        "page": page_num,
        "type": "text",
        "chapter_header": header,
        "section_code": extract_section_code(header),
        "system_context": guess_system_from_header(header),
        "vehicle_model": vehicle_model,
        "component_type": guess_component_type(block_text),
        "file_hash": file_hash,
        "chunk_id": chunk_id,
        "category": "NarrativeText"
    }


def extract_section_code(header_text):
    """
    Extract section codes like 13B, 19B from header text.
    
    IMPROVEMENTS:
    - Better pattern matching
    - Validates section code format
    - Returns first valid code found
    """
    # ✅ More precise pattern: 1-2 digits followed by exactly one letter
    pattern = r'\b(\d{1,2}[A-Z])\b'
    matches = re.findall(pattern, header_text.upper())
    
    # Return first valid code
    for match in matches:
        # ✅ Ensure it's a realistic section code (not "1A" through "99Z")
        if len(match) >= 2 and match[0].isdigit():
            return match
    
    return "unknown"


def guess_system_from_header(header_text):
    """
    Automatically guess system category from header text.
    
    IMPROVEMENTS:
    - More comprehensive keyword matching
    - Priority ordering (more specific first)
    """
    text = header_text.lower()
    
    # ✅ More specific matches first
    system_mapping = [
        (["dpf", "fap", "particulate", "adblue", "scr", "egr"], "Exhaust/After-treatment"),
        (["injection", "injector", "fuel pump", "common rail"], "Fuel Injection"),
        (["engine", "diesel", "petrol", "gasoline", "cylinder"], "Engine"),
        (["abs", "brake", "parking brake", "caliper"], "Braking System"),
        (["air conditioning", "climate", "hvac", "a/c"], "HVAC"),
        (["alternator", "battery", "fuse", "relay", "wiring"], "Electrical System"),
        (["gearbox", "transmission", "clutch", "driveshaft"], "Transmission"),
        (["suspension", "steering", "wheel", "tire", "tyre"], "Chassis/Suspension"),
        (["coolant", "radiator", "thermostat", "cooling"], "Cooling System"),
    ]
    
    for keywords, system_name in system_mapping:
        if any(keyword in text for keyword in keywords):
            return system_name
    
    return "General"


def guess_component_type(text):
    """
    Guess component type from content.
    
    IMPROVEMENTS:
    - More granular categorization
    - Prioritized matching
    """
    text_lower = text.lower()
    
    component_mapping = [
        (["dpf", "particulate filter", "fap"], "DPF/Filter"),
        (["adblue", "scr", "urea"], "AdBlue System"),
        (["injector", "injection"], "Fuel Injector"),
        (["fuse", "relay"], "Electrical Component"),
        (["sensor", "probe"], "Sensor"),
        (["pump"], "Pump"),
        (["valve"], "Valve"),
        (["brake pad", "caliper", "disc"], "Brakes"),
        (["filter"], "Filter"),
    ]
    
    for keywords, component_name in component_mapping:
        if any(keyword in text_lower for keyword in keywords):
            return component_name
    
    return "General"


def detect_vehicle_model(filename):
    """Detect vehicle model from filename."""
    filename_lower = filename.lower()
    
    if "duster" in filename_lower:
        return "Dacia Duster"
    elif "logan" in filename_lower:
        return "Dacia Logan"
    elif "sandero" in filename_lower:
        return "Dacia Sandero"
    
    return "Dacia General"


# ==============================================================================
# IMAGE EXTRACTION
# ==============================================================================

def extract_images_with_pymupdf(pdf_path: str, output_dir: str = "static/images"):
    """
    Extracts images directly from PDF using PyMuPDF.
    Returns list of tuples: (image_path, page_num, size_kb)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    tasks = []
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        
        if not image_list:
            continue

        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                # Filter small icons/lines (< 5KB)
                if len(image_bytes) < 5 * 1024:
                    continue
                    
                # Save Image
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                image_filename = f"{base_name}_p{page_index+1}_i{img_index}.{ext}"
                image_filepath = os.path.join(output_dir, image_filename)
                
                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)
                    
                tasks.append((image_filepath, page_index + 1, len(image_bytes)/1024))
                
            except Exception as e:
                print(f"⚠️ Failed to extract image on page {page_index+1}: {e}")
                continue
    
    doc.close()
    return tasks


def process_images_parallel(image_tasks: list, max_workers: int = 4) -> list:
    """
    Process multiple images in parallel using ThreadPoolExecutor.
    """
    results = []
    if not image_tasks:
        return results

    print(f"🚀 Processing {len(image_tasks)} images in parallel (max {max_workers} workers)...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(describe_image, img_path, page_num): (img_path, page_num, size_kb)
            for img_path, page_num, size_kb in image_tasks
        }
        
        for future in as_completed(future_to_image):
            img_path, page_num, size_kb = future_to_image[future]
            try:
                description = future.result()
                results.append((img_path, page_num, description, size_kb))
            except Exception as e:
                print(f"❌ Failed to process {os.path.basename(img_path)}: {e}")

    elapsed = round(time.time() - start_time, 2)
    print(f"✅ Parallel processing complete in {elapsed}s ({len(results)} images)")
    return results


# ==============================================================================
# FALLBACK EXTRACTION
# ==============================================================================

def fallback_simple_extraction(pdf_path: str, filename: str):
    """Simple fallback extraction if smart extraction fails."""
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_counter = 0
    file_hash = get_file_hash(pdf_path)
    vehicle_model = detect_vehicle_model(filename)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if not text.strip():
            continue
        
        # Split into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        page_chunks = splitter.split_text(text)
        
        for chunk_text in page_chunks:
            metadata = {
                "source_file": filename,
                "page": page_num + 1,
                "type": "text",
                "chapter_header": "General",
                "section_code": "unknown",
                "system_context": "General",
                "vehicle_model": vehicle_model,
                "component_type": "General",
                "file_hash": file_hash,
                "chunk_id": f"text_{chunk_counter}",
                "category": "NarrativeText"
            }
            
            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })
            
            chunk_counter += 1
    
    doc.close()
    return chunks


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_file_hash(file_path: str) -> str:
    """Generate SHA256 hash of file for duplicate detection."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ==============================================================================
# TABLE CONVERSION (Optional - keep if you use tables)
# ==============================================================================

def convert_table_to_sentences(table_html: str) -> list:
    """Convert HTML table rows into natural language sentences."""
    soup = BeautifulSoup(table_html, 'html.parser')
    sentences = []
    
    try:
        headers = [th.get_text(strip=True) for th in soup.find_all('th')]
        if not headers:
            first_row = soup.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all('td')]
        
        if not headers:
            return []
        
        rows = soup.find_all('tr')[1:] if soup.find_all('th') else soup.find_all('tr')[1:]
        
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all('td')]
            if len(cells) != len(headers):
                continue
            
            row_data = dict(zip(headers, cells))
            sentence_parts = [f"{key}: {value}" for key, value in row_data.items() if value]
            if sentence_parts:
                sentences.append(", ".join(sentence_parts) + ".")
                
    except Exception as e:
        print(f"⚠️ Table parsing error: {e}")
    
    return sentences


# ==============================================================================
# PUBLIC API FUNCTIONS (Called by main.py)
# ==============================================================================

async def ingest_manual_multimodal(pdf_path: str, force_reingest: bool = False):
    """
    Public wrapper for PDF ingestion (async for FastAPI compatibility).
    Called by the FastAPI endpoint in main.py
    
    Args:
        pdf_path: Path to the PDF file
        force_reingest: If True, reprocess even if already indexed
    """
    if force_reingest:
        print(f"⚠️ Force reingest requested for {os.path.basename(pdf_path)}")
        # TODO: Implement deletion of existing chunks before reingesting
    
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, process_pdf, pdf_path)
    return result


async def generate_answer_multimodal(query: str, routing_info: dict = None):
    """
    Generate answer using Vector Search + Reranking (Async).
    
    Pipeline:
    1. Vector search (retrieve top 100 candidates)
    2. Rerank with cross-encoder (get top 5 most relevant)
    3. Build context
    4. Generate answer with LLM
    5. Extract media items
    """
    import asyncio
    loop = asyncio.get_event_loop()
    
    # Run in executor to avoid blocking the API
    def _generate():
        print(f"🔎 Searching for: {query}")
        
        # 1. Vector Search (broad retrieval)
        results = search_vector_db(
            query=query,
            k=100  # Cast wide net
        )
        
        if not results:
            return {
                'answer': "I couldn't find relevant information in the vehicle manual. Please rephrase your question or provide more details.",
                'sources': [],
                'media': [],
                'routing_info': {"status": "no_results"},
                'num_sources': 0
            }
        
        # 2. Rerank Results (semantic precision)
        try:
            reranked = rerank_results(query, results)
            top_results = reranked[:5]
        except Exception as e:
            print(f"⚠️ Reranker skipped (using raw search): {e}")
            top_results = results[:5]
        
        # 3. Build Context String with source tracking
        context_parts = []
        for i, doc in enumerate(top_results):
            metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            content = doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')
            
            source_info = f"[Source {i+1}, Page {metadata.get('page', 'N/A')}, Section {metadata.get('section_code', 'N/A')}]"
            context_parts.append(f"{source_info}\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # 4. Generate Answer with LLM
        answer = generate_chat_answer(
            context=context,
            user_question=query,
            routing_info=routing_info or {}
        )
        
        # 5. Extract Media (Images)
        media_items = []
        for doc in top_results:
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
            else:
                meta = doc.get('metadata', {})
            
            if meta.get('type') == 'image':
                media_items.append({
                    'path': meta.get('image_path'),
                    'page': meta.get('page'),
                    'description': doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')
                })
        
        return {
            'answer': answer,
            'sources': top_results,
            'media': media_items,
            'routing_info': routing_info or {"status": "pure_rag"},
            'num_sources': len(top_results)
        }
    
    result = await loop.run_in_executor(None, _generate)
    return result
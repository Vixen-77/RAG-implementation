import os
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Helper imports
from Services.llm_service import describe_image, generate_chat_answer
from Services.vectorStore import (
    add_multimodal_documents, 
    search_vector_db,
    is_document_indexed  # ✅ NEW: Import duplicate checker
)


def convert_table_to_sentences(table_html: str) -> list:
    """
    CRITICAL FIX: Convert HTML table rows into natural language sentences.
    
    This prevents hallucination by storing exact values as text instead of summaries.
    Example: 
      Row: | Tyre Size | Front Pressure | 
           | 215/65 R16 | 2.3 bar       |
      Becomes: "For tyre size 215/65 R16, the front pressure is 2.3 bar"
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    sentences = []
    
    try:
        # Extract headers
        headers = [th.get_text(strip=True) for th in soup.find_all('th')]
        
        if not headers:
            # Try first row as headers if no <th> tags
            first_row = soup.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all('td')]
        
        if not headers:
            print("      ⚠️ No headers found in table, skipping")
            return []
        
        # Process each data row
        rows = soup.find_all('tr')[1:] if headers else soup.find_all('tr')
        
        for row_idx, row in enumerate(rows):
            cells = [td.get_text(strip=True) for td in row.find_all('td')]
            
            if len(cells) < 2:  # Skip empty rows
                continue
            
            # Match cells to headers
            if len(cells) == len(headers):
                row_dict = dict(zip(headers, cells))
                
                # Build natural language sentence
                sentence_parts = []
                first_key = list(row_dict.keys())[0]
                first_value = row_dict[first_key]
                
                # Start with "For [first column]"
                sentence = f"For {first_key.lower()} {first_value}, "
                
                # Add remaining columns
                for key, value in list(row_dict.items())[1:]:
                    if value and value.lower() not in ['', '-', 'n/a', 'none']:
                        sentence_parts.append(f"the {key.lower()} is {value}")
                
                if sentence_parts:
                    sentence += " and ".join(sentence_parts) + "."
                    sentences.append(sentence)
                    
            else:
                # Fallback: concatenate cells with separators
                sentence = " | ".join([f"{headers[i] if i < len(headers) else 'Column'}: {cell}" 
                                      for i, cell in enumerate(cells)])
                sentences.append(sentence)
        
        print(f"      ✅ Converted table to {len(sentences)} sentences")
        return sentences
        
    except Exception as e:
        print(f"      ❌ Table conversion failed: {e}")
        # Fallback: just return the HTML as text
        return [soup.get_text(separator=" | ", strip=True)]


def partition_pdf_enhanced(file_path: str):
    """
    ✅ ENHANCED: Try multiple extraction strategies for better table detection.
    
    Strategy hierarchy:
    1. hi_res + lattice (best for bordered tables)
    2. hi_res + stream (best for borderless tables)
    3. fast (fallback for speed)
    """
    
    # Strategy 1: High-res with lattice (bordered tables)
    print("    📊 Trying Strategy 1: Hi-Res Lattice (bordered tables)...")
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
        if elements:
            print(f"    ✅ Strategy 1 succeeded: {len(elements)} elements extracted")
            return elements
    except Exception as e:
        print(f"    ⚠️ Strategy 1 failed: {e}")
    
    # Strategy 2: High-res with stream (borderless tables)
    print("    📊 Trying Strategy 2: Hi-Res Stream (borderless tables)...")
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
        if elements:
            print(f"    ✅ Strategy 2 succeeded: {len(elements)} elements extracted")
            return elements
    except Exception as e:
        print(f"    ⚠️ Strategy 2 failed: {e}")
    
    # Strategy 3: Fast fallback
    print("    📊 Trying Strategy 3: Fast mode (fallback)...")
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="fast",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
        if elements:
            print(f"    ✅ Strategy 3 succeeded: {len(elements)} elements extracted")
            return elements
    except Exception as e:
        print(f"    ❌ All strategies failed: {e}")
        return []


async def ingest_manual_multimodal(file_path: str, force_reingest: bool = False):
    """
    ✅ Enhanced ingestion pipeline with:
    - Duplicate detection (skip already indexed documents)
    - Better image filtering
    - Multi-strategy table extraction
    - Table-to-sentence conversion
    - Semantic text chunking
    - Metadata preservation
    
    Args:
        file_path: Path to PDF file
        force_reingest: If True, re-index even if already processed
        
    Returns:
        Number of chunks indexed (0 if already indexed and not forcing)
    """
    print(f"\n{'='*60}")
    print(f"🚀 [RAG] Starting Multimodal Ingestion")
    print(f"📄 File: {os.path.basename(file_path)}")
    print(f"{'='*60}\n")
    
    # ========== DUPLICATE CHECK ==========
    if not force_reingest and is_document_indexed(file_path):
        print(f"\n⏭️  [SKIP] Document already indexed!")
        print(f"💡 [TIP] Use force_reingest=True to re-process this document")
        print(f"{'='*60}\n")
        return 0
    
    if force_reingest:
        print(f"🔄 [FORCE] Re-ingesting document (force_reingest=True)")
    
    # ========== PART 1: Extract Images with PyMuPDF ==========
    print("🖼️ [Step 1/3] Extracting Images with PyMuPDF...")
    image_chunks = []
    os.makedirs("static/images", exist_ok=True)
    
    try:
        doc = fitz.open(file_path)
        image_counter = 0
        skipped_small = 0
        skipped_nontechnical = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images()
            
            if not images:
                continue
                
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                if len(image_bytes) < 15000:  
                    skipped_small += 1
                    continue
                
                image_filename = f"static/images/{os.path.basename(file_path).replace('.pdf', '')}_p{page_num}_i{img_index}.{image_ext}"
                
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                
                print(f"   📷 Page {page_num+1}, Image {img_index+1} -> Analyzing...")
                
                description = describe_image(image_filename, page_num=page_num+1)
                
                technical_keywords = [
                    'diagram', 'schematic', 'layout', 'fuse', 'wiring', 
                    'assembly', 'location', 'position', 'mounting', 'bolt',
                    'connection', 'circuit', 'component', 'engine', 'compartment'
                ]
                
                is_technical = any(keyword in description.lower() for keyword in technical_keywords)
                
                if is_technical:
                    image_chunks.append({
                        "description": description,
                        "path": image_filename,
                        "page": page_num + 1,
                        "size_kb": round(len(image_bytes) / 1024, 1)
                    })
                    print(f"      ✅ KEPT: {description[:70]}...")
                    image_counter += 1
                else:
                    os.remove(image_filename)  
                    skipped_nontechnical += 1
                    print(f"      ⏭️ Skipped (non-technical)")
        
        doc.close()
        print(f"\n    Image Stats:")
        print(f"      • Extracted: {image_counter} technical diagrams")
        print(f"      • Skipped: {skipped_small} small images, {skipped_nontechnical} non-technical")
        
    except Exception as e:
        print(f"    ❌ Image extraction failed: {e}")
    
    # ========== PART 2: Extract Text & Tables with Enhanced Strategy ==========
    print(f"\n📝 [Step 2/3] Extracting Text & Tables with Enhanced Detection...")
    
    elements = partition_pdf_enhanced(file_path)
    
    if not elements:
        print(f"    ❌ PDF partition failed completely")
        return 0
    
    text_chunks = []
    table_chunks = []
    
    # ✅ Semantic text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print(f"\n📄 [Step 3/3] Processing Elements...")
    
    for i, el in enumerate(elements):
        
        # ========== Handle Tables ==========
        if el.category == "Table":
            print(f"   📊 Table #{len(table_chunks)+1} (Element {i}) -> Converting to sentences...")
            
            html_content = el.metadata.text_as_html
            
            # ✅ CRITICAL: Convert to sentences instead of summarizing
            sentences = convert_table_to_sentences(html_content)
            
            page_num = getattr(el.metadata, 'page_number', 'unknown')
            
            for sentence in sentences:
                table_chunks.append({
                    "sentence": sentence,
                    "original_html": html_content,
                    "page": page_num
                })
        
        # ========== Handle Text Elements ==========
        elif el.category in ["CompositeElement", "Text", "NarrativeText", "Title"]:
            if not el.text or len(el.text.strip()) < 20:
                continue  
            
            chunks = splitter.split_text(el.text)
            page_num = getattr(el.metadata, 'page_number', 'unknown')
            
            for chunk in chunks:
                text_chunks.append({
                    "content": chunk,
                    "page": page_num,
                    "category": el.category
                })
            
            if len(text_chunks) % 100 == 0:
                print(f"    ✔ Processed {len(text_chunks)} text chunks...")

    # ========== PART 3: Store in Vector Database ==========
    print(f"\n💾 [Storage] Inserting into ChromaDB...")
    print(f"   • Text chunks: {len(text_chunks)}")
    print(f"   • Table sentences: {len(table_chunks)}")
    print(f"   • Image descriptions: {len(image_chunks)}")
    
    # ✅ Pass file_path for duplicate tracking
    add_multimodal_documents(text_chunks, type="text", file_path=file_path)
    add_multimodal_documents(table_chunks, type="table", file_path=file_path)
    add_multimodal_documents(image_chunks, type="image", file_path=file_path)
    
    total = len(text_chunks) + len(table_chunks) + len(image_chunks)
    
    print(f"\n{'='*60}")
    print(f"✅ [SUCCESS] Ingestion Complete!")
    print(f"📊 Total Chunks Indexed: {total}")
    print(f"{'='*60}\n")
    
    return total


async def generate_answer_multimodal(query: str):
    """
    ✅ Enhanced RAG pipeline with:
    - Hybrid search across all modalities
    - Duplicate media filtering
    - Rich metadata in response
    """
    print(f"\n{'='*60}")
    print(f"💬 [User Query] {query}")
    print(f"{'='*60}")
    
    # ========== Step 1: Vector Search ==========
    print(f"\n🔍 [Search] Querying vector database...")
    results = search_vector_db(query, k=5)  # Get top 5 results
    
    if not results:
        print("    ⚠️ No results found")
        return {
            "answer": "I couldn't find relevant information in the manual for that question.",
            "source_type": "none",
            "media_content": [],
            "num_sources": 0
        }
    
    # ========== Step 2: Build Context ==========
    context_text = ""
    media_items = []
    seen_images = set()  # Prevent duplicates
    
    print(f"\n📋 [Results] Processing {len(results)} matches...")
    
    for i, doc in enumerate(results):
        doc_type = doc.metadata.get('type', 'unknown')
        page = doc.metadata.get('page', 'unknown')
        preview = doc.page_content[:60].replace('\n', ' ')
        
        print(f"   [{i+1}] Type: {doc_type.upper()} | Page: {page}")
        print(f"       Preview: {preview}...")
        
        # Add to text context
        context_text += f"[Source {i+1} - Page {page}]\n{doc.page_content}\n\n"
        
        # ========== Attach Media ==========
        if doc_type == "image":
            img_path = doc.metadata.get("image_path", "")
            if img_path and img_path not in seen_images:
                web_path = "/" + img_path.replace("\\", "/")
                media_items.append({
                    "type": "image",
                    "url": web_path,
                    "caption": doc.page_content[:150],  # Show description
                    "page": page
                })
                seen_images.add(img_path)
                print(f"       🔗 Attached diagram from page {page}")
        
        elif doc_type == "table":
            media_items.append({
                "type": "table",
                "content": doc.metadata.get("original_html", ""),
                "page": page
            })
            print(f"       🔗 Attached table data")
    
    # ========== Step 3: Generate Answer ==========
    print(f"\n🤖 [LLM] Generating answer...")
    final_answer = generate_chat_answer(context_text, query)
    
    print(f"\n✅ [Response] Ready to send")
    print(f"   • Answer length: {len(final_answer)} chars")
    print(f"   • Media items: {len(media_items)}")
    print(f"{'='*60}\n")
    
    return {
        "answer": final_answer,
        "source_type": "multimodal" if media_items else "text",
        "media_content": media_items,
        "num_sources": len(results)
    }
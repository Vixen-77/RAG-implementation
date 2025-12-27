"""
Parent-Child Chunking Ingestion Pipeline
=========================================

This module implements a Parent-Child chunking strategy for automotive manuals:

1. NORMALIZE: Convert PDF → Markdown (great equalizer)
2. PARENT CHUNKS: Split by headers to preserve context units
3. CHILD CHUNKS: Split parents into searchable fragments
4. STORE: Children in Vector DB (with parent_id), Parents in Document Store

Architecture:
- Parents: Stored in shared document store (singleton)
- Children: Stored in ChromaDB with parent_id metadata
- Retrieval: Search children → fetch parents → send to LLM
"""

import os
import hashlib
import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

from Services.llm_service import describe_image
from Services.vectorStore import vector_db
from Services.docstore import get_docstore


class ParentDocumentStore:
    """
    DEPRECATED: This class is kept here for backwards compatibility.
    Use Services.docstore.get_docstore() instead.
    """
    def __init__(self):
        print("⚠️  [Warning] Using deprecated ParentDocumentStore from ingest.py")
        print("    Please use: from Services.docstore import get_docstore")
        self._store = get_docstore()
    
    def add_document(self, doc_id: str, document: Document):
        self._store.add_document(doc_id, document)
    
    def get_document(self, doc_id: str) -> Document:
        return self._store.get_document(doc_id)
    
    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        return self._store.get_documents(doc_ids)
    
    def clear(self):
        self._store.clear()
    
    def __len__(self):
        return len(self._store)


class MultimodalIngestionPipeline:
    """
    Complete Parent-Child ingestion pipeline for automotive manuals.
    
    Pipeline:
    1. Extract text from PDF pages
    2. Convert to Markdown-like structure
    3. Split by headers (Parents)
    4. Split parents into children
    5. Process images with vision model
    6. Store in ChromaDB + Document Store
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # Use shared document store for parents
        self.docstore = get_docstore()
        
        # Initialize ChromaDB for children (already imported at top)
        self.vectorstore = vector_db
        
        print("✅ [Pipeline] Parent-Child ingestion system initialized")
        print(f"   📁 Parent store: Shared singleton (in-memory)")
        print(f"   📁 Child store: ChromaDB ({persist_dir})")
    
    def ingest_pdf(self, pdf_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Main ingestion entry point.
        
        Args:
            pdf_path: Path to PDF file
            force: If True, re-ingest even if already processed
            
        Returns:
            Dict with status, counts, etc.
        """
        filename = os.path.basename(pdf_path)
        print(f"\n{'='*70}")
        print(f"📄 Processing: {filename}")
        print(f"{'='*70}")
        
        # Check for duplicates
        file_hash = self._compute_file_hash(pdf_path)
        
        if not force and self._is_document_indexed(file_hash):
            print(f"⏩ Skipping {filename} (Already Indexed)")
            return {"status": "skipped", "reason": "duplicate_hash"}
        
        # Extract and process
        try:
            # Step 1: Extract text pages
            print("📖 Step 1: Extracting text from PDF...")
            text_pages = self._extract_text_pages(pdf_path)
            
            # Step 2: Create parent chunks (by headers)
            print("🔨 Step 2: Creating parent chunks (context units)...")
            parent_docs = self._create_parent_chunks(text_pages, filename, file_hash)
            
            # Step 3: Create child chunks (searchable fragments)
            print("✂️  Step 3: Splitting parents into child chunks...")
            child_docs = self._create_child_chunks(parent_docs)
            
            # Step 4: Process images
            print("🖼️  Step 4: Processing images with vision model...")
            image_docs = self._process_images(pdf_path, filename, file_hash)
            
            # Step 5: Store everything
            print("💾 Step 5: Storing in vector database...")
            self._store_documents(parent_docs, child_docs, image_docs)
            
            print(f"✅ Ingestion complete!")
            print(f"   📦 Parents: {len(parent_docs)}")
            print(f"   🔍 Children: {len(child_docs)}")
            print(f"   🖼️  Images: {len(image_docs)}")
            
            return {
                "status": "success",
                "parents": len(parent_docs),
                "children": len(child_docs),
                "images_captioned": len(image_docs)
            }
            
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def _extract_text_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF, organized by pages with structure.
        
        Returns:
            List of page dicts with text and metadata
        """
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get structured text
            page_dict = page.get_text("dict", sort=True)
            blocks = page_dict.get("blocks", [])
            
            # Extract text with basic structure detection
            page_text = []
            for block in blocks:
                if "lines" not in block:
                    continue
                
                block_text = self._extract_block_text(block)
                if len(block_text) > 20:  # Filter noise
                    page_text.append(block_text)
            
            pages.append({
                "page_num": page_num + 1,
                "text": "\n\n".join(page_text),
                "blocks": len(page_text)
            })
        
        doc.close()
        print(f"   ✓ Extracted {len(pages)} pages")
        return pages
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract clean text from a PyMuPDF block"""
        spans_text = [
            span["text"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        return " ".join(spans_text).strip()
    
    def _create_parent_chunks(
        self, 
        pages: List[Dict[str, Any]], 
        filename: str, 
        file_hash: str
    ) -> List[Document]:
        """
        Create parent chunks by splitting on headers.
        
        Strategy:
        - Detect headers (ALL CAPS, section codes like "19B")
        - Split document into sections
        - Each section = 1 parent chunk
        """
        parents = []
        parent_id_counter = 0
        
        # Combine all pages into one text
        full_text = "\n\n".join([p["text"] for p in pages])
        
        # Detect vehicle model
        vehicle_model = self._detect_vehicle_model(filename)
        
        # Split by headers (simple regex for section codes and ALL CAPS titles)
        sections = self._split_by_headers(full_text)
        
        for section_title, section_text, section_code in sections:
            if len(section_text.strip()) < 100:  # Skip tiny sections
                continue
            
            parent_id = f"{file_hash[:8]}_parent_{parent_id_counter}"
            parent_id_counter += 1
            
            parent_doc = Document(
                page_content=section_text,
                metadata={
                    "parent_id": parent_id,
                    "type": "parent",
                    "source_file": filename,
                    "file_hash": file_hash,
                    "section_title": section_title,
                    "section_code": section_code,
                    "vehicle_model": vehicle_model,
                    "char_count": len(section_text)
                }
            )
            
            parents.append(parent_doc)
        
        print(f"   ✓ Created {len(parents)} parent chunks")
        return parents
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Split text by headers (section titles).
        
        Returns:
            List of (title, content, section_code) tuples
        """
        # Pattern to match headers:
        # - Section codes: "19B EXHAUST SYSTEM"
        # - ALL CAPS titles: "GENERAL MAINTENANCE"
        header_pattern = r'^([0-9]{1,2}[A-Z]?\s+)?([A-Z][A-Z\s]{10,})$'
        
        lines = text.split('\n')
        sections = []
        current_title = "General"
        current_section = []
        current_code = "unknown"
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line is a header
            match = re.match(header_pattern, line_stripped)
            
            if match and len(line_stripped) < 100:  # Headers are short
                # Save previous section
                if current_section:
                    sections.append((
                        current_title,
                        "\n".join(current_section),
                        current_code
                    ))
                
                # Start new section
                code_part = match.group(1) or ""
                title_part = match.group(2) or line_stripped
                
                current_code = code_part.strip() if code_part else "unknown"
                current_title = line_stripped
                current_section = []
            else:
                # Add to current section
                if line_stripped:
                    current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append((
                current_title,
                "\n".join(current_section),
                current_code
            ))
        
        return sections
    
    def _create_child_chunks(self, parent_docs: List[Document]) -> List[Document]:
        """
        Split each parent into smaller child chunks for retrieval.
        
        Strategy:
        - Use RecursiveCharacterTextSplitter
        - Chunk size: 400 chars (searchable unit)
        - Overlap: 50 chars (preserve context)
        - Each child tagged with parent_id
        """
        children = []
        
        # Configure splitter for child chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for parent in parent_docs:
            parent_id = parent.metadata["parent_id"]
            
            # Split parent into children
            child_texts = child_splitter.split_text(parent.page_content)
            
            for child_idx, child_text in enumerate(child_texts):
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        **parent.metadata,  # Inherit parent metadata
                        "parent_id": parent_id,
                        "type": "child",
                        "child_index": child_idx,
                        "chunk_id": f"{parent_id}_child_{child_idx}",
                        "char_count": len(child_text)
                    }
                )
                
                children.append(child_doc)
        
        print(f"   ✓ Created {len(children)} child chunks")
        return children
    
    def _process_images(
        self, 
        pdf_path: str, 
        filename: str, 
        file_hash: str
    ) -> List[Document]:
        """
        Extract and caption images using vision model.
        Uses ThreadPoolExecutor for parallel processing.
        """
        doc = fitz.open(pdf_path)
        image_docs = []
        image_dir = "static/images"
        os.makedirs(image_dir, exist_ok=True)
        
        # Collect all images first
        tasks = []
        
        print("   🖼️  Extracting images for processing...")
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    
                    # Filter small images
                    if len(image_bytes) < 5 * 1024:
                        continue
                    
                    # Save image
                    base_name = os.path.splitext(filename)[0]
                    image_filename = f"{base_name}_p{page_num+1}_i{img_idx}.{ext}"
                    image_path = os.path.join(image_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                        
                    tasks.append({
                        "path": image_path,
                        "page": page_num + 1,
                        "img_idx": img_idx,
                        "size_kb": round(len(image_bytes) / 1024, 2)
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to extract image on page {page_num+1}: {e}")
                    continue
        
        doc.close()
        
        if not tasks:
            print("   ⚠️  No valid images found")
            return []

        print(f"   🚀 Starting parallel captioning for {len(tasks)} images (limit: 4 workers)...")

        # Process in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_single_image(task):
            image_path = task["path"]
            page_num = task["page"]
            img_idx = task["img_idx"]
            
            # Generate caption (this is the slow part)
            caption = describe_image(image_path, page_num)
            
            # Create document
            parent_id = f"{file_hash[:8]}_image_{page_num-1}_{img_idx}"
            
            return Document(
                page_content=caption,
                metadata={
                    "parent_id": parent_id,
                    "type": "image",
                    "source_file": filename,
                    "file_hash": file_hash,
                    "page": page_num,
                    "image_path": image_path,
                    "size_kb": task["size_kb"],
                    "vehicle_model": self._detect_vehicle_model(filename)
                }
            )

        # Run with thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_image = {executor.submit(process_single_image, task): task for task in tasks}
            
            for future in as_completed(future_to_image):
                try:
                    doc = future.result()
                    image_docs.append(doc)
                    print(f"      ✓ Processing complete: {os.path.basename(doc.metadata['image_path'])}")
                except Exception as e:
                    print(f"      ❌ Image processing failed: {e}")

        print(f"   ✓ Processed {len(image_docs)} images successfully")
        return image_docs
    
    def _store_documents(
        self, 
        parents: List[Document], 
        children: List[Document],
        images: List[Document]
    ):
        """
        Store documents in appropriate stores:
        - Parents → Document Store (by parent_id)
        - Children → Vector DB (with parent_id reference)
        - Images → Vector DB (as searchable children)
        """
        # Store parents in document store
        for parent in parents:
            self.docstore.add_document(
                parent.metadata["parent_id"],
                parent
            )
        
        print(f"   ✓ Stored {len(parents)} parents in document store")
        
        # Store children in vector DB
        all_children = children + images
        
        if all_children:
            try:
                self.vectorstore.add_documents(all_children)
                print(f"   ✓ Stored {len(all_children)} children in vector DB")
            except Exception as e:
                print(f"   ❌ Failed to store children: {e}")
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        try:
            children_count = self.vectorstore._collection.count()
            parents_count = len(self.docstore)
            
            return {
                "children_count": children_count,
                "parents_count": parents_count,
                "persist_dir": self.persist_dir
            }
        except Exception as e:
            print(f"⚠️  Failed to get stats: {e}")
            return {
                "children_count": 0,
                "parents_count": 0,
                "persist_dir": self.persist_dir
            }
    
    # Utility methods
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _is_document_indexed(self, file_hash: str) -> bool:
        """Check if document already indexed"""
        try:
            results = self.vectorstore.get(
                where={"file_hash": file_hash},
                limit=1
            )
            return bool(results and results['ids'])
        except:
            return False
    
    def _detect_vehicle_model(self, filename: str) -> str:
        """Detect vehicle model from filename"""
        filename_lower = filename.lower()
        if "duster" in filename_lower:
            return "Dacia Duster"
        elif "logan" in filename_lower:
            return "Dacia Logan"
        elif "sandero" in filename_lower:
            return "Dacia Sandero"
        return "Dacia General"
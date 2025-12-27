
import os
import fitz
from typing import List
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.llm.client import describe_image
from .pdf_processor import detect_vehicle_model

def process_images(pdf_path: str, filename: str, file_hash: str) -> List[Document]:
    doc = fitz.open(pdf_path)
    image_docs = []
    image_dir = "static/images"
    os.makedirs(image_dir, exist_ok=True)
    
    tasks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                if len(image_bytes) < 5 * 1024:
                    continue
                
                base_name = os.path.splitext(filename)[0]
                image_filename = f"{base_name}_p{page_num+1}_i{img_idx}.{ext}"
                image_path = os.path.join(image_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                    
                tasks.append({
                    "path": image_path, "page": page_num + 1, "img_idx": img_idx,
                    "size_kb": round(len(image_bytes) / 1024, 2)
                })
            except Exception as e:
                print(f"[WARN] Failed to extract image on page {page_num+1}: {e}")
    
    doc.close()
    if not tasks:
        print("[INFO] No valid images found")
        return []

    print(f"[INFO] Captioning {len(tasks)} images...")

    def process_single(task):
        caption = describe_image(task["path"], task["page"])
        parent_id = f"{file_hash[:8]}_image_{task['page']-1}_{task['img_idx']}"
        return Document(
            page_content=caption,
            metadata={
                "parent_id": parent_id,
                "type": "image",
                "source_file": filename,
                "file_hash": file_hash,
                "page": task["page"],
                "image_path": task["path"],
                "vehicle_model": detect_vehicle_model(filename)
            }
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single, t) for t in tasks]
        for future in as_completed(futures):
            try:
                doc = future.result()
                image_docs.append(doc)
            except Exception as e:
                print(f"[ERROR] Image processing failed: {e}")

    print(f"[INFO] Processed {len(image_docs)} images")
    return image_docs

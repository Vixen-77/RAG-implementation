"""
PDF EXTRACTOR FOR CAR RAG SYSTEM
This module extracts text from car PDFs for the RAG application.

NOTE TO TEAM: The path to Data folder needs adjustment.
Currently looks for: ../../Data from Services folder.
Please check and fix the path if needed.
"""

import os
import sys
from typing import List, Dict, Optional
import json

# Try to import PyMuPDF, but don't fail if not installed
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyMuPDF not installed. Install with: pip install pymupdf")

class CarPDFExtractor:
    """
    Extracts text from car specification PDFs.
    
    ISSUE: The Data folder path might need adjustment.
    Current assumption: Data folder is at ../../Data from Services folder.
    """
    
    @staticmethod
    def get_data_directory() -> Optional[str]:
        """
        Find the Data directory containing PDFs.
        
        TODO: Team needs to verify/correct this path logic.
        The Data folder should contain car PDFs (dacia_duster_2019.pdf, etc.)
        """
        # Current file location (Services folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different possible locations
        possible_paths = [
            # From Services: FastAPI/App/Services/ -> ../../Data/
            os.path.join(current_dir, "..", "..", "Data"),
            # Alternative: Data folder in project root
            os.path.join(current_dir, "..", "..", "..", "Data"),
            # Just Data folder in current directory
            os.path.join(current_dir, "Data"),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isdir(abs_path):
                print(f"✅ Found Data folder: {abs_path}")
                return abs_path
        
        print("❌ Could not find Data folder automatically.")
        print("   Please check path configuration.")
        return None
    
    @staticmethod
    def extract_pdf(pdf_path: str) -> Optional[Dict]:
        """Extract text from a single PDF file."""
        if not PDF_SUPPORT:
            print("❌ PyMuPDF not installed. Run: pip install pymupdf")
            return None
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            return None
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text() + "\n\n"
            
            doc.close()
            
            return {
                "filename": os.path.basename(pdf_path),
                "content": text.strip(),
                "pages": len(doc),
                "characters": len(text)
            }
            
        except Exception as e:
            print(f"❌ Error extracting {pdf_path}: {e}")
            return None
    
    @staticmethod
    def load_car_documents() -> List[Dict]:
        """
        MAIN FUNCTION for RAG system.
        Loads all car PDFs and prepares them for vector store.
        
        TODO: Team needs to:
        1. Verify Data folder location
        2. Add PDF files to Data folder
        3. Install pymupdf: pip install pymupdf
        """
        print("=" * 60)
        print("🚗 LOADING CAR DOCUMENTS FOR RAG SYSTEM")
        print("=" * 60)
        
        # Find Data folder
        data_dir = CarPDFExtractor.get_data_directory()
        if not data_dir:
            return []
        
        # Find PDF files
        pdf_files = []
        for file in os.listdir(data_dir):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(data_dir, file))
        
        if not pdf_files:
            print(f"❌ No PDF files found in: {data_dir}")
            print("   Please add car PDF files to the Data folder")
            return []
        
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        # Extract all PDFs
        all_documents = []
        for pdf_file in pdf_files:
            print(f"\n🔍 Processing: {os.path.basename(pdf_file)}")
            
            extracted = CarPDFExtractor.extract_pdf(pdf_file)
            if extracted:
                # Split into chunks for RAG
                text = extracted["content"]
                words = text.split()
                
                # RAG chunking parameters
                chunk_size = 1000
                overlap = 200
                chunks = []
                
                i = 0
                while i < len(words):
                    chunk = ' '.join(words[i:i + chunk_size])
                    chunks.append(chunk)
                    i += chunk_size - overlap
                
                # Create RAG documents
                for i, chunk in enumerate(chunks):
                    rag_doc = {
                        "id": f"{extracted['filename']}_chunk_{i}",
                        "content": chunk,
                        "metadata": {
                            "source": extracted["filename"],
                            "pages": extracted["pages"],
                            "chunk": i,
                            "total_chunks": len(chunks)
                        }
                    }
                    all_documents.append(rag_doc)
                
                print(f"   ✅ Created {len(chunks)} chunks")
        
        print(f"\n✅ Total: {len(all_documents)} document chunks ready for RAG")
        print("=" * 60)
        
        # Save results for debugging
        if all_documents:
            with open("extraction_debug.json", "w") as f:
                json.dump({
                    "data_directory": data_dir,
                    "total_documents": len(all_documents),
                    "pdfs_processed": [os.path.basename(f) for f in pdf_files]
                }, f, indent=2)
            print("📊 Debug info saved to: extraction_debug.json")
        
        return all_documents


# ====== SIMPLE TEST FUNCTION ======
def test_pdf_extraction():
    """
    Test function - run this to verify PDF extraction works.
    Shows exactly what needs to be fixed.
    """
    print("\n🧪 PDF EXTRACTION TEST")
    print("=" * 60)
    
    # Check if pymupdf is installed
    if not PDF_SUPPORT:
        print("❌ PyMuPDF not installed")
        print("   Fix: pip install pymupdf")
        return False
    
    # Find Data folder
    data_dir = CarPDFExtractor.get_data_directory()
    if not data_dir:
        print("\n❌ ISSUE: Cannot find Data folder")
        print("   Please check folder structure:")
        print("   Expected: Data folder with PDFs at same level as FastAPI/")
        print("   Or adjust path in get_data_directory() method")
        return False
    
    # Check for PDFs
    pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdfs:
        print(f"\n❌ ISSUE: No PDF files in {data_dir}")
        print("   Please add car PDF files to the Data folder")
        return False
    
    print(f"\n✅ Found PDFs: {pdfs}")
    
    # Test extraction
    test_pdf = os.path.join(data_dir, pdfs[0])
    result = CarPDFExtractor.extract_pdf(test_pdf)
    
    if result:
        print(f"\n🎉 TEST PASSED!")
        print(f"   File: {result['filename']}")
        print(f"   Pages: {result['pages']}")
        print(f"   Characters: {result['characters']}")
        
        # Save sample
        with open("test_output.txt", "w") as f:
            f.write(result['content'][:1000])
        
        print(f"\n📄 Sample saved to: test_output.txt")
        return True
    else:
        print("\n❌ TEST FAILED")
        return False


# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    """
    When run directly, shows what needs to be fixed.
    """
    print("\n" + "=" * 60)
    print("CAR PDF EXTRACTOR - README FOR TEAM")
    print("=" * 60)
    print("\nTO FIX THIS MODULE, TEAM NEEDS TO:")
    print("1. ✅ Install: pip install pymupdf")
    print("2. ✅ Verify Data folder location with car PDFs")
    print("3. ✅ Update path in get_data_directory() if needed")
    print("4. ✅ Run test: python rag_services.py")
    print("=" * 60)
    
    # Run test
    success = test_pdf_extraction()
    
    if success:
        print("\n✅ Ready for RAG integration!")
        print("\nUsage in main app:")
        print("from rag_services import CarPDFExtractor")
        print("documents = CarPDFExtractor.load_car_documents()")
    else:
        print("\n❌ Needs fixing (see issues above)")
 
import os

print("🔍 CHECKING SETUP FROM SERVICES FOLDER")
print("=" * 60)

# Current location
print(f"1. I am at: {os.getcwd()}")

# Check Data folder path
data_relative = "../../Data"
data_absolute = os.path.abspath(data_relative)
print(f"\n2. Data folder should be at: {data_absolute}")
print(f"   Does it exist? {os.path.exists(data_absolute)}")

# Check if Data folder exists
if os.path.exists(data_absolute):
    print(f"\n3. ✅ Data folder FOUND!")
    print(f"   Location: {data_absolute}")
    
    # List files in Data
    print(f"\n4. Files in Data folder:")
    try:
        files = os.listdir(data_absolute)
        for file in files:
            print(f"   - {file}")
        
        # Check for PDFs
        pdfs = [f for f in files if f.lower().endswith('.pdf')]
        if pdfs:
            print(f"\n5. ✅ PDFs found: {pdfs}")
            
            # Test first PDF
            test_pdf = os.path.join(data_absolute, pdfs[0])
            print(f"\n6. Testing PDF: {test_pdf}")
            
            # Try to import fitz
            try:
                import fitz
                print("   ✅ PyMuPDF is installed")
                
                # Try to open PDF
                doc = fitz.open(test_pdf)
                text = doc[0].get_text()  # Just first page
                doc.close()
                
                print(f"   ✅ Can read PDF: Yes")
                print(f"   First page characters: {len(text)}")
                
                # Save sample
                with open("test_output.txt", "w", encoding="utf-8") as f:
                    f.write(text[:500])
                print(f"   📄 Sample saved to: test_output.txt")
                
            except ImportError:
                print("   ❌ PyMuPDF not installed")
                print("   Run: pip install pymupdf")
            except Exception as e:
                print(f"   ❌ Error reading PDF: {e}")
        else:
            print(f"\n5. ❌ No PDF files in Data folder")
            
    except Exception as e:
        print(f"   Error listing files: {e}")
else:
    print(f"\n3. ❌ Data folder NOT FOUND")
    print(f"   Please create it at: {data_absolute}")
    
    # Check project root
    print(f"\nChecking project root...")
    root = os.path.abspath("../../..")
    print(f"   Project root: {root}")
    print(f"   Files in root:")
    for item in os.listdir(root):
        print(f"     - {item}")

print("\n" + "=" * 60)
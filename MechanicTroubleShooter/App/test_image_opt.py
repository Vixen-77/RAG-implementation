import os
import time
from Services.llm_service import describe_image

def test_caching():
    print("\n--- Testing Image Caption Caching ---")
    
    # Use a dummy image path - the caching relies on file hash, so we need a real file
    # We'll use this script file itself as a dummy "image" just to test hashing/caching mechanism
    # (The LLM might be confused but the cache logic doesn't care content type for hashing)
    dummy_image = "test_image.png"
    
    # Create dummy image
    with open(dummy_image, "wb") as f:
        f.write(b"fake image content for testing caching")
        
    try:
        print("1. First Call (Should hit LLM)...")
        start = time.time()
        # We expect this might return an error string from LLM or a confused description
        # but the important part is it runs 
        res1 = describe_image(dummy_image) 
        t1 = time.time() - start
        print(f"Time: {t1:.2f}s")
        
        print("2. Second Call (Should hit Cache)...")
        start = time.time()
        res2 = describe_image(dummy_image)
        t2 = time.time() - start
        print(f"Time: {t2:.2f}s")
        
        if t2 < 1.0 and res1 == res2:
            print("✅ Caching PASSED (Second call was instant and identical)")
        else:
            print("❌ Caching FAILED")
            print(f"T1: {t1}, T2: {t2}")
            
    finally:
        if os.path.exists(dummy_image):
            os.remove(dummy_image)

if __name__ == "__main__":
    test_caching()

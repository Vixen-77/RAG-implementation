from pdf2image import convert_from_path

try:
    import pytesseract
except ImportError:
    raise ImportError("pytesseract is not installed. Install it using: pip install pytesseract")

images = convert_from_path("clio.pdf", dpi=300)
text = "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images)

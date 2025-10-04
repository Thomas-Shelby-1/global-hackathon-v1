from pathlib import Path
from typing import List

def load_pdf_text(path: Path) -> List[str]:
    """Return list of page texts; uses PyMuPDF if available, else pypdf."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = [page.get_text() or "" for page in doc]
        doc.close()
        return pages
    except Exception:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return pages

from pathlib import Path
import fitz  # PyMuPDF
from backend.rag.embed import embed_texts

doc = fitz.open("sample.pdf")
texts=[]
for page in doc:
    texts.append(page.get_text("text"))
# TODO: split to chunks first; this is just a smoke test
vecs = embed_texts(texts, dim=int(os.getenv("EMBEDDING_DIM","512")))
print("Embedded", len(vecs), "chunks")

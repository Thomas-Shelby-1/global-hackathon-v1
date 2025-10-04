from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import orjson, json

from backend.config import DATA_DIR, GRAPH_JSON
from backend.loaders.pdf import load_pdf_text
from backend.preprocess.chunk import chunk_text
from backend.extract.triples_llm import extract_triples_many
from backend.graph.build import GraphStore
from backend.rag.embed import embed_texts
from backend.rag.index_hnsw import SKIndex

app = FastAPI(title="hackon")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class HealthResp(BaseModel):
    ok: bool

class IngestResp(BaseModel):
    ok: bool
    stored_as: str
    chunks: int

class BuildResp(BaseModel):
    ok: bool
    nodes: int
    edges: int

@app.get("/health", response_model=HealthResp)
def health():
    return {"ok": True}

@app.post("/ingest/pdf", response_model=IngestResp)
async def ingest_pdf(file: UploadFile = File(...)):
    out_path = DATA_DIR / "ingested" / file.filename
    with out_path.open("wb") as f:
        f.write(await file.read())

    # 1) pdf -> raw text
    pages = load_pdf_text(out_path)

    # 2) chunk
    chunks = chunk_text("\n".join(pages), max_chars=1200, overlap=150)
    # store chunks for transparency
    chunks_dir = DATA_DIR / "chunks"
    (chunks_dir / (out_path.stem + ".json")).write_text(orjson.dumps(chunks).decode(), encoding="utf-8")

    return {"ok": True, "stored_as": str(out_path), "chunks": len(chunks)}

@app.post("/extract/build", response_model=BuildResp)
def extract_and_build():
    # load last chunks file (simple: pick latest .json in chunks/)
    chunks_dir = DATA_DIR / "chunks"
    latest = max(chunks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    chunks = json.loads(latest.read_text(encoding="utf-8"))

    # 3) LLM extraction → nodes/edges (typed)
    triples = extract_triples_many(chunks)

    # 4) build/merge graph
    gs = GraphStore.from_path(GRAPH_JSON)
    gs.add_triples(triples)
    gs.save(GRAPH_JSON)

    # 5) embed chunks → hnsw index
    vecs = embed_texts([c["text"] for c in chunks])
    index = SKIndex(metric="cosine")
    index.add(vecs)
    index.save(str(DATA_DIR / "index" / "chunks_sklearn.joblib"))

    return {"ok": True, "nodes": len(gs.nodes), "edges": len(gs.edges)}

@app.get("/graph")
def get_graph():
    if GRAPH_JSON.exists():
        return orjson.loads(GRAPH_JSON.read_bytes())
    return {"nodes": [], "edges": []}

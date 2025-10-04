import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(DATA_DIR / "index"))).resolve()
GRAPH_JSON = Path(os.getenv("GRAPH_JSON", str(DATA_DIR / "graph.json"))).resolve()

for p in [DATA_DIR, DATA_DIR / "ingested", DATA_DIR / "chunks", INDEX_DIR]:
    p.mkdir(parents=True, exist_ok=True)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "ollama")

GEN_MODEL    = os.getenv("GEN_MODEL", "qwen2.5:7b-instruct")
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "llama3.1:8b-instruct")
PLANNER_MODEL= os.getenv("PLANNER_MODEL", "deepseek-r1:8b")

EMBED_MODEL  = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

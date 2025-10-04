import json
from typing import List, Dict
from openai import OpenAI
import os
from backend.config import OPENAI_BASE_URL, OPENAI_API_KEY, GEN_MODEL

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

PROMPT = """You are an information extraction model.
From the text, output JSON with:
- "concepts": [{name, aliases, definition?}]
- "relations": [{src, type, dst, evidence_span}]
Allowed relation types: prerequisite_of, part_of, example_of, explains, contrasts_with, equivalent_to, cites, depends_on.
Return strictly valid JSON.
TEXT:
<<<{chunk}>>>"""

def _call_llm(text: str) -> Dict:
    msg = PROMPT.format(chunk=text)
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role":"user","content": msg}],
        temperature=0.2,
    ).choices[0].message.content
    try:
        return json.loads(resp)
    except Exception:
        # be defensive; try to extract JSON blob
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(resp[start:end+1])
        return {"concepts": [], "relations": []}

def extract_triples_many(chunks: List[Dict]) -> List[Dict]:
    out = []
    for i, ch in enumerate(chunks):
        data = _call_llm(ch["text"])
        # normalize
        concepts = data.get("concepts", [])
        rels = data.get("relations", [])
        out.append({"chunk_id": i, "concepts": concepts, "relations": rels})
    return out

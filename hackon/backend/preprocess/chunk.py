import re
from typing import List, Dict

def _split_sentences(text: str) -> List[str]:
    # quick sentence-ish split; good enough to start
    return re.split(r'(?<=[.!?])\s+', text.strip())

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[Dict]:
    sents = _split_sentences(text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                chunks.append({"text": buf})
                # overlap: keep tail of previous buf
                tail = buf[-overlap:] if overlap>0 else ""
                buf = f"{tail} {s}".strip()
            else:
                chunks.append({"text": s[:max_chars]})
                buf = s[max_chars-overlap:max_chars]
    if buf:
        chunks.append({"text": buf})
    return chunks

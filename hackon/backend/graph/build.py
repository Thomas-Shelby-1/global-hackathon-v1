import json
from pathlib import Path
from typing import Dict, List
from rapidfuzz import fuzz, process

class GraphStore:
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # key=name
        self.edges: List[Dict] = []

    @staticmethod
    def from_path(path: Path) -> "GraphStore":
        gs = GraphStore()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for n in data.get("nodes", []):
                gs.nodes[n["name"]] = n
            gs.edges = data.get("edges", [])
        return gs

    def _canonical(self, name: str) -> str:
        name = name.strip()
        if name in self.nodes:
            return name
        # fuzzy match to existing names
        if self.nodes:
            match = process.extractOne(name, list(self.nodes.keys()), scorer=fuzz.WRatio)
            if match and match[1] >= 90:
                return match[0]
        return name

    def add_triples(self, triples: List[Dict]):
        for item in triples:
            for c in item.get("concepts", []):
                nm = self._canonical(c.get("name","").strip())
                if not nm: 
                    continue
                node = self.nodes.get(nm, {"id": f"concept:{nm}", "name": nm, "aliases": [], "definition": None, "sources": []})
                if c.get("aliases"):
                    node["aliases"] = sorted(set(node.get("aliases", []) + c["aliases"]))
                if c.get("definition") and not node.get("definition"):
                    node["definition"] = c["definition"]
                self.nodes[nm] = node

            for r in item.get("relations", []):
                src = self._canonical(r.get("src","").strip())
                dst = self._canonical(r.get("dst","").strip())
                typ = r.get("type","").strip() or "related_to"
                if not src or not dst or src == dst: 
                    continue
                self.edges.append({
                    "src": f"concept:{src}",
                    "dst": f"concept:{dst}",
                    "type": typ,
                    "confidence": 0.5,
                    "evidence": [{"span": r.get("evidence_span","")}]
                })

    def save(self, path: Path):
        data = {
            "nodes": [v for v in self.nodes.values()],
            "edges": self.edges,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

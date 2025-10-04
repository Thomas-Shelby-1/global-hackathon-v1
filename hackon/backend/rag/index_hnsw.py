import hnswlib, numpy as np
from pathlib import Path

class HNSWIndex:
    def __init__(self, dim:int, space:str="cosine"):
        self.index = hnswlib.Index(space=space, dim=dim)
        self._initialized = False
        self._count = 0

    def add(self, vectors: np.ndarray, ids=None, ef_construction=200, M=16):
        if not self._initialized:
            self.index.init_index(max_elements=len(vectors), ef_construction=ef_construction, M=M)
            self._initialized = True
        if ids is None:
            ids = list(range(self._count, self._count + len(vectors)))
        self.index.add_items(vectors, ids)
        self._count += len(vectors)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(path))

    def load(self, path: str):
        self.index.load_index(str(path))
        self._initialized = True

    def search(self, vectors: np.ndarray, k=5):
        labels, dists = self.index.knn_query(vectors, k=k)
        return labels, dists

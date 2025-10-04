# backend/rag/index_sklearn.py
from pathlib import Path
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SKIndex:
    def __init__(self, metric="cosine"):
        self.metric = metric
        self.nn = None
        self._count = 0

    def add(self, vectors: np.ndarray, ids=None):
        self.nn = NearestNeighbors(metric=self.metric, algorithm="auto")
        self.nn.fit(vectors)
        self._count = vectors.shape[0]

    def search(self, vectors: np.ndarray, k=5):
        dists, idx = self.nn.kneighbors(vectors, n_neighbors=min(k, self._count), return_distance=True)
        return idx, dists

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.nn, path)

    def load(self, path: str):
        self.nn = joblib.load(path)
        self._count = getattr(self.nn._fit_X, "shape", [0])[0]

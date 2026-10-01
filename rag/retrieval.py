"""Step 4: hybrid search. dense + BM25 -> RRF fuse -> cross-encoder rerank."""
import pickle

from . import config
from .schemas import Chunk
from .utils import tokenize


def rrf_fuse(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. k=60 is the default from the RRF paper."""
    scores: dict[str, float] = {}
    for ranking in (dense_ids, sparse_ids):
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class HybridRetriever:
    """Loads indices + models once. Reuse one instance for many queries."""

    def __init__(self):
        if not config.BM25_PATH.exists():
            raise RuntimeError("Index not found. Run first:  python main.py build")
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        try:
            self.collection = client.get_collection(config.CHROMA_COLLECTION)
        except Exception as e:
            raise RuntimeError("Chroma collection missing. Run:  python main.py build") from e

        with open(config.BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids: list[str] = data["chunk_ids"]
        self.lookup: dict[str, Chunk] = data["lookup"]
        self._embed = None
        self._rerank = None

    # ---- lazy model loading ----
    def _embed_model(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer(config.EMBED_MODEL)
        return self._embed

    def _rerank_model(self):
        if self._rerank is None:
            from sentence_transformers import CrossEncoder
            self._rerank = CrossEncoder(config.RERANK_MODEL)
        return self._rerank

    # ---- stages ----
    def dense_search(self, query: str, top_k: int = config.DENSE_TOP_K) -> list[str]:
        q = self._embed_model().encode([query], normalize_embeddings=True)
        n = min(top_k, self.collection.count())
        res = self.collection.query(query_embeddings=q.tolist(), n_results=n)
        return res["ids"][0]

    def sparse_search(self, query: str, top_k: int = config.SPARSE_TOP_K) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, s in ranked[:top_k] if s > 0]

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = config.RERANK_TOP_K) -> list[dict]:
        if not candidates:
            return []
        scores = self._rerank_model().predict([(query, c["text"]) for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

    def retrieve(self, query: str, verbose: bool = True) -> list[dict]:
        dense_ids = self.dense_search(query)
        sparse_ids = self.sparse_search(query)
        fused = rrf_fuse(dense_ids, sparse_ids)[:max(config.DENSE_TOP_K, config.SPARSE_TOP_K)]

        candidates = []
        for cid in fused:
            ch = self.lookup.get(cid)
            if ch:
                candidates.append({
                    "chunk_id": ch.chunk_id,
                    "text": ch.text,
                    "source": ch.source,
                    "doc_type": ch.doc_type,
                })
        top = self.rerank(query, candidates)
        if verbose:
            print(f"[retrieve] dense={len(dense_ids)} sparse={len(sparse_ids)} "
                  f"fused={len(fused)} final={len(top)}")
        return top

"""Step 3: build dense (Chroma) + sparse (BM25) indices."""
import pickle

from rank_bm25 import BM25Okapi

from . import config
from .schemas import Chunk
from .utils import tokenize

BATCH = 256


def _clean_meta(chunk: Chunk) -> dict:
    """Chroma metadata only allows str/int/float/bool."""
    meta = {"source": chunk.source, "doc_type": chunk.doc_type}
    for k, v in chunk.metadata.items():
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
    return meta


def build_dense_index(chunks: list[Chunk]):
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # fresh build: drop old collection so stale chunks never linger
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [c.text for c in batch]
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=emb.tolist(),
            documents=texts,
            metadatas=[_clean_meta(c) for c in batch],
        )
        print(f"[index] dense {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"[index] dense: {collection.count()} vectors in Chroma")
    return collection


def build_sparse_index(chunks: list[Chunk]):
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c.chunk_id for c in chunks],
            "lookup": {c.chunk_id: c for c in chunks},
        }, f)
    print(f"[index] sparse: BM25 over {len(chunks)} chunks -> {config.BM25_PATH.name}")
    return bm25


def build_all_indices(chunks: list[Chunk]):
    if not chunks:
        raise ValueError("No chunks to index. Put files in data/docs, data/pdfs, data/code "
                         "or pass --repo.")
    build_dense_index(chunks)
    build_sparse_index(chunks)

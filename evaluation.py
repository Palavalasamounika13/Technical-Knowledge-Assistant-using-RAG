"""Step 6: auto-generate QA pairs from chunks, measure retrieval hit-rate + MRR."""
import json
import pickle
import random
import time

from pydantic import BaseModel

from . import config
from .llm import generate_json
from .retrieval import HybridRetriever
from .schemas import Chunk, EvalPair

EVAL_GEN_PROMPT = """You are creating a test dataset for a RAG system.
Use ONLY the information in the context below - don't add outside facts.
Generate {n} question-answer pairs a real user might ask about this content.
Questions must be answerable from this context alone and must be specific
(do not write questions like "what does this text say").

Context:
{context}"""


class _QA(BaseModel):
    question: str
    answer: str


class _QASet(BaseModel):
    pairs: list[_QA]


def load_chunks() -> list[Chunk]:
    with open(config.BM25_PATH, "rb") as f:
        return list(pickle.load(f)["lookup"].values())


def generate_eval_set(chunks: list[Chunk], n_per_chunk: int = 1,
                      sample_size: int = 20, min_chars: int = 200,
                      pause: float = 1.0) -> list[EvalPair]:
    """Sample chunks, ask Gemini for QA grounded in each. source_chunk_id = gold label."""
    pool = [c for c in chunks if len(c.text) >= min_chars]
    sample = random.sample(pool, min(sample_size, len(pool)))
    pairs: list[EvalPair] = []

    for i, chunk in enumerate(sample, 1):
        try:
            raw = generate_json(
                EVAL_GEN_PROMPT.format(n=n_per_chunk, context=chunk.text), _QASet)
            for qa in _QASet.model_validate_json(raw).pairs:
                pairs.append(EvalPair(question=qa.question, answer=qa.answer,
                                      source_chunk_id=chunk.chunk_id))
        except Exception as e:
            print(f"[eval] skipped chunk {chunk.chunk_id}: {e}")
        print(f"[eval] generating {i}/{len(sample)}", end="\r")
        time.sleep(pause)  # be kind to rate limits

    print(f"\n[eval] {len(pairs)} QA pairs from {len(sample)} chunks")
    return pairs


def save_eval_set(pairs: list[EvalPair], path=config.EVAL_SET_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in pairs], f, indent=2)
    print(f"[eval] saved -> {path}")


def load_eval_set(path=config.EVAL_SET_PATH) -> list[EvalPair]:
    with open(path, encoding="utf-8") as f:
        return [EvalPair(**d) for d in json.load(f)]


def run_retrieval_eval(pairs: list[EvalPair], retriever: HybridRetriever,
                       k: int = config.RERANK_TOP_K) -> dict:
    """hit-rate@k: gold chunk in top-k. MRR: mean of 1/rank of gold chunk."""
    hits, rr = 0, 0.0
    for p in pairs:
        ids = [c["chunk_id"] for c in retriever.retrieve(p.question, verbose=False)[:k]]
        if p.source_chunk_id in ids:
            hits += 1
            rr += 1.0 / (ids.index(p.source_chunk_id) + 1)
    n = len(pairs) or 1
    out = {"hit_rate": hits / n, "mrr": rr / n, "n": len(pairs)}
    print(f"[eval] hit-rate@{k}: {out['hit_rate']:.2%} ({hits}/{len(pairs)})  MRR: {out['mrr']:.3f}")
    return out

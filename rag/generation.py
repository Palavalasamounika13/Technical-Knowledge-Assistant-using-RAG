"""Step 5: LLM answer with Pydantic schema, citation validation + retry.

The provider (gemini / ollama / xai) lives in rag/llm.py. Set LLM_PROVIDER in .env.
"""
from pydantic import BaseModel, ValidationError

from .llm import generate_json
from .schemas import RAGAnswer

SYSTEM_PROMPT = """You are a technical Q&A assistant. Answer ONLY using the
provided context chunks. Every factual claim in your answer must cite the
chunk_id it came from. If the context doesn't contain the answer, say so
plainly and set confidence to "low" - never invent information.
Use only chunk_id values that appear in the context below.
Copy each chunk_id and its source EXACTLY as written in the context."""


def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[chunk_id: {c['chunk_id']}] (source: {c['source']})\n{c['text']}" for c in chunks
    )


def _parse(prompt: str, schema: type[BaseModel], tries: int = 2):
    """Call LLM, validate JSON against schema. Retry if model returned broken JSON."""
    last = None
    for _ in range(tries):
        try:
            return schema.model_validate_json(generate_json(prompt, schema))
        except ValidationError as e:
            last = e
    raise RuntimeError(f"Model returned invalid JSON {tries} times: {last}")


def _call_llm(query: str, context: str) -> RAGAnswer:
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
    return _parse(prompt, RAGAnswer)


def validate_citations(result: RAGAnswer, valid_ids: set[str]) -> tuple[bool, list[str]]:
    bad = [c.chunk_id for c in result.citations if c.chunk_id not in valid_ids]
    return (not bad, bad)


def generate_answer(query: str, retrieved: list[dict], max_retries: int = 1) -> RAGAnswer:
    if not retrieved:
        return RAGAnswer(answer="No relevant context found in the index.",
                         citations=[], confidence="low")

    context = build_context(retrieved)
    valid_ids = {c["chunk_id"] for c in retrieved}

    result = _call_llm(query, context)
    ok, bad = validate_citations(result, valid_ids)

    attempt = 0
    while not ok and attempt < max_retries:
        fix = (f"\n\nYour previous answer cited chunk_id(s) not present in context: {bad}. "
               f"Redo using ONLY valid chunk_ids listed above.")
        result = _call_llm(query, context + fix)
        ok, bad = validate_citations(result, valid_ids)
        attempt += 1

    if not ok:  # still bad: drop fake citations, flag low confidence
        result.citations = [c for c in result.citations if c.chunk_id in valid_ids]
        result.confidence = "low"
    return result

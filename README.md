# Technical Knowledge Assistant using RAG

Hybrid retrieval: dense (Chroma + BGE) + sparse (BM25) -> RRF fusion -> cross-encoder rerank ->
Gemini answer with Pydantic-enforced, validated citations. Includes auto-generated retrieval eval.

```
data/docs/   .md .txt .html          data/pdfs/  PDFs (text-based)
data/code/   local source code       --repo URL  any git repo (shallow clone)
                    |
   ingest -> chunk (AST for .py) -> index (Chroma + BM25) -> storage/
                    |
   query -> dense + BM25 -> RRF -> rerank -> Gemini -> answer + citations
```

## Setup in VS Code

1. Install **Python 3.10+** and VS Code with the **Python** extension.
2. `File > Open Folder...` -> this folder.
3. Terminal (`` Ctrl+` ``):

   ```bash
   python -m venv .venv
   # Windows PowerShell:  .venv\Scripts\Activate.ps1
   # macOS / Linux:       source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. `Ctrl+Shift+P` -> **Python: Select Interpreter** -> pick `.venv`.
5. Copy `.env.example` to `.env`, paste your key from https://aistudio.google.com/apikey:

   ```
   GEMINI_API_KEY=your_key_here
   ```

## Run

Sample doc already in `data/docs/`. Drop your own files in `data/...` first if you like.

```bash
python main.py build                                   # index data/docs, data/pdfs, data/code
python main.py build --repo https://github.com/psf/requests   # also index a git repo
python main.py ask "How does the retry logic work?"
python main.py chat                                    # interactive
python main.py eval --n 20                             # hit-rate@5 + MRR
streamlit run app.py                                   # web UI
```

Or press **F5** and pick a config from `.vscode/launch.json`.

First `build` downloads BGE models (~1.3 GB embed + ~2.2 GB reranker). One time only.
Re-run `build` whenever sources change (it rebuilds both indices from scratch).

## Layout

```
main.py            CLI (build / ask / chat / eval)
app.py             Streamlit UI
rag/config.py      models, chunk size, top-k, paths  <- tune here
rag/ingest.py      repo / code / docs / PDF loaders
rag/chunking.py    AST chunking for Python, char splitter for rest
rag/indexing.py    Chroma + BM25 builders
rag/retrieval.py   HybridRetriever (dense+sparse+RRF+rerank)
rag/generation.py  Gemini call, citation validation + retry
rag/evaluation.py  eval-set generation, hit-rate, MRR
```

## Tuning

- Retrieval bad? Run `eval`, change `CHUNK_SIZE`, `*_TOP_K` or models in `rag/config.py`,
  `build` again, `eval` again. Compare hit-rate. Use `--regen` only if chunking changed
  (chunk IDs change then, old eval set becomes stale).
- Smaller/faster models: set `EMBED_MODEL=BAAI/bge-small-en-v1.5` in `.env`
  and use a smaller reranker such as `BAAI/bge-reranker-base`.
- GPU: sentence-transformers uses CUDA automatically if PyTorch CUDA is installed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY missing` | create `.env` from `.env.example` |
| `Index not found` | run `python main.py build` first |
| 429 errors on eval | free-tier rate limit; lower `--n` or raise `pause` in `evaluation.py` |
| PDF skipped: no text layer | scanned PDF; OCR it first (e.g. `ocrmypdf`) |
| `ModuleNotFoundError: rag` | run commands from project root |

## Notes vs. original notebook

- PDF parsing uses `pypdf` (light) instead of `unstructured[pdf]` (needs poppler/tesseract).
- Fixed infinite-loop risk in overlap splitter tail; fixed byte/char slicing bug for non-ASCII code.
- Big Python functions split instead of truncated; file path prepended to code chunks.
- BM25 tokenizer splits snake_case/camelCase; indices and models load once per session.
- Chroma upserts batched, cosine space set, collection reset on rebuild (no stale chunks).
- Added `data/code`, MRR metric, Streamlit UI, Gemini 429/5xx backoff.

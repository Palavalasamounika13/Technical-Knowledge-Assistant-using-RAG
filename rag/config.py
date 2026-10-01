"""Central config. Override via .env where noted."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- LLM provider: "gemini" (default), "groq", "xai" (Grok) or "ollama" (local) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.3")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Groq (groq.com), keys start with gsk_
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # None -> http://localhost:11434

# --- Embedding / rerank models (run locally) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Chunking (characters, not tokens) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CODE_CHUNK = 3000  # code node bigger than this gets split further

# --- Retrieval ---
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RERANK_TOP_K = 5

# --- Storage ---
STORAGE_DIR = ROOT / "storage"
CHROMA_DIR = str(STORAGE_DIR / "chroma_db")
CHROMA_COLLECTION = "tech_qa_bot"
BM25_PATH = STORAGE_DIR / "bm25_index.pkl"
REPO_CACHE_DIR = STORAGE_DIR / "repo_cache"
EVAL_SET_PATH = ROOT / "eval_set.json"

# --- Default source folders ---
DEFAULT_DOCS_DIR = ROOT / "data" / "docs"
DEFAULT_PDF_DIR = ROOT / "data" / "pdfs"
DEFAULT_CODE_DIR = ROOT / "data" / "code"

CODE_EXTENSIONS = (".py", ".js", ".ts", ".go", ".java")
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    "test", "tests", "__tests__",
}

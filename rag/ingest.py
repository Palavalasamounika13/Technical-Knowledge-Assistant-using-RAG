"""Step 1: pull raw text from git repos, local code, docs folders, PDFs."""
from pathlib import Path

from bs4 import BeautifulSoup

from . import config
from .schemas import RawDoc


def _skip(rel_path: Path) -> bool:
    return any(part in config.SKIP_DIRS for part in rel_path.parts)


def _read(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return text if text.strip() else None


def _collect_code(root: Path, source_fn) -> list[RawDoc]:
    docs = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in config.CODE_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        if _skip(rel):
            continue
        text = _read(path)
        if text is None:
            continue
        docs.append(RawDoc(
            text=text,
            source=source_fn(rel),
            doc_type="code",
            metadata={"language": path.suffix.lstrip(".")},
        ))
    return docs


def ingest_repo(repo_url: str) -> list[RawDoc]:
    """Shallow-clone (or reuse) a git repo and pull source files."""
    import git  # lazy: only needed for repos

    repo_url = repo_url.rstrip("/")
    name = repo_url.split("/")[-1].removesuffix(".git")
    local = config.REPO_CACHE_DIR / name
    if not local.exists():
        config.REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[ingest] cloning {repo_url} ...")
        git.Repo.clone_from(repo_url, str(local), depth=1)

    base = repo_url.removesuffix(".git")
    return _collect_code(local, lambda rel: f"{base}/blob/HEAD/{rel.as_posix()}")


def ingest_code_folder(folder: str | Path) -> list[RawDoc]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return _collect_code(folder, lambda rel: (folder / rel).as_posix())


def ingest_docs_folder(folder: str | Path) -> list[RawDoc]:
    """Markdown / text / HTML docs."""
    folder = Path(folder)
    docs = []
    if not folder.exists():
        return docs
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt", ".rst", ".html", ".htm"):
            continue
        text = _read(path)
        if text is None:
            continue
        if path.suffix.lower() in (".html", ".htm"):
            text = BeautifulSoup(text, "html.parser").get_text("\n")
        docs.append(RawDoc(text=text, source=path.as_posix(), doc_type="doc"))
    return docs


def ingest_pdfs(folder: str | Path) -> list[RawDoc]:
    from pypdf import PdfReader  # lazy

    folder = Path(folder)
    docs = []
    if not folder.exists():
        return docs
    for path in folder.rglob("*.pdf"):
        try:
            reader = PdfReader(str(path))
            pages = [(p.extract_text() or "") for p in reader.pages]
        except Exception as e:  # corrupt / encrypted pdf
            print(f"[ingest] skipped {path.name}: {e}")
            continue
        text = "\n\n".join(pages)
        if not text.strip():
            print(f"[ingest] {path.name}: no text layer (scanned PDF?), skipped")
            continue
        docs.append(RawDoc(text=text, source=path.as_posix(), doc_type="pdf"))
    return docs


def ingest_all(repo_url: str | None = None,
               docs_folder: str | Path | None = None,
               pdf_folder: str | Path | None = None,
               code_folder: str | Path | None = None) -> list[RawDoc]:
    all_docs: list[RawDoc] = []
    if repo_url:
        all_docs += ingest_repo(repo_url)
    if code_folder:
        all_docs += ingest_code_folder(code_folder)
    if docs_folder:
        all_docs += ingest_docs_folder(docs_folder)
    if pdf_folder:
        all_docs += ingest_pdfs(pdf_folder)
    count = lambda t: sum(d.doc_type == t for d in all_docs)  # noqa: E731
    print(f"[ingest] {len(all_docs)} raw docs "
          f"(code={count('code')}, doc={count('doc')}, pdf={count('pdf')})")
    return all_docs

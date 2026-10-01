"""Step 2: AST-aware split for Python, paragraph-aware char split for the rest."""
import hashlib

from . import config
from .schemas import Chunk, RawDoc

_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tspython.language()))
    return _parser


def _make_id(source: str, index: int) -> str:
    h = hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:8]
    return f"{h}_{index}"


def recursive_split(doc: RawDoc) -> list[Chunk]:
    """Char-based splitter with overlap; prefers to cut at blank lines."""
    text = doc.text
    n = len(text)
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    chunks: list[Chunk] = []
    pos = idx = 0

    while pos < n:
        end = min(pos + size, n)
        piece = text[pos:end]
        if end < n:
            brk = piece.rfind("\n\n")
            if brk > size * 0.6:
                end = pos + brk
                piece = text[pos:end]
        if piece.strip():
            chunks.append(Chunk(
                chunk_id=_make_id(doc.source, idx),
                text=piece.strip(),
                source=doc.source,
                doc_type=doc.doc_type,
                metadata=dict(doc.metadata),
            ))
            idx += 1
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)  # always move forward
    return chunks


def chunk_python_code(doc: RawDoc) -> list[Chunk]:
    """One chunk per top-level function/class. Oversized nodes get split further."""
    src = doc.text.encode("utf8")
    tree = _get_parser().parse(src)
    header = f"# File: {doc.source}\n"
    chunks: list[Chunk] = []
    idx = 0

    for node in tree.root_node.children:
        if node.type not in ("function_definition", "class_definition", "decorated_definition"):
            continue
        snippet = src[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
        meta = {**doc.metadata, "node_type": node.type}

        if len(snippet) > config.MAX_CODE_CHUNK:
            sub = recursive_split(RawDoc(snippet, doc.source, "code", meta))
            for s in sub:
                s.chunk_id = _make_id(doc.source, idx)
                s.text = header + s.text
                chunks.append(s)
                idx += 1
        else:
            chunks.append(Chunk(
                chunk_id=_make_id(doc.source, idx),
                text=header + snippet,
                source=doc.source,
                doc_type="code",
                metadata=meta,
            ))
            idx += 1

    return chunks or recursive_split(doc)


def chunk_all(raw_docs: list[RawDoc]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in raw_docs:
        if doc.doc_type == "code" and doc.metadata.get("language") == "py":
            all_chunks += chunk_python_code(doc)
        else:
            all_chunks += recursive_split(doc)
    print(f"[chunk] {len(all_chunks)} chunks from {len(raw_docs)} docs")
    return all_chunks

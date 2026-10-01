"""CLI for the RAG bot.

  python main.py build [--repo URL] [--docs DIR] [--pdfs DIR] [--code DIR]
  python main.py ask "your question"
  python main.py chat
  python main.py eval [--n 20] [--regen]
"""
import argparse

from rag import config


def print_answer(result):
    print("\n=== ANSWER ===")
    print(result.answer)
    print(f"\nConfidence: {result.confidence}")
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  - [{c.chunk_id}] {c.source}")


def cmd_build(args):
    from rag.chunking import chunk_all
    from rag.indexing import build_all_indices
    from rag.ingest import ingest_all

    raw = ingest_all(
        repo_url=args.repo,
        docs_folder=args.docs,
        pdf_folder=args.pdfs,
        code_folder=args.code,
    )
    chunks = chunk_all(raw)
    build_all_indices(chunks)
    print("Build done. Try:  python main.py chat")


def cmd_ask(args):
    from rag.generation import generate_answer
    from rag.retrieval import HybridRetriever

    retriever = HybridRetriever()
    top = retriever.retrieve(args.question)
    print_answer(generate_answer(args.question, top))


def cmd_chat(_args):
    from rag.generation import generate_answer
    from rag.retrieval import HybridRetriever

    retriever = HybridRetriever()
    print("RAG chat. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        print_answer(generate_answer(q, retriever.retrieve(q)))


def cmd_eval(args):
    from rag import evaluation as ev
    from rag.retrieval import HybridRetriever

    if args.regen or not config.EVAL_SET_PATH.exists():
        pairs = ev.generate_eval_set(ev.load_chunks(), sample_size=args.n)
        ev.save_eval_set(pairs)
    else:
        pairs = ev.load_eval_set()
        print(f"[eval] loaded {len(pairs)} pairs from {config.EVAL_SET_PATH.name} "
              f"(use --regen for a new set)")
    ev.run_retrieval_eval(pairs, HybridRetriever())


def main():
    ap = argparse.ArgumentParser(description="Hybrid-retrieval RAG bot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="ingest + chunk + index")
    b.add_argument("--repo", help="git repo URL to clone and index")
    b.add_argument("--docs", default=str(config.DEFAULT_DOCS_DIR), help="folder of .md/.txt/.html")
    b.add_argument("--pdfs", default=str(config.DEFAULT_PDF_DIR), help="folder of PDFs")
    b.add_argument("--code", default=str(config.DEFAULT_CODE_DIR), help="folder of local source code")
    b.set_defaults(fn=cmd_build)

    a = sub.add_parser("ask", help="one question")
    a.add_argument("question")
    a.set_defaults(fn=cmd_ask)

    c = sub.add_parser("chat", help="interactive loop")
    c.set_defaults(fn=cmd_chat)

    e = sub.add_parser("eval", help="retrieval hit-rate + MRR")
    e.add_argument("--n", type=int, default=20, help="chunks to sample")
    e.add_argument("--regen", action="store_true", help="regenerate eval set")
    e.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

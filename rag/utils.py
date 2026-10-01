import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Code-friendly tokenizer for BM25: splits snake_case and camelCase."""
    tokens: list[str] = []
    for word in _TOKEN_RE.findall(text.replace("_", " ")):
        tokens.extend(p.lower() for p in _CAMEL_RE.split(word) if p)
    return tokens

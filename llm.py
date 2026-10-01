"""One function, two backends: generate_json(prompt, schema) -> JSON text.
Provider chosen by LLM_PROVIDER in .env  ("gemini", "groq", "xai" = Grok, "ollama" = local)."""
import json
import time

from pydantic import BaseModel

from . import config

_gemini_client = None
_ollama_client = None


def extract_json(text: str) -> str:
    """Grab the outermost {...} so stray markdown fences or chatter don't break parsing."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _schema_hint(schema: type[BaseModel]) -> str:
    return ("\n\nReply with ONLY one valid JSON object (no markdown, no extra text) "
            "that matches this JSON schema:\n" + json.dumps(schema.model_json_schema()))


# ---------------- OpenAI-compatible providers (Groq, xAI Grok) ----------------
_oa_clients: dict = {}


def _openai_compat(name: str, prompt: str, schema: type[BaseModel],
                   api_key: str | None, base_url: str, model: str, key_hint: str) -> str:
    if not api_key:
        raise RuntimeError(f"{name.upper()}_API_KEY missing. Add it to .env ({key_hint}).")
    from openai import BadRequestError, OpenAI

    client = _oa_clients.get(name)
    if client is None:
        client = _oa_clients[name] = OpenAI(api_key=api_key, base_url=base_url,
                                            max_retries=4)  # SDK auto-retries 429 / 5xx
    messages = [{"role": "user", "content": prompt + _schema_hint(schema)}]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"},
        )
    except BadRequestError:
        # some models reject response_format; schema hint in prompt still works
        resp = client.chat.completions.create(model=model, messages=messages)
    return extract_json(resp.choices[0].message.content or "")


def _xai(prompt: str, schema: type[BaseModel]) -> str:
    return _openai_compat("xai", prompt, schema, config.XAI_API_KEY, config.XAI_BASE_URL,
                          config.XAI_MODEL, "key from console.x.ai, starts with xai-")


def _groq(prompt: str, schema: type[BaseModel]) -> str:
    return _openai_compat("groq", prompt, schema, config.GROQ_API_KEY, config.GROQ_BASE_URL,
                          config.GROQ_MODEL, "key from console.groq.com/keys, starts with gsk_")


# ---------------- Google Gemini ----------------
def _gemini(prompt: str, schema: type[BaseModel], tries: int = 3) -> str:
    global _gemini_client
    from google import genai
    from google.genai import errors as genai_errors

    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing. Add it to .env.")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

    for attempt in range(tries):
        try:
            resp = _gemini_client.models.generate_content(
                model=config.GEMINI_MODEL, contents=prompt,
                config={"response_mime_type": "application/json", "response_schema": schema},
            )
            return extract_json(resp.text)
        except genai_errors.APIError as e:
            if attempt == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


# ---------------- Ollama (local, no key) ----------------
def _ollama(prompt: str, schema: type[BaseModel]) -> str:
    global _ollama_client
    import ollama

    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.OLLAMA_HOST) if config.OLLAMA_HOST else ollama.Client()
    try:
        resp = _ollama_client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),   # constrains output to the schema
            options={"temperature": 0},
        )
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed: {e}\n"
            f"Check: 1) Ollama app is running  2) model is pulled:  ollama pull {config.OLLAMA_MODEL}"
        ) from e
    return extract_json(resp["message"]["content"])


def generate_json(prompt: str, schema: type[BaseModel]) -> str:
    if config.LLM_PROVIDER == "groq":
        return _groq(prompt, schema)
    if config.LLM_PROVIDER == "xai":
        return _xai(prompt, schema)
    if config.LLM_PROVIDER == "gemini":
        return _gemini(prompt, schema)
    if config.LLM_PROVIDER == "ollama":
        return _ollama(prompt, schema)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'gemini', 'groq', 'xai' or 'ollama'.")

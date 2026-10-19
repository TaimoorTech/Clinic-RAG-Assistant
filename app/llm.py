"""LLM client built on LangChain's ChatOpenAI.

Settings live in a .env file in the project root (never committed to git):
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=<a chat model name from OpenAI's model list>
"""
import os

from dotenv import load_dotenv

load_dotenv()  # copies the values from .env into os.environ


class LLMError(Exception):
    """The LLM call failed (network, quota, bad response...)."""


class ConfigError(LLMError):
    """Settings are missing or invalid."""


def get_llm(env=os.environ):
    api_key = env.get("OPENAI_API_KEY", "").strip()
    model = env.get("OPENAI_MODEL", "").strip()
    missing = [n for n, v in (("OPENAI_API_KEY", api_key), ("OPENAI_MODEL", model)) if not v]
    if missing:
        raise ConfigError("Missing setting(s) in .env: " + ", ".join(missing))

    from langchain_openai import ChatOpenAI  # imported here so tests don't need it

    # No temperature / max_tokens on purpose: some newer models reject custom
    # values, and a token cap can cut off models that "think" before answering.
    return ChatOpenAI(model=model, api_key=api_key, timeout=30, max_retries=2)


def generate(system: str, user: str, llm=None) -> str:
    """Send one system + user message and return the answer text."""
    llm = llm or get_llm()
    try:
        resp = llm.invoke([("system", system), ("human", user)])
    except Exception as exc:  # LangChain/OpenAI raise many different error types
        raise LLMError(f"OpenAI call failed: {exc.__class__.__name__}: {str(exc)[:200]}") from exc

    content = resp.content
    if not isinstance(content, str):  # some models return a list of content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    content = content.strip()
    if not content:
        raise LLMError("OpenAI returned an empty answer")
    return content
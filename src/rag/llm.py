"""LLM wrapper — optional Ollama / OpenAI integration."""

import logging

logger = logging.getLogger(__name__)

# Optional imports — the system works without any LLM installed
try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None

try:
    from langchain_openai import OpenAI as LangchainOpenAI
except ImportError:
    LangchainOpenAI = None


def get_llm(provider: str = "ollama", model: str = "mistral",
            endpoint: str = "http://localhost:11434", temperature: float = 0.3):
    """
    Create a LangChain LLM instance.

    Returns None if the provider is not available.
    """
    if provider == "ollama":
        if Ollama is None:
            logger.warning("langchain-community not installed — Ollama unavailable")
            return None
        try:
            llm = Ollama(model=model, base_url=endpoint, temperature=temperature)
            # quick connectivity check
            llm.invoke("hi")
            logger.info(f"Ollama connected ({model} @ {endpoint})")
            return llm
        except Exception as e:
            logger.warning(f"Ollama not reachable: {e}")
            return None

    if provider == "openai":
        if LangchainOpenAI is None:
            logger.warning("langchain-openai not installed")
            return None
        try:
            llm = LangchainOpenAI(model=model, temperature=temperature)
            logger.info(f"OpenAI connected ({model})")
            return llm
        except Exception as e:
            logger.warning(f"OpenAI init failed: {e}")
            return None

    logger.warning(f"Unknown LLM provider: {provider}")
    return None

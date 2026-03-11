import logging
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def ollama_client(*, url: str, token: str | None, model: str, proxy: str | None) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=0,
        base_url=url,
        client_kwargs={
            'headers': {'Authorization': f'Bearer {token}'} if token else {},
            'proxy': proxy,
        },
    )


def openai_client(*, url: str, token: str | None, model: str, proxy: str | None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url=url,
        api_key=token,
    )


# TODO: set up the right llm for user_config

llm_kwargs = {
    'url': os.environ.get('LLM_API_URL'),
    'token': os.environ.get('LLM_API_TOKEN') or None,
    'model': os.environ.get('LLM_API_MODEL'),
    'proxy': os.environ.get('LLM_API_PROXY') or None,
}
logging.info('LLM args: %s', llm_kwargs)

llm = globals()[os.environ.get('LLM_API_TYPE', 'ollama_client')](**llm_kwargs)

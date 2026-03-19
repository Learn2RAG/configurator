import logging
import os
from pydantic import SecretStr
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Callable, Any


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


def openai_client(*, url: str, token: SecretStr, model: str, proxy: str | None) -> ChatOpenAI:
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

# the keys are written by the configurator UI
llms: dict[str, Callable[..., Any]] = {
    'ChatOllama': ollama_client,
    'ChatOpenAI': openai_client,
}

llm = llms[os.environ.get('LLM_API_TYPE', 'ChatOllama')](**llm_kwargs)

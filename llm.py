import logging
import os

import langchain_ollama
import langchain_openai


def ChatOllama(*, url, token, model, proxy):
    return langchain_ollama.ChatOllama(
        model=model,
        temperature=0,
        base_url=url,
        client_kwargs={
            'headers': {'Authorization': f'Bearer {token}'} if token else {},
            'proxy': proxy,
        },
    )


def ChatOpenAI(*, url, token, model, proxy):
    return langchain_openai.ChatOpenAI(
        model=model,
        temperature=0,
        base_url=url,
        api_key=token,
    )


# TODO: set up the right llm for user_config

llm_kwargs = {
    'url': os.environ['LLM_API_URL'],
    'token': os.environ.get('LLM_API_TOKEN'),
    'model': os.environ['LLM_API_MODEL'],
    'proxy': os.environ.get('LLM_API_PROXY'),
}
logging.info('LLM args: %s', llm_kwargs)

llm = globals()[os.environ.get('LLM_API_TYPE', 'ChatOllama')](**llm_kwargs)

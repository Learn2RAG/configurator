import logging
import os

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
import langchain_core.prompts
import langchain_ollama
import langchain_openai
import langchain_redis
import redis


# https://python.langchain.com/api_reference/redis/vectorstores/langchain_redis.vectorstores.RedisVectorStore.html
class RedisVectorStore(langchain_redis.RedisVectorStore):
    def __init__(self, embeddings: Embeddings, index_name: str) -> None:
        super().__init__(embeddings, config=langchain_redis.RedisConfig(
            index_name=index_name,
            redis_client=redis.from_url(
                os.environ.get('REDIS_URL', 'redis://localhost:6379'),
                retry=redis.retry.Retry(redis.backoff.ExponentialBackoff(), 10),
            ),
            metadata_schema=[
                {'name': 'category', 'type': 'tag'},
            ],
        ))


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


load_prompt = langchain_core.prompts.loading.load_prompt

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

llm_kwargs = {
    'url': os.environ['LLM_API_URL'],
    'token': os.environ.get('LLM_API_TOKEN'),
    'model': os.environ['LLM_API_MODEL'],
    'proxy': os.environ.get('LLM_API_PROXY'),
}
logging.info('LLM args: %s', llm_kwargs)

llm = globals()[os.environ.get('LLM_API_TYPE', 'ChatOllama')](**llm_kwargs)

__all__ = [
    'embeddings',
    'llm',
    'load_prompt',
    'InMemoryVectorStore',
    'RedisVectorStore',
]

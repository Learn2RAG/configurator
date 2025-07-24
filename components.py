import logging
import os

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.embeddings.embeddings import Embeddings
import langchain_core.prompts
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


load_prompt = langchain_core.prompts.loading.load_prompt

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

ollama_url = os.environ.get('OLLAMA_URL')
logging.info('Using Ollama URL: %s', ollama_url)
ollama_proxy = os.environ.get('OLLAMA_PROXY') or None
logging.info('Using proxy for Ollama: %s', ollama_proxy)
llm_client_headers = {}
if ollama_auth := os.environ.get('OLLAMA_AUTH'):
    llm_client_headers['Authorization'] = ollama_auth
llm = ChatOllama(
    model='llama3.3:70b',
    temperature=0,
    base_url=ollama_url,
    client_kwargs={
        'headers': llm_client_headers,
        'proxy': ollama_proxy,
    },
)

__all__ = [
    'embeddings',
    'llm',
    'load_prompt',
    'InMemoryVectorStore',
    'RedisVectorStore',
]

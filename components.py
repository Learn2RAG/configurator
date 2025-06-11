import os

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_ollama import ChatOllama
import langchain_core.embeddings.embeddings
import langchain_redis


# https://python.langchain.com/api_reference/redis/vectorstores/langchain_redis.vectorstores.RedisVectorStore.html
class RedisVectorStore(langchain_redis.RedisVectorStore):
    def __init__(self, embeddings: langchain_core.embeddings.embeddings.Embeddings) -> None:
        super().__init__(embeddings, config=langchain_redis.RedisConfig(
            index_name='test',
            redis_url=os.environ.get('REDIS_URL', 'redis://localhost:6379'),
            metadata_schema=[
                {'name': 'category', 'type': 'tag'},
            ],
        ))


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# Define prompt for question-answering
prompt = hub.pull('rlm/rag-prompt')

llm = ChatOllama(
    model='llama3.3:70b',
    temperature=0,
    base_url=os.environ.get('OLLAMA_URL'),
    client_kwargs={
        'headers': {
            'Authorization': os.environ.get('OLLAMA_AUTH'),
        },
        # 'proxy': 'socks5://HOST:PORT',
    },
)

__all__ = [
    'embeddings',
    'llm',
    'prompt',
    'InMemoryVectorStore',
    'RedisVectorStore',
]

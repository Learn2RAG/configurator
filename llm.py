import logging
import os

from langchain_ollama import ChatOllama

# TODO: set up the right llm for user_config

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
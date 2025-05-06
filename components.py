from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vector_store = InMemoryVectorStore(embeddings)

# Define prompt for question-answering
prompt = hub.pull('rlm/rag-prompt')

llm = ChatOllama(
    model='llama3.3:70b',
    temperature=0,
    base_url='',
    client_kwargs={'headers': {'Authorization': 'Bearer X'}},
)

#%%
import os
from uuid import uuid4
from tqdm import tqdm
import sys

from datasets import load_dataset
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import BGEM3FlagModel

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, MultiVectorConfig, MultiVectorComparator, PointStruct, SparseVector

from langchain_qdrant import QdrantVectorStore


from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader

from langchain_ollama import ChatOllama

from .config import user_config


#%%
# Load documents from repliqa

pdf_dir = "data/pdfs/repliqa_4"
all_documents = []

for file in os.listdir(pdf_dir)[:10]:
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, file), mode='single') # very important: default of mode is page-vise!
        docs = loader.load()
        all_documents.extend(docs)
#%%
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_documents)

chunks[0].metadata['source']
chunks[0].metadata['author']
chunks[0].page_content


#%%
# Initialize embeddingmodel
#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
vectors = model.encode(chunks["chunk"][0:10], batch_size=512, return_dense=True, return_sparse=True, return_colbert_vecs=False)

#%%
# Setup Qdrant client
qdrant = QdrantClient(
    host="localhost",
    port=6333,
    api_key=user_config['qdrant']['api_key'],
    https=False,
)
# Create collection if not exists
collection_name = "Learn2RAG-repliqa_4"

if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        }
    )

chunks_list = chunks.to_list()
chunks_with_vectors = [
    chunk | {'dense_vec': dense, 'sparse_vec': sparse}
    for chunk, dense, sparse in zip(chunks_list, list(vectors['dense_vecs']), list(vectors['lexical_weights']))]
#%%

def insert(example: dict):
    qdrant.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            PointStruct(
                id=uuid4().hex,
                vector={
                    "dense": example["dense_vec"],
                    "sparse": SparseVector(
                        indices=[int(x) for x in example["sparse_vec"].keys()],
                        values=example["sparse_vec"].values(),
                    )
                },
                payload={"docid": example["c_documentid"],
                    # "category_ids": example['category_id_paths'],
                    "language": example["c_language"],
                    "content": example["chunk"],
                    "chunk_idx": example["chunk_index"]}),
        ],
    )   

for example in tqdm(chunks_with_vectors, file=sys.stdout):
    insert(example)


#if not qdrant.collection_exists(collection_name):
#    qdrant.create_collection(
#        collection_name=collection_name,
#    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
#)

#%%
# Store documents in Qdrant via LangChain
vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=collection_name,
    embedding=embeddings,
)


#%%

uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)

#%%
#similarity search
query = "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
results = vector_store.similarity_search(query, k=4)

for result in results:
    print(result.metadata['source'])

#%%
#Setup LLm
ollama_url = os.environ.get('OLLAMA_URL')
ollama_proxy = os.environ.get('OLLAMA_PROXY') or None
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

#%%

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = """Kontext:
{context}

Beantworte die folgende Frage:
{question}
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

context = "\n\n".join([doc.page_content for doc in results])

response = chain.run(context=context, question=query)

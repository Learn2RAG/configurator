import json
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant import Qdrant
import loaders


def index(user_config, opt_config):
    # TODO: enable list of file paths in loader and adapt user_config
    # Load the documents from pdf
    all_documents = loaders.sync_pdf_loader(user_config["file_path"])
    # TODO: use ifdt loader to load pdf in json, then:
    # all_documents = loaders.json_loader("loaded_documents.json")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=opt_config["chunk_size"], chunk_overlap=opt_config["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(all_documents)

    # Initialize embeddingmodel
    encoder = HuggingFaceEmbeddings(model_name=opt_config["embedding_model"])

    # Init vector store
    qdrant = Qdrant(
        collection_name=user_config["collection_name"],
        encoder=encoder,
        vector_size=opt_config["vector_size"][opt_config["embedding_model"]],
    )

    # Store documents in Qdrant via LangChain
    # TODO: do not ingest same chunks twice
    # TODO: ingest without using langchain
    # TODO: add logging
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    qdrant.vector_store.add_documents(documents=chunks, ids=uuids)

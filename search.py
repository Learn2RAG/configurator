from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant import Qdrant


#similarity search
def search(query, user_config, opt_config) -> list:

    # Initialize embeddingmodel
    encoder = HuggingFaceEmbeddings(model_name=opt_config['embedding_model'])

    # Init vector store
    qdrant = Qdrant(
        collection_name=user_config['collection_name'],
        encoder=encoder,
        vector_size=opt_config['vector_size'][opt_config['embedding_model']]
    )

    results = qdrant.vector_store.similarity_search(query, opt_config['top_k'])

    return results
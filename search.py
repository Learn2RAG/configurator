from qdrant import Qdrant
from embeddings import create_embeddings

# similarity search
def search(query, user_config, opt_config) -> list:
    collection_name = user_config["collection_name"]
    # Init vector store
    qdrant = Qdrant(
        collection_name=collection_name,
        vector_size=opt_config["vector_size"][opt_config["embedding_model"]],
    )

    if opt_config["embedding_model"] == "BAAI/bge-m3":
        query_embedding = create_embeddings(query, opt_config["embedding_model"])["dense_vecs"]
    else:
        query_embedding = create_embeddings(query, opt_config["embedding_model"])

    results = qdrant.client.search(
        collection_name=collection_name,
        query_vector=("dense", query_embedding),
        limit=opt_config["top_k"],
    )

    return results


def extract_context_content(search_results) -> str:
    contents = []
    for result in search_results:
        if hasattr(result, "payload") and "content" in result.payload:
            contents.append(result.payload["content"])
    return "\n".join(contents)


def context_content(query, user_config, opt_config) -> str:
    results = search(query, user_config, opt_config)
    return extract_context_content(results)

# %%
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant import Qdrant


# Read eval data
df = pd.read_parquet("data/data/repliqa_4-00000-of-00001.parquet")
queries = df["question"]
labels = df["document_id"]


collection_name = "Learn2RAG-repliqa_4_dense"
vector_size = 1024
encoder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True}
)
qdrant = Qdrant(collection_name, encoder, vector_size)


def recall(search_results, labels):
    count = 0
    for label, search_result in zip(
        labels, [[hit for hit in result] for result in search_results]
    ):
        print(label)
        print(search_result)
        if label in search_result:
            count += 1
    return count / len(labels)


search_results = [
    [
        Qdrant.vector_store.similarity_search(query, k=4)[i]
        .metadata["source"]
        .split("/")[-1]
        .split(".pdf")[0]
        for i in range(4)
    ]
    for query in queries
]

# %%
print("ONLY DENSE: Recall = ", recall(search_results, labels))


recall(search_results, labels)

# 0.5665738161559889 # dense, sentence-transformers/all-mpnet-base-v2, k=4
# 0.7894707520891365 # dense, bgem3, k=4


# %%

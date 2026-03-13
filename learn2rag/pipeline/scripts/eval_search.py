# %%
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..qdrant import Qdrant
from .. import search
from ..config import user_config, opt_config


# Read eval data
#df = pd.read_parquet("data/data/repliqa_4-00000-of-00001.parquet")
#queries = df["question"]
#labels = df["document_id"]

df = pd.read_csv('/home/usicwalter/l2r/learn2rag/pipeline/data/eval_samples_chatgpt.csv', sep=";")
queries = df['query'][:2]
labels = df["document_id"][:2]

dg = pd.read_csv('/home/usicwalter/l2r/learn2rag/pipeline/data/eval_samples_csc.csv', sep=";")
all_queries = pd.concat([queries, dg['query']])
all_labels = pd.concat([labels, dg["document_id"]])


#query = {"content": "What is USM AI?", "title": "USM AI Documentation", "summary": "In this document the basic usage of USM AI is described.", "source_path":"USU/ITSM/"}

def mmquery(query):
    return {"content": query, "title": query, "summary": query, "source_path": query}

if opt_config["search_mode"] == "multi_search":
    search_results = [[search.search(mmquery(query), user_config, opt_config)] for query in queries]
else:
    search_results = [[search.search(query, user_config, opt_config)] for query in queries]

def recall(search_results, labels):
    count = 0
    for q in range(len(search_results)):
        hits = [search_results[q][0].points[i].payload['path'] for i in range(opt_config['top_k'])]
        label = labels[q]
        print('label ',label, ' hits: ',hits)
        if str(label) in hits:
            count += 1
    return count / len(labels)

def retrieve_repliqa(queries):
    return [
        [
            Qdrant.vector_store.similarity_search(query, k=4)[i]
            .metadata["source"]
            .split("/")[-1]
            .split(".pdf")[0]
            for i in range(4)
        ]
        for query in queries
    ]


print(recall(search_results, labels))

#print("ONLY DENSE: Recall = ", recall(retrieve_repliqa(queries), labels))
# data/data/repliqa_4-00000-of-00001.parquet:
# 0.5665738161559889 # dense, sentence-transformers/all-mpnet-base-v2, k=4
# 0.7894707520891365 # dense, bgem3, k=4

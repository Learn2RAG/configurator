import json
import subprocess
import os
from pathlib import Path
from itertools import product
from copy import deepcopy
import time
import pandas as pd

from ..qdrant import Qdrant
from .. import search
from ..config import user_config


df = pd.read_csv('/home/usicwalter/l2r/learn2rag/pipeline/data/eval_samples_chatgpt.csv', sep=";")
queries = df['query'][:2]
labels = df["document_id"][:2]

BASE_CONFIG = {
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "embedding_model": "BAAI/bge-m3",
    "vector_size": {
      "sentence-transformers/all-mpnet-base-v2": 768,
      "BAAI/bge-m3": 1024
    },
    "prefetch_limit_dense": 25,
    "prefetch_limit_sparse": 25,
    "prefetch_limit_colbert": 25,
    "fusion_mode": "DBSF",
    "multi_search": ["title", "summary", "source_path"],
    "prompt": "# Role and Objective\\nYou will act as a smart AI chatbot that answers questions only by using the content from the provided information list.\\n\\n# Instructions\\nYour rules for answering:\\n- You should respond in the language in which the Current Question has been asked.\\n- You should answer clear and concise.\\n- IMPORTANT: Only use the information provided after 'Information' to answer the user question.\\n- NEVER use your general knowledge.\\n\\n## Sub-categories for detailed instructions\\n- You will be given excerpts of information that come from various sources of information and have a 'Source' and 'Content'. The excerpts from information sources are separated by lines of '-----'.\\n\\n# Reasoning Steps\\n- Decide which excepts of information are relevant to the question.\\n- Revise your information list and only keep excerpts of information that contain parts of your answer.\\n- Always only use your pre-selected sources of information.\\n- If the provided information does not contain the answer: Let me know and never answer the user question.\\n\\n# Output Format\\n- Always use Mardown as the output format for your entire answer.\\n- If you refer to sources of information within your answer, please use the Source.\\n- Sort the pre-selected information by importance. The most relevant information should be at the top. List your pre-selected sources of information in a bulleted list at the end of your answer, only using the Source. If you can't answer to the user question, don't provide a list.\\n\\n# Information:\\n{context}"
  }


PARAM_GRID = {
    "search_mode": ["multi_search"], #["dense", "dense_sparse", "dense_sparse_colbert", "reranking_with_colbert", "reranking_with_flagreranker"],
    "top_k": [4,10]
  }

def update_config(base, param_values):
    config = deepcopy(base)
    for key, value in param_values.items():
        config[key] = value
    return config

def write_config(config, filepath="config.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def mmquery(query):
    return {"content": query, "title": query, "summary": query, "source_path": query}

def recall(search_results, labels, top_k):
    count = 0
    for q in range(len(search_results)):
        if not hasattr(search_results[0][0], "points"):
            hits = [search_results[q][0][i].payload['path'] for i in range(top_k)]
        else:   
            hits = [search_results[q][0].points[i].payload['path'] for i in range(top_k)]
        label = labels[q]
        print('label ', label, ' hits: ', hits)
        if str(label) in hits:
            count += 1
    return count / len(labels)

def main():
    config_path = "learn2rag/pipeline/opt_config.json"
    
    all_results = []
    
    param_keys = list(PARAM_GRID.keys())
    param_values_list = [PARAM_GRID[key] for key in param_keys]
    
    for combo_idx, values in enumerate(product(*param_values_list)):
        param_values = dict(zip(param_keys, values))
        print(f"Run {combo_idx + 1}: {param_values}")
        
        config = update_config(BASE_CONFIG, param_values)
        write_config(config, config_path)
        
        with open(os.environ.get("PIPELINE_OPT_CONFIG", "learn2rag/pipeline/opt_config.json"), "r") as file:
            opt_config = json.load(file)

        start = time.time()
        if opt_config["search_mode"] == "multi_search":
            search_results = [[search.search(mmquery(query), user_config, opt_config)] for query in queries]
        else:
            search_results = [[search.search(query, user_config, opt_config)] for query in queries]
        recall_value = recall(search_results, labels, opt_config['top_k'])
        end = time.time()
        duration = end - start
        print(recall_value)
        all_results.append({"config": param_values, "recall": recall_value, "time": duration})

        del opt_config
        
    print(all_results)
    with open("eval_csc_chatgpt_mm.json", "w") as f:
        json.dump(all_results, f, indent=2)
    

if __name__ == "__main__":
    main()

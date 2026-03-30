import logging
import logging.config
import os
import yaml
import json

from . import ingestion
from . import search
from . import generate
from . import rewrite


if __name__ == "__main__":
    try:
        logging.config.dictConfig(yaml.safe_load(open("logging.yaml").read()))
    except FileNotFoundError:
        logging.basicConfig()

    from .config import user_config, opt_config

    # ingestion.index(user_config, opt_config)

    if opt_config["query_mode"] == "multi":
        # in query_mode 'multi' different querys for each vector in the multi-vector are allowed
        multi_query = {"content": "What is USM AI?", "title": "What is USM AI?", "summary": "What is USM AI?", "source_path":"USU/ITSM/"}
        results = search.search_multi(multi_query, user_config, opt_config, request_id=None)
        # modify the query for generation part
        query = " ".join(f"{k}={v}" for k, v in multi_query.items())
    else: 
        query = "What si USM AI?" #"What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"

        rewritten_query = rewrite.rewrite_query(query)
        if rewritten_query in (None, ''):
            q = query
        else:
            q = rewritten_query
        subqueries = rewrite.generate_subqueries(q, n=5)
        keywords = rewrite.generate_keywords(q, n=5)

        results = search.search(query, user_config, opt_config, request_id=None)

        opt_config_rewritten_query = opt_config
        opt_config_rewritten_query["top_k"] = 3
        results_rewritten_query = search.search(rewritten_query, user_config, opt_config_rewritten_query, request_id=None)

        opt_config_subqueries = opt_config
        opt_config_subqueries["top_k"] = 3
        results_subqueries = []
        for sq in subqueries:
            results_subqueries.append(search.search(sq, user_config, opt_config_subqueries, request_id=None))

        opt_config_keywords = opt_config
        # opt_config_keywords["search_mode"] = "sparse"
        opt_config_keywords["top_k"] = 3

        keywords_concatenated = " ".join(keywords)
        results_keywords_concatenated = search.search(keywords_concatenated, user_config, opt_config_keywords, request_id=None)

        results_keywords_single = []
        for kw in keywords:
            results_keywords_single.append(search.search(kw, user_config, opt_config_keywords, request_id=None))

    points_all = []
    for point in results_rewritten_query.points:
        points_all.append(point)
    for item in results_subqueries:
        for point in item.points:
            points_all.append(point)
    for item in results_keywords_single: # alternative: results_keywords_concatenated
        for point in item.points:
            points_all.append(point)

    points = sorted(points_all, key=lambda p: p.score, reverse=True)
    # todo reranking

    sources = "\n".join(set(point.payload['path'] for point in points)) # type: ignore[index]

    for point in points:
        print(f"ID: {point.id}, Path: {point.payload['path']}, Score: {point.score}") # type: ignore[index]

    answer = generate.generate(query, points, opt_config)

    print(query)
    print(answer)
    print(sources)

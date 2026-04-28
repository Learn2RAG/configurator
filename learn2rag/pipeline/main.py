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
    points_all = []
    if opt_config["query_mode"] == "multi":
        # in query_mode 'multi' different querys for each vector in the multi-vector are allowed
        multi_query = {"content": "What is USM AI?", "title": "What is USM AI?", "summary": "What is USM AI?", "source_path":"USU/ITSM/"}
        results = search.search_multi(multi_query, user_config, opt_config, request_id=None)
        for point in results.points:
            points_all.append(point)
        # modify the query for generation part
        query = " ".join(f"{k}={v}" for k, v in multi_query.items())
    else: 
        query = "Was sind A, B und C?" #"What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
        results = search.search(query, user_config, opt_config, request_id=None)
        for point in results.points:
            points_all.append(point)

        if opt_config["rewrite"] == "True":
            opt_config_rewritten_query = dict(opt_config)
            opt_config_rewritten_query["top_k"] = 3

            if opt_config["rewrite_mode"] in ["subqueries", "subqueries_keywords"]:
                subqueries = rewrite.generate_subqueries(query, n=2)
                print(f"SUBQUERIES: {subqueries}")
                for sq in subqueries:
                    res_sq = search.search(sq, user_config, opt_config_rewritten_query, request_id=None)
                    points_all.extend(res_sq.points)

            if opt_config["rewrite_mode"] in ["keywords", "subqueries_keywords"]:
                opt_config_keywords = dict(opt_config_rewritten_query)
                opt_config_keywords["search_mode"] = "sparse"
                keywords = rewrite.generate_keywords(query, n=5)
                # keywords_concatenated = " ".join(keywords)
                # res_kw = search.search(keywords_concatenated, user_config, opt_config_keywords, request_id=None)
                # points_all.extend(res_kw.points)
                print(f"KEYWORDS: {keywords}")
                for kw in keywords:
                    res_kw = search.search(kw, user_config, opt_config_keywords, request_id=None)
                    points_all.extend(res_kw.points)

    points = sorted(points_all, key=lambda p: p.score, reverse=True)
    # TODO: reranking? clipping?

    sources = "\n".join(set(point.payload['path'] for point in points)) # type: ignore[index]

    for point in points:
        print(f"ID: {point.id}, Path: {point.payload['path']}, Score: {point.score}") # type: ignore[index]

    answer = generate.generate(query, points, opt_config)

    print(query)
    print(answer)
    print(sources)

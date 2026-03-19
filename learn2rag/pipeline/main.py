import logging
import logging.config
import os
import yaml
import json

from . import ingestion
from . import search
from . import generate


if __name__ == "__main__":
    try:
        logging.config.dictConfig(yaml.safe_load(open("logging.yaml").read()))
    except FileNotFoundError:
        logging.basicConfig()

    from .config import user_config, opt_config

    #ingestion.index(user_config, opt_config)

    if opt_config["query_mode"] == "multi":
        # in query_mode 'multi' different querys for each vector in the multi-vector are allowed
        multi_query = {"content": "What is USM AI?", "title": "What is USM AI?", "summary": "What is USM AI?", "source_path":"USU/ITSM/"}
        results = search.search_multi(multi_query, user_config, opt_config)
        # modify the query for generation part
        query = " ".join(f"{k}={v}" for k, v in multi_query.items())
    else: 
        query = "What is USM AI?" #"What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
        results = search.search(query, user_config, opt_config)

    points = results.points

    sources = "\n".join(set(point.payload['path'] for point in points)) # type: ignore[index]

    for point in points:
        print(f"ID: {point.id}, Path: {point.payload['path']}, Score: {point.score}") # type: ignore[index]

    answer = generate.generate(query, points, opt_config)

    print(query)
    print(answer)
    print(sources)

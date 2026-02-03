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

    ingestion.index(user_config, opt_config)

    query = "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
    if opt_config["search_mode"] == "multi_search":
        query = {"content": "What is USM AI?", "title": "USM AI Documentation", "summary": "In this document the basic usage of USM AI is described.", "source_path":"USU/ITSM/"}
    results = search.search(query, user_config, opt_config)

    if hasattr(results, "points"):
        results = results.points

    sources = "\n".join(set(result.payload['path'] for result in results))

    for result in results:
        print(f"ID: {result.id}, Path: {result.payload['path']}, Score: {result.score}")

    answer = generate.generate(query, results, opt_config)

    print(query)
    print(answer)
    print(sources)

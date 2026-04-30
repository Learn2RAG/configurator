import logging.config
import yaml
import asyncio

from . import ingestion
from . import search
from . import generate


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
        points = results.points
        # modify the query for generation part
        query = " ".join(f"{k}={v}" for k, v in multi_query.items())
    else:
        query = "Was sind A, B und C?"
        user = "anonymous"
        points = asyncio.run(search.search_authorized(query, user, request_id=None))

    sources = "\n".join(set(point.payload['path'] for point in points)) # type: ignore[index]

    for point in points:
        print(f"ID: {point.id}, Path: {point.payload['path']}, Score: {point.score}") # type: ignore[index]

    answer = generate.generate(query, points, opt_config)

    print(query)
    print(answer)
    print(sources)

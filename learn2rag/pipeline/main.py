import asyncio
import json
import logging
import logging.config
import yaml
from operator import itemgetter

from langchain_core.documents.base import Document

from . import ingestion
from . import search
from . import generate
from .operators import BasicPipeline
from .store import delete_collection, delete_documents, get_documents, update_documents


async def main() -> None:
    try:
        logging.config.dictConfig(yaml.safe_load(open("./learn2rag/pipeline/logging.yaml").read()))
    except FileNotFoundError:
        logging.basicConfig()

    from .config import user_config, opt_config

    with open("loaded_documents.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    documents = [
        Document(page_content=d["content"], metadata=d["metadata"])
        for d in raw
    ]
    ingestion.index(documents, user_config, opt_config)

    if opt_config["query_mode"] == "multi":
        # in query_mode 'multi' different querys for each vector in the multi-vector are allowed
        multi_query = {"content": "What is USM AI?", "title": "What is USM AI?", "summary": "What is USM AI?", "source_path":"USU/ITSM/"}
        results = search.search_multi(multi_query, user_config, opt_config)
        points = results.points
        # modify the query for generation part
        query = " ".join(f"{k}={v}" for k, v in multi_query.items())
        answer = generate.generate(query, points, opt_config)
    else:
        pipeline = BasicPipeline()
        query = "Was sind A, B und C?"
        answer, points = itemgetter('answer', 'documents')(await pipeline(
            inputs={'question': query, 'user': 'anonymous'},
        ))

    sources = "\n".join(set(point.payload['source'] for point in points)) # type: ignore[index]

    for point in points:
        print(f"ID: {point.id}, Path: {point.payload['source']}, Score: {point.score}") # type: ignore[index]

    print(query)
    print(answer)
    print(sources)

if __name__ == "__main__":
    asyncio.run(main())

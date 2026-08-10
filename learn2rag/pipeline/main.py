import asyncio
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

    #delete_collection(loader_id="local_docs", user_config=user_config, opt_config=opt_config)
    #results = get_documents(loader_id="local_docs", user_config=user_config, opt_config=opt_config)

    documents = [
        Document(page_content=d["content"], metadata=d["metadata"])
        for d in [
            {
                "metadata": {
                    "source": "C:C:\\Users\\foo\\Revised Manuscript_Text categorization approach.docx",
                    "content_hash": "e18e509d138cf86c22df0b0dfafc5ca5b8f1e266f5e3470de68190f3ebe495b0",
                    "source_path": "C:\\Users\\foo",
                    "file_extension": "docx",
                    "process_date": "2025-07-28",
                    "process_time": "14:42:02",
                    "loader_type": "DirectoryLoader",
                    "loader_id": "local_docs",
                    "title": "The title of a real document",
                    "summary": "This document is awesome"
                },
                "content": "A brand-new Corpus-based Real-time Text Classification and Tagging Approach for Social Data..."
            },
            {
                "metadata": {
                    "source": "C:C:\\Users\\foo\\qdrant.docx",
                    "content_hash": "7f3b9c1a0d4e6f8b2c5a7d9e1f0b3c6d8a4e2f1c9b7d0a6e5f1c3a8b9d2e4f0",
                    "source_path": "C:\\Users\\foo",
                    "file_extension": "docx",
                    "process_date": "2025-07-28",
                    "process_time": "14:42:02",
                    "loader_type": "DirectoryLoader",
                    "loader_id": "local_docs",
                    "title": "The title of a real document",
                    "summary": "This document is awesome"
                },
                "content": "Qdrant ist eine Open-Source-Vektordatenbank..."
            },
        ]
]
    #update_documents(loader_id="local_docs", documents=documents, user_config=user_config, opt_config=opt_config)
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

    sources = "\n".join(set(point.payload['path'] for point in points)) # type: ignore[index]

    for point in points:
        print(f"ID: {point.id}, Path: {point.payload['source']}, Score: {point.score}") # type: ignore[index]

    print(query)
    print(answer)
    print(sources)

if __name__ == "__main__":
    asyncio.run(main())

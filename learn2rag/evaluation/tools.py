import collections
import csv
import hashlib
import pathlib
import json
import logging
import datasets
import json_stream  # type: ignore[import-untyped]
from typing import Any, Callable

import learn2rag.pipeline.ingestion
from learn2rag.pipeline.config import opt_config


def key_document(key: str) -> Callable[[Any], Any]:
    def document(example: Any) -> Any:
        yield example[key]
    return document


def key_document_list(key: str) -> Callable[[Any], Any]:
    def document(example: Any) -> Any:
        for item in example[key]:
            yield item
    return document


def hotpot_documents(example: dict[str, Any]) -> Any:
    context = example['context']
    for title, sentences in zip(context['title'], context['sentences']):
        yield ' '.join([title] + sentences)


@json_stream.streamable_list  # type: ignore[untyped-decorator]
def generate_documents(dataset_dict: Any, id_key: str | None, content_getter: Callable[[Any], Any], dataset_name: str, content_counter: collections.Counter[str]) -> Any:
    for split, dataset in dataset_dict.items():
        for i, example in enumerate(dataset):
            for content in content_getter(example):
                content_counter[content] += 1
                if content_counter[content] == 1:  # don't include duplicate content
                    yield {
                        'content': content,
                        'metadata': {
                            'content_hash': hashlib.sha256(content.encode('utf-8')).hexdigest(),
                            'loader_type': 'Evaluation',
                            'source': f'{dataset_name}/{split}/{example[id_key] if id_key is not None else i}',
                        },
                    }


def import_dataset_documents(dataset_name: str, subdirectory: str, id_key: str | None, content_getter: Callable[[Any], Any]) -> None:
    logging.debug(f'{dataset_name=}')
    dataset_work_dir = pathlib.Path('./datasets') / dataset_name
    dataset_dict = datasets.load_from_disk(dataset_work_dir / 'source' / subdirectory)
    content_counter = collections.Counter[str]()
    with (dataset_work_dir / 'loaded_documents.json').open('w') as f:
        json.dump(generate_documents(dataset_dict, id_key, content_getter, dataset_name, content_counter), f)
    questions_per_document_count = collections.Counter(sorted(content_counter.values(), reverse=True))
    logging.debug(f'{questions_per_document_count=}')
    with (dataset_work_dir / 'questions_per_document_count.csv').open('w') as f:
        wr = csv.writer(f)
        for item in questions_per_document_count.items():
            wr.writerow(item)


def ingest_dataset_documents(dataset_name: str) -> None:
    logging.debug(f'{dataset_name=}')
    dataset_work_dir = pathlib.Path('./datasets') / dataset_name
    documents_path = dataset_work_dir / 'loaded_documents.json'
    assert documents_path.is_file()
    user_config = {
        'file_path': None,
        'collection_name': dataset_name,
        'imported_documents_file_path': documents_path,
        'llm': None,
    }
    raise NotImplementedError()
    # FIXME
    # learn2rag.pipeline.ingestion.index(user_config, opt_config)


def read_dataset_qa(dataset_name: str, subdirectory: str, split: str | None=None) -> Any:
    logging.debug(f'{dataset_name=}')
    dataset_work_dir = pathlib.Path('./datasets') / dataset_name
    dataset_dict = datasets.load_from_disk(dataset_work_dir / 'source' / subdirectory)
    return dataset_dict[split] if split is not None else dataset_dict


def basic_pipeline(dataset_name: str, question: str) -> dict[str, Any]:
    user_config = {
        'file_path': None,
        'collection_name': dataset_name,
        'imported_documents_file_path': None,
        'llm': None,
    }
    documents = learn2rag.pipeline.search.search(question, user_config, opt_config)
    answer = learn2rag.pipeline.generate.generate(question, documents.points, opt_config)
    return {
        'documents': documents,
        'answer': answer,
    }

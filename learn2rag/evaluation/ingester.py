import logging

from .tools import ingest_dataset


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, force=True)
    ingest_dataset_documents('rag-mini-bioasq')
    ingest_dataset_documents('repliqa')
    ingest_dataset_documents('hotpot_qa')

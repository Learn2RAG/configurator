import logging

from .tools import import_dataset, key_document, hotpot_documents


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, force=True)
    import_dataset_documents('rag-mini-bioasq', 'text-corpus', 'id', key_document('passage'))
    import_dataset_documents('repliqa', '', 'document_id', key_document('document_extracted'))
    import_dataset_documents('hotpot_qa', 'distractor', 'id', hotpot_documents)

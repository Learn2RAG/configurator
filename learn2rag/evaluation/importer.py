import logging

from .tools import import_dataset_documents, key_document, key_document_list, hotpot_documents


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, force=True)
    import_dataset_documents('rag-mini-bioasq', 'text-corpus', 'id', key_document('passage'))
    import_dataset_documents('repliqa', '', 'document_id', key_document('document_extracted'))
    import_dataset_documents('hotpot_qa', 'distractor', 'id', hotpot_documents)
    import_dataset_documents('WikiEval', '', None, key_document_list('context_v2'))

import logging
import os
import sys

import main

logging.basicConfig(level=logging.DEBUG)

main.vector_store = main.RedisVectorStore(main.embeddings, os.environ.get('REDIS_INDEX_NAME'))
import_dirs = os.environ.get('IMPORT_DIRS')
logging.info('Import dirs: %s', import_dirs)

logging.info('Loading documents...')
for import_dir in import_dirs.split('|'):
    main.load(import_dir)

logging.info('Indexing...')
main.index()

logging.info('Importer done')

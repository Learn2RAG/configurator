import os
import sys

import main

main.vector_store = main.RedisVectorStore(main.embeddings, os.environ.get('REDIS_INDEX_NAME'))
main.load(os.environ.get('IMPORT_DIR'))
main.index()
print('Importer done')
sys.exit(0)

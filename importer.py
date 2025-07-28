import os
import sys

import main

main.vector_store = main.RedisVectorStore(main.embeddings, os.environ.get('REDIS_INDEX_NAME'))
import_dirs = os.environ.get('IMPORT_DIRS')
print(f'{import_dirs=}')
for import_dir in import_dirs.split('|'):
    main.load(import_dir)

main.index()
print('Importer done')
sys.exit(0)

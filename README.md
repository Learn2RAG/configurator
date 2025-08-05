# Learn2RAG basic implementation

## Requirements

- Python 3.13.5
- poetry
- direnv
- Docker
- Ollama

To install the dependencies run:
```
poetry install
poetry shell
```

Configure options in ``.envrc``:
```
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_AUTH=""
```
You might need to run ``direnv allow`` in the same directory.

### Ollama (not tested by USU!)
```
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -xzvf ollama-linux-amd64.tgz
bin/ollama serve
```
```
curl http://localhost:11434/api/pull -d '{"model": "llama3.3:70b"}'
```

### Start the containers for qdrant and ??? by running
```
docker compose up -d
```

### Optional: get repliqa dataset for recall evaluation

```
python data/get_repliqa_pdfs.py
```




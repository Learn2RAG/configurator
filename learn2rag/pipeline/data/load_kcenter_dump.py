#%%
import json
from pathlib import Path
from bs4 import BeautifulSoup
import hashlib

root = Path("/home/large-file-storage/download-uknowit")
paths = list(root.rglob("raw-document.json"))
document_paths = "document.json"

loaded_documents = []

for doc in paths:
    with open(doc, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(doc.parent / document_paths, "r", encoding="utf-8") as f:
        document = json.load(f)

    html_content = raw.get("content", "")
    soup = BeautifulSoup(html_content, "html.parser")
    content = soup.get_text(separator=" ", strip=True)
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    new = {
        "metadata": {
            "source": raw.get("id"),
            "content_hash": content_hash,
            "source_path": raw['categories'][0]['branchText'],
            "file_extension": "",
            "process_date": "",
            "process_time": "",
            "loader_type": "KCenterFormatter",
            "loader_id": "my_kcenter_dump",
            "document_id": document.get("docId"),
            "title": raw.get("title"),
            "summary": raw.get("summary"),
            "keywords": document.get("keywords"),
            "uri": document["link"]["uri"]           
        },
        "content": content,
    }

    loaded_documents.append(new)

with open("loaded_documents_kcenter.json", "w", encoding="utf-8") as f:
    json.dump(loaded_documents, f, ensure_ascii=False, indent=2)
# %%

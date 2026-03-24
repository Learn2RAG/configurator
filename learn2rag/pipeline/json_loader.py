from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document


def json_loader(file_path: str) -> list[Document]:
    loader = JSONLoader(
        file_path,
        jq_schema=".[]",
        content_key="content",
        metadata_func=lambda record, meta: record.get("metadata", {}),
    )
    return loader.load()
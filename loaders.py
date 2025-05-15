import asyncio
from typing import Sequence

from bs4 import SoupStrainer  # type: ignore[attr-defined]

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document


def pdf_loader(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    docs = asyncio.run(loader.aload())
    return docs


def web_loader(web_path: Sequence[str]) -> list[Document]:
    bs4_strainer = SoupStrainer(class_=[
        'page__content',
        'post-content',
    ])
    loader = WebBaseLoader(
        web_path=web_path,
        bs_kwargs={'parse_only': bs4_strainer},
    )
    docs = loader.load()
    return docs

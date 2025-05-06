import bs4

from langchain_community.document_loaders import WebBaseLoader


def web_loader(urls):
    bs4_strainer = bs4.SoupStrainer(class_=[
        'page__content',
        'post-content',
    ])
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={'parse_only': bs4_strainer},
    )
    docs = loader.load()
    return docs

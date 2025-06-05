#!.venv/bin/python3
# https://python.langchain.com/docs/tutorials/rag/
import logging
import logging.config
import sys
import yaml

from langchain.chains.openai_functions.qa_with_structure import AnswerWithSources
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from components import llm, prompt, vector_store
import loaders


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: AnswerWithSources


class StateUpdate(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: AnswerWithSources


# Define application steps
def retrieve(state: State) -> StateUpdate:
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State) -> StateUpdate:
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({
        'question': state['question'],
        'context': docs_content,
    })
    structured_llm = llm.with_structured_output(AnswerWithSources)
    response = structured_llm.invoke(messages)
    return {'answer': response}  # type: ignore[typeddict-item]  # FIXME


def main() -> None:
    docs = []
    docs += loaders.web_loader([
        'https://www.paderborn.de/tourismus-kultur/sehenswuerdigkeiten/Hasenfenster_Sehensw.php',
        'https://www1.wdr.de/nachrichten/mehr-feldhasen-nrw-100.html',
        'https://learn2rag.de/',
        'https://hobbit-project.github.io/',
    ])
    docs += loaders.pdf_loader('tests/data/pdf/HOBBIT.pdf')
    docs += loaders.wikibooks_loader('tests/data/wikibooks/pages-articles.xml.bz2', limit=20)
    for doc in docs:
        assert len(doc.page_content) != 0, doc
    logging.debug('Documents loaded: %i', len(docs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(docs)
    logging.debug('Document splits: %i', len(all_splits))

    logging.debug('Indexing...')
    vector_store.add_documents(documents=all_splits)
    logging.debug('Indexing done')

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, 'retrieve')
    graph = graph_builder.compile()
    print(graph.get_graph().draw_ascii(), file=sys.stderr)

    def rag_query(question: str) -> None:
        logging.info('Question: %s', question)
        response = graph.invoke({'question': question})
        logging.info('Answer: %s', response['answer'].answer)
        logging.info('Sources: %s', response['answer'].sources)

    # web
    rag_query('Wie viele Hasen auf dem Fenster zu sehen sind')
    rag_query('Warum gibt es in NRW so viele Hasen')
    # no corresponding sources
    rag_query('Wie viele Hasen leben auf der Welt')
    # PDF
    rag_query('Was ist HOBBIT')
    rag_query('Was ist IGUANA')
    # web
    rag_query('Was ist das Ziel von Projekt learn2rag')
    rag_query('Wer ist an learn2rag beteiligt')
    # Wikibooks
    rag_query('Warum 1 keine Primzahl ist')


if __name__ == '__main__':
    logging.config.dictConfig(yaml.safe_load(open('logging.yaml').read()))
    main()

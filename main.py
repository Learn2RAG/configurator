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
        'https://learn2rag.de/',
        'https://hobbit-project.github.io/',
    ])
    docs += loaders.pdf_loader('tests/data/pdf/HOBBIT.pdf')
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
        logging.info('Answer: %s', response['answer'])

    rag_query('Wie viele Hasen gibt es in Paderborn')
    rag_query('Wie viele Hasen gibt es in Deutschland')
    rag_query('Wie viele Hasen gibt es auf der Welt')
    rag_query('Wie viele verschiedene Arten von Hasen gibt es')
    rag_query('Was ist ein hobbit')
    rag_query('Was ist das Ziel von Projekt learn2rag')
    rag_query('Wer ist an learn2rag beteiligt')


if __name__ == '__main__':
    logging.config.dictConfig(yaml.safe_load(open('logging.yaml').read()))
    main()

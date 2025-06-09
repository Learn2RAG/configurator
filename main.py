#!.venv/bin/python3
# https://python.langchain.com/docs/tutorials/rag/
from typing import Any
import argparse
import logging
import logging.config
import sys
import yaml

from langchain.chains.openai_functions.qa_with_structure import AnswerWithSources
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import cliff
import cliff.app
import cliff.command
import cliff.commandmanager

from components import embeddings, llm, prompt, InMemoryVectorStore, RedisVectorStore
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
    assert vector_store is not None, 'vector_store should be defined'
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


vector_store = None
docs: list[Document] = []
graph = None


def load() -> None:
    docs.extend(loaders.web_loader([
        'https://www.paderborn.de/tourismus-kultur/sehenswuerdigkeiten/Hasenfenster_Sehensw.php',
        'https://www1.wdr.de/nachrichten/mehr-feldhasen-nrw-100.html',
        'https://learn2rag.de/',
        'https://hobbit-project.github.io/',
    ]))
    docs.extend(loaders.pdf_loader('tests/data/pdf/HOBBIT.pdf'))
    docs.extend(loaders.wikibooks_loader('tests/data/wikibooks/pages-articles.xml.bz2', limit=2))
    docs.extend(loaders.html_loader('tests/data/html/AIAct.html'))
    for doc in docs:
        assert len(doc.page_content) != 0, doc
    logging.debug('Documents loaded: %i', len(docs))


def index() -> None:
    assert vector_store is not None, 'vector_store should be defined'

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(docs)
    logging.debug('Document splits: %i', len(all_splits))

    logging.debug('Indexing...')
    vector_store.add_documents(documents=all_splits)
    logging.debug('Indexing done')


def build() -> None:
    global graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, 'retrieve')
    graph = graph_builder.compile()
    print(graph.get_graph().draw_ascii(), file=sys.stderr)


def query(query: str) -> None:
    assert graph is not None, 'graph should be defined'
    logging.info('Question: %s', query)
    response = graph.invoke({'question': query})
    logging.info('Answer: %s', response['answer'].answer)
    logging.info('Sources: %s', response['answer'].sources)


def run_example_queries() -> None:
    # web
    query('Wie viele Hasen auf dem Fenster zu sehen sind')
    query('Warum gibt es in NRW so viele Hasen')
    # no corresponding sources
    query('Wie viele Hasen leben auf der Welt')
    # PDF
    query('Was ist HOBBIT')
    query('Was ist IGUANA')
    # web
    query('Was ist das Ziel von Projekt learn2rag')
    query('Wer ist an learn2rag beteiligt')
    # Wikibooks
    query('Warum 1 keine Primzahl ist')
    # AI Act
    query('Was besagt das AI-Gesetz über Hasen?')
    query('Was besagt das AI-Gesetz über Banken?')


class Use(cliff.command.Command):
    def get_parser(self, prog_name: str) -> cliff._argparse.ArgumentParser:
        parser = super().get_parser(prog_name)
        parser.add_argument('variable', help='Variable to configure')
        parser.add_argument('constructor', help='Class name or function')
        return parser

    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        global vector_store
        if parsed_args.variable == 'vector_store':
            vector_store = globals()[parsed_args.constructor](embeddings)
        else:
            raise NotImplementedError


class Load(cliff.command.Command):
    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        load()


class Index(cliff.command.Command):
    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        index()


class Build(cliff.command.Command):
    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        build()


class Query(cliff.command.Command):
    def get_parser(self, prog_name: str) -> cliff._argparse.ArgumentParser:
        parser = super().get_parser(prog_name)
        parser.add_argument('query', help='Input query text')
        return parser

    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        query(parsed_args.query)


class Example(cliff.command.Command):
    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        run_example_queries()


class Run(cliff.command.Command):
    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        global vector_store
        load()
        vector_store = InMemoryVectorStore(embeddings)
        index()
        build()
        run_example_queries()


class App(cliff.app.App):
    def __init__(self) -> None:
        super().__init__(
            description='Learn2RAG',
            version='0.1',
            command_manager=cliff.commandmanager.CommandManager('learn2rag'),
            deferred_help=True,
        )

    def initialize_app(self, argv: list[str]) -> None:
        for command in [
                Use,
                Load,
                Index,
                Build,
                Query,
                Example,
                Run,
        ]:
            self.command_manager.add_command(command.__name__.lower(), command)  # type:ignore[type-abstract]


if __name__ == '__main__':
    logging.config.dictConfig(yaml.safe_load(open('logging.yaml').read()))
    sys.exit(App().run(sys.argv[1:]))

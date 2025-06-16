#!.venv/bin/python3
# https://python.langchain.com/docs/tutorials/rag/
from typing import Any
import argparse
import csv
import logging
import logging.config
import pathlib
import sys
import yaml

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cliff
import cliff.app
import cliff.command
import cliff.commandmanager
import cliff.lister

from components import embeddings, llm, load_prompt, InMemoryVectorStore, RedisVectorStore
import pipeline
import loaders


vector_store = None
docs: list[Document] = []
graph = None
prompt = None


def load(path: str) -> None:
    logging.debug('Loading: %s', path)
    loaded_docs = []
    if path.startswith('https://'):
        loaded_docs = loaders.web_loader(path)
    elif path.endswith('.html'):
        loaded_docs = loaders.html_loader(path)
    elif path.endswith('.pdf'):
        loaded_docs = loaders.pdf_loader(path)
    else:
        p = pathlib.Path(path)
        if p.is_dir():
            for f in p.iterdir():
                load(str(f))
        else:
            raise ValueError('Unsupported path', path)
    docs.extend(loaded_docs)


def index() -> None:
    assert vector_store is not None, 'vector_store should be defined'

    for doc in docs:
        assert len(doc.page_content) != 0, doc

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
    assert prompt is not None, 'prompt should be defined'
    assert vector_store is not None, 'vector_store should be defined'
    graph = pipeline.pipeline(
        vector_store=vector_store,
        llm=llm,
        prompt=prompt,
    )
    print(graph.get_graph().draw_ascii(), file=sys.stderr)


def query(query: str) -> pipeline.State:
    assert graph is not None, 'graph should be defined'
    return graph.invoke({'question': query})


def queries_from_file(input_path: str) -> list[pipeline.State]:
    with open(input_path, newline='') as input_object:
        return [query(row['query']) for row in csv.DictReader(input_object)]


class Use(cliff.command.Command):
    def get_parser(self, prog_name: str) -> cliff._argparse.ArgumentParser:
        parser = super().get_parser(prog_name)
        parser.add_argument('variable', help='Variable to configure')
        parser.add_argument('constructor', help='Class name or function')
        parser.add_argument('constructor_args', nargs='*', help='Arguments for the constructor')
        return parser

    def take_action(self, args: argparse.Namespace) -> Any:
        global vector_store
        constructor_args = args.constructor_args
        if args.variable == 'vector_store':
            constructor_args.insert(0, embeddings)
        globals()[args.variable] = globals()[args.constructor](*constructor_args)


class Load(cliff.command.Command):
    def get_parser(self, prog_name: str) -> cliff._argparse.ArgumentParser:
        parser = super().get_parser(prog_name)
        parser.add_argument('path', type=str, help='Path to index')
        return parser

    def take_action(self, args: argparse.Namespace) -> Any:
        load(args.path)


class Index(cliff.command.Command):
    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        index()


class Build(cliff.command.Command):
    def take_action(self, parsed_args: argparse.Namespace) -> Any:
        build()


class Query(cliff.lister.Lister):
    def get_parser(self, prog_name: str) -> cliff._argparse.ArgumentParser:
        parser = super().get_parser(prog_name)
        parser.add_argument('--input-file', type=str, help='Input csv file')
        parser.add_argument('--output-file', type=str, help='Output csv file')
        parser.add_argument('query', nargs='?', help='Input query text')
        return parser

    def take_action(self, args: argparse.Namespace) -> Any:
        if args.input_file is not None:
            items = queries_from_file(args.input_file)
        else:
            items = [query(args.query)]
        output_fields = 'query', 'response', 'sources'
        output_rows = [(item['question'], item['answer'].answer, item['answer'].sources) for item in items]
        if args.output_file is not None:
            with open(args.output_file, 'w', newline='') as output_object:
                csv.writer(output_object).writerows([output_fields] + output_rows)
        return (output_fields, output_rows)


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
        ]:
            self.command_manager.add_command(command.__name__.lower(), command)  # type:ignore[type-abstract]


if __name__ == '__main__':
    logging.config.dictConfig(yaml.safe_load(open('logging.yaml').read()))
    sys.exit(App().run(sys.argv[1:]))

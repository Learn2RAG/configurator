from os import PathLike
from pathlib import Path
from typing import Sequence

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from flask import Flask, send_from_directory
from jinja2 import BaseLoader, ChoiceLoader, FileSystemLoader

static_path = '/learn2rag.bootstrap.static'
static_name = 'learn2rag_bootstrap_static'


def setup_fastapi(app: FastAPI, template_directory: Sequence[str | PathLike[str]]) -> Sequence[str | PathLike[str]]:
    app.mount(static_path, StaticFiles(directory=Path(__file__).parent / 'static'), name=static_name)
    return [
        *template_directory,
        Path(__file__).parent.parent / 'bootstrap' / 'templates',
    ]


def setup_flask(app: Flask) -> None:
    app.add_url_rule(static_path + '/<path:path>', endpoint=static_name, view_func=lambda path: send_from_directory(Path(__file__).parent / 'static', path))
    loader: BaseLoader = FileSystemLoader(Path(__file__).parent / 'templates')
    if app.jinja_loader is not None:
        loader = ChoiceLoader([
            app.jinja_loader,
            loader,
        ])
    app.jinja_loader = loader

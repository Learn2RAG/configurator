import argparse

import uvicorn

from .app import build_app


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument('--host', type=str, default='127.0.0.1')
        self.add_argument('--port', type=int, default=9001)


def main(args: argparse.Namespace) -> None:
    # TODO use uvicorn.config.Config
    app = build_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level='debug',
    )

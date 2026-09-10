import argparse

import uvicorn

from .app import build_app


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument('--host', type=str, default='127.0.0.1')
        self.add_argument('--port', type=int, default=9001)
        self.add_argument('--tls-certfile', default=None)
        self.add_argument('--tls-keyfile', default=None)
        self.add_argument('--basic-username', type=str, default='')
        self.add_argument('--basic-password', type=str, default='')


def main(args: argparse.Namespace) -> None:
    # TODO use uvicorn.config.Config
    app = build_app(
        basic_username=args.basic_username,
        basic_password=args.basic_password,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.tls_certfile,
        ssl_keyfile=args.tls_keyfile,
        log_level='debug',
    )

import argparse
import importlib
import importlib.resources
import logging
import logging.config
import os
import pathlib
import sys
from types import TracebackType
from typing import Any, Unpack

import yaml


class LauncherArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument('module', type=str, nargs='?', default='learn2rag.ui')
        self.add_argument('--logging-config', type=pathlib.Path)


def excepthook(*exc_info: Unpack[tuple[type[BaseException], BaseException, TracebackType | None]]) -> None:
    os.environ['NO_COLOR'] = '1'
    logging.critical('Uncaught exception', exc_info=exc_info)
    sys.__excepthook__(*exc_info)


def configure_logging(config_path: pathlib.Path, debug: bool) -> None:
    if config_path is None:
        if not debug:
            config_path = importlib.resources.files("learn2rag") / "logging.yml"
        else:
            config_path = importlib.resources.files("learn2rag") / "logging-debug.yml"
    if config_path is not None:
        with config_path.open(encoding='utf-8') as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO if not debug else logging.DEBUG)
        logging.info('Using basic logging config')


if __name__ == '__main__':
    sys.excepthook = excepthook

    config = {}
    try:
        config = yaml.safe_load(open('config.yml'))
    except FileNotFoundError:
        pass

    args, rest = LauncherArgumentParser().parse_known_args()
    module = importlib.import_module(args.module)
    configure_logging(args.logging_config, config.get('logging', {}).get('debug', False))
    logging.debug('Learn2RAG launcher starting: %s, %s', args, rest)
    # TODO
    if args.module == 'learn2rag.ollama_tool':
        # FIXME default config values
        module.main(rest, config=config.get('OLLAMA', {'port': 11434}))
    elif args.module == 'learn2rag.importer':
        module.main(module.ImporterArgumentParser().parse_args(rest))
    elif args.module == 'learn2rag.ui':
        module.main(config)
    else:
        module.main()

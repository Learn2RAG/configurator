import argparse
import importlib
import logging
import sys

import yaml


class LauncherArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('module', type=str, nargs='?', default='learn2rag.ui')


def excepthook(*exc_info: tuple) -> None:
    logging.critical('Uncaught exception', exc_info=exc_info)


if __name__ == '__main__':
    sys.excepthook = excepthook

    config = {}
    try:
        config = yaml.safe_load(open('config.yml'))
    except FileNotFoundError:
        pass

    args, rest = LauncherArgumentParser().parse_known_args()
    module = importlib.import_module(args.module)
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

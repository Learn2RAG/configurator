import argparse
import importlib
import importlib.resources
import logging
import logging.config
import os
import pathlib
import sys
from datetime import datetime, timedelta
from types import TracebackType
from typing import Unpack

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import TypeAdapter


class LauncherArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument('module', type=str, nargs='?', default='learn2rag.ui')
        self.add_argument('--logging-config', type=pathlib.Path)
        self.add_argument('--schedule-interval', type=TypeAdapter(timedelta).validate_python)


def excepthook(*exc_info: Unpack[tuple[type[BaseException], BaseException, TracebackType | None]]) -> None:
    os.environ['NO_COLOR'] = '1'
    logging.critical('Uncaught exception', exc_info=exc_info)
    # also print it since logging might be not configured properly
    print(f'Uncaught exception: {exc_info}')
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
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        if len(sys.argv) == 1:
            print('You can create config.yml for more configuration options')
            print('https://docs.learn2rag.de/en/basic/administrator/#advanced-configuration')

    args, rest = LauncherArgumentParser().parse_known_args()
    configure_logging(args.logging_config, config.get('logging', {}).get('debug', False))
    logging.debug('Learn2RAG launcher starting: %s, %s', args, rest)
    module = importlib.import_module(args.module)
    # TODO
    module_args = tuple()
    module_kwargs = {}
    if args.module == 'learn2rag.ollama_tool':
        # FIXME default config values
        module_args = (
            rest,
        )
        module_kwargs = {'config': config.get('OLLAMA', {'port': 11434})}
    elif args.module == 'learn2rag.importer':
        module_args = (
            module.ImporterArgumentParser().parse_args(rest),
        )
    elif args.module == 'learn2rag.ui':
        module_args = (
            config,
        )

    if args.schedule_interval:
        scheduler = BlockingScheduler()
        trigger = IntervalTrigger(seconds=args.schedule_interval.total_seconds())
        scheduler.add_job(
            module.main,
            trigger,
            next_run_time=datetime.utcnow(),
            max_instances=1,
            args=module_args,
            kwargs=module_kwargs,
        )
        scheduler.start()
    else:
        module.main(*module_args, **module_kwargs)

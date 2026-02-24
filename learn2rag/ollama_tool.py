from typing import Any

import ollama


def do_list(args: list[str], *, ollama_client: ollama.Client) -> None:
    assert len(args) == 0
    print(ollama_client.list())


def do_pull(args: list[str], *, ollama_client: ollama.Client) -> None:
    assert len(args) == 1
    print(ollama_client.pull(args[0]))


def main(args: list[str], *, config: dict[str, Any]) -> None:
    ollama_client = ollama.Client(host='http://localhost:' + str(config['port']))
    if args[0:1] == ['list']:
        do_list(args[1:], ollama_client=ollama_client)
    elif args[0:1] == ['pull']:
        do_pull(args[1:], ollama_client=ollama_client)
    else:
        raise Exception(f'Arguments: {args}')

import platform
import subprocess
import sys
import os
from typing import Any

import uvicorn
import yaml


def webbrowser_open(url: str) -> None:
    try:
        if platform.system() == 'Windows':
            subprocess.Popen(['explorer', url])
        else:
            subprocess.Popen(['xdg-open', url])
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)


def start_ui(config: dict[str, Any]) -> None:
    from learn2rag.ui import create_app
    app = create_app(config=config)

    port = config.get('port', '9000')
    host = config.get('host', '0.0.0.0')

    ssl_key = config.get('ssl_keyfile')
    ssl_cert = config.get('ssl_certfile')

    use_https = False
    if ssl_key and ssl_cert:
        if os.path.exists(ssl_key) and os.path.exists(ssl_cert):
            print(f" SSL files defined and found at {ssl_key} or {ssl_cert}")
            use_https = True
        else:
            print(f"Warning: SSL files defined but not found at {ssl_key} or {ssl_cert}")
    else:
        print(f"no SSL files provided then switch to HTTP mode")

    protocol = 'https' if use_https else 'http'
    url = f"{protocol}://localhost:{port}"
    webbrowser_open(url)
    print('*' * 40)
    print('Learn2RAG: ' + url)
    print('*' * 40)

    uvicorn_kwargs = {
        "app": app,
        "host": host,
        "port": int(port),
        "log_level": "info",
        "interface": "wsgi",
    }

    if use_https:
        uvicorn_kwargs["ssl_keyfile"] = ssl_key
        uvicorn_kwargs["ssl_certfile"] = ssl_cert

    uvicorn.run(**uvicorn_kwargs)


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, 'ui', 'config.yml')
    config = {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"No config file used !")
        pass

    if sys.argv[1:2] == ['ollama']:
        import learn2rag.ollama_tool as ollama_tool
        # FIXME default config values
        ollama_tool.main(sys.argv[2:], config=config.get('OLLAMA', {'port': 11434}))
    # TODO
    elif sys.argv[1:] == ['learn2rag.pipeline']:
        import learn2rag.pipeline as pipeline
        pipeline.main()
    elif sys.argv[1:2] == ['learn2rag.pipeline.importer']:
        import learn2rag.importer as importer
        importer.main(importer.ImporterArgumentParser().parse_args(sys.argv[2:]))
    elif sys.argv[1:] == ['learn2rag.pipeline.ingestion']:
        import learn2rag.pipeline.ingestion as ingestion
        ingestion.main()
    elif sys.argv[1:] == []:
        start_ui(config)
    else:
        raise Exception(f'Arguments: {sys.argv[1:]}')

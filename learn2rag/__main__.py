import platform
import subprocess
import sys

import uvicorn
import yaml


def webbrowser_open(url):
    try:
        if platform.system() == 'Windows':
            subprocess.Popen(['explorer', url])
        else:
            subprocess.Popen(['xdg-open', url])
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)


def start_ui(config):
    from learn2rag.ui import create_app
    app = create_app(config=config)

    port = config.get('port', '9000')

    url = 'http://localhost:' + port
    webbrowser_open(url)
    print('*' * 40)
    print('Learn2RAG: ' + url)
    print('*' * 40)

    uvicorn.run(
        app,
        interface='wsgi',
        host=config.get('host', '0.0.0.0'),
        port=int(port),
    )


if __name__ == '__main__':
    config = {}
    try:
        config = yaml.safe_load(open('config.yml'))
    except FileNotFoundError:
        pass

    if sys.argv[1:2] == ['ollama']:
        import learn2rag.ollama_tool as ollama_tool
        # FIXME default config values
        ollama_tool.main(sys.argv[2:], config=config.get('OLLAMA', {'port': 11434}))
    elif sys.argv[1:] == ['learn2rag.pipeline']:
        import learn2rag.pipeline as pipeline
        pipeline.main()
    elif sys.argv[1:] == ['learn2rag.pipeline.ingestion']:
        import learn2rag.pipeline.ingestion as ingestion
        ingestion.main()
    elif sys.argv[1:] == []:
        start_ui(config)
    else:
        raise Exception(f'Arguments: {sys.argv[1:]}')

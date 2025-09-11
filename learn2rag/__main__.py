import uvicorn
import yaml
import learn2rag.ui

config = {}
try:
    config = yaml.safe_load(open('config.yml'))
except FileNotFoundError:
    pass

app = learn2rag.ui.create_app(config=config)

uvicorn.run(
    app,
    interface='wsgi',
    host=config.get('host', '0.0.0.0'),
    port=int(config.get('port', 9000)),
)

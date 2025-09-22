from pathlib import Path
import atexit
import importlib
import logging
import os
import socket
import urllib

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_babel import Babel, gettext, ngettext, pgettext
import flask.logging
import jinja2
import ollama
import yaml

from learn2rag.compose import Project
import learn2rag.data

from datetime import datetime  # <-- ADD THIS


logging.getLogger().addHandler(flask.logging.default_handler)
logging.getLogger().setLevel(logging.DEBUG)


def start_project(name, template_file, storage_path, render_context = {}):
    logging.debug('UI starting project: %s', name)
    storage_path = storage_path.expanduser().absolute()
    logging.debug('Storage path: %s', storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    project_file = storage_path / 'compose.yml'

    template = jinja2.Template(template_file.read_text())
    project_file.write_text(template.render(render_context | {
        'learn2rag_path': Path('.').absolute(),
        'storage_path': storage_path,
    }))
    project = None
    if project := Project.get(name):
        assert not project.running
        project.remove()
    project = Project.create(project_file, name)
    assert project is not None, 'project should not be None'
    project.start()
    return project


def stop_project(name):
    project = Project.get(name)
    assert project is not None, 'project should not be None'
    project.stop()


def find_free_ports(n):
    ports = []
    sockets = []
    for i in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sockets.append(s)
        ports.append(s.getsockname()[1])
    for s in sockets:
        s.close()
    return ports


def merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value
    return destination


def create_app(config={}):
    # create and configure the app
    app = Flask(
        __name__,
        instance_path=config.get('flask', {}).get('instance_path'),
    )
    app.config.from_mapping(
        SECRET_KEY='dev',
        OLLAMA={'port': 11434},
        SUGGESTED_MODELS={},
    )
    packaged_config = yaml.safe_load((importlib.resources.files(__package__) / 'config.yml').open())
    app.logger.debug('Packaged config: %s', packaged_config)
    app.config.from_mapping(merge(config, packaged_config))

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    babel = Babel(app)

    def get_locale():
        translations = map(str, babel.list_translations())
        return request.accept_languages.best_match(translations)
    babel.init_app(app, locale_selector=get_locale)

    app.logger.info('create_app')
    app.logger.debug('cwd: %s', os.getcwd())
    app.logger.debug('root_path: %s', app.root_path)
    compose_template_path = Path(app.root_path) / app.template_folder / 'compose'
    app.pipelines_template_path = compose_template_path / 'pipelines'
    app.components_template_path = compose_template_path / 'components'
    app.pipeline_templates = {str(item.stem): yaml.safe_load(item.open()) for item in app.pipelines_template_path.glob('*.yml')}
    app.logger.debug('Loaded %i pipeline_templates: %s', len(app.pipeline_templates), list(app.pipeline_templates.keys()))

    @app.context_processor
    def inject_info():
        return {
            'compose_templates': app.pipeline_templates,
            'ollama_available': hasattr(app, 'ollama_client'),
        }

    @app.context_processor
    def inject_data():
        return {
            'suggested_models': app.config.get('SUGGESTED_MODELS'),
            'models': learn2rag.data.get_all(app.instance_path, 'models'),
            'sources': learn2rag.data.get_all(app.instance_path, 'sources'),
            'pipelines': learn2rag.data.get_all(app.instance_path, 'pipelines'),
        }

    @app.context_processor
    def inject_current_year():
        return {'current_year': datetime.now().year}

    # TODO: let the user configure the directory for ollama data before starting it?
    try:
        project = start_project('ollama', app.components_template_path / 'ollama.yml', Path(app.instance_path) / 'ollama', app.config['OLLAMA'])
        app.ollama_client = ollama.Client(host='http://localhost:' + str(app.config['OLLAMA']['port']))
    except Exception as e:
        app.logger.exception(e)
        app.logger.warning('Ollama is already running or failed to start')

    @app.get('/')
    def start():
        pipelines = learn2rag.data.get_all(app.instance_path, 'pipelines')
        projects = Project.get_all()
        running_pipelines = sum(1 for k, p in projects.items() if k in pipelines and p.running)

        return render_template(
            'start.html',
            running_pipelines=running_pipelines
        )

    @app.get('/components')
    def components():
        projects = Project.get_all()
        return render_template('components.html', projects=projects)

    @app.post('/components/<name>')
    def component_action(name):
        return False

    @app.get('/models')
    def models_list():
        ollama_models = []
        try:
            if hasattr(app, 'ollama_client'):
                ollama_models = app.ollama_client.list()
                app.logger.info('Ollama models: %s', ollama_models)
                if hasattr(ollama_models, "models"):
                    ollama_models = ollama_models.models
                else:
                    app.logger.warning('ollama list response has no models attribute')
        except Exception as e:
            app.logger.warning("Ollama not available: %s", e)

        return render_template(
            'models_list.html',
            ollama_models=ollama_models,
        )

    @app.post('/models')
    def model_create():
        ok = True
        label = request.form['label']
        model = request.form['model']
        api = request.form['api']
        if api == 'ChatOllama':
            url = request.form.get('url') or 'http://127.0.0.1:' + str(app.config['OLLAMA']['port']) + '/'
            # TODO setup tokens for locally running ollama
            token = request.form.get('token') or ''
            if request.form.get('ollama') == 'pull':
                # TODO download in background
                app.ollama_client.pull(model)
                flash(gettext('Downloaded a language model: %(model)s', model=model))
        elif api == 'ChatOpenAI':
            url = request.form['url']
            token = request.form['token']
        else:
            ok = False
            flash(gettext('API is not supported: %(api)s', api=api), 'error')
        if ok:
            learn2rag.data.create_entry(app.instance_path, 'models', {
                'label': label,
                'url': url,
                'token': token,
                'model': model,
                'api': api,
            })
            flash(pgettext('flash', 'Added a new language model configuration: %(label)s', label=label))
        return redirect(url_for('models_list'))

    @app.post('/models/<model>')
    def model_action(model):
        if request.form['action'] == 'delete':
            learn2rag.data.delete_entry(app.instance_path, 'models', model)
        return redirect(url_for('models_list'))

    @app.get('/sources')
    def sources_list():
        return render_template('sources_list.html')

    @app.post('/sources')
    def source_create():
        learn2rag.data.create_entry(app.instance_path, 'sources', {
            'label': request.form['label'],
            'path': request.form['path'],
        })
        flash(pgettext('flash', 'Added a new data source configuration: %(label)s', label=label))
        return redirect(url_for('sources_list'))

    @app.post('/sources/<source>')
    def source_action(source):
        return 'Not implemented'

    @app.get('/pipelines')
    def pipelines_list():
        projects = Project.get_all()
        return render_template('pipelines_list.html', projects=projects)

    @app.post('/pipelines')
    def pipeline_create():
        label = request.form['label']
        ports = [int(port) for port in request.form.getlist("ports") if port]
        name = learn2rag.data.create_entry(app.instance_path, 'pipelines', {
            'label': label,
            'storage_path': request.form['storage_path'],
            'language_model': request.form['language_model'],
            'sources': request.form.getlist('sources'),
            'ports': ports,
        })
        flash(pgettext('flash', 'Added a new pipeline configuration: %(label)s', label=label))
        if request.form.get('import'):
            pipeline = learn2rag.data.get_entry(app.instance_path, 'pipelines', name)
            assert pipeline is not None
            start_pipeline(name, pipeline, 'import')
        return redirect(url_for('pipelines_list'))

    def start_pipeline(name, pipeline, template_name):
        url = urllib.parse.urlparse(request.base_url)

        sources = learn2rag.data.get_entries(app.instance_path, 'sources', pipeline['sources'])
        # expand "~" in paths
        for path_name, source in sources.items():
            source['path'] = str(Path(source['path']).expanduser().absolute())

        render_context = {
            'learn2rag_hostname': url.hostname,
            'pipeline': pipeline,
            'language_model': learn2rag.data.get_entry(app.instance_path, 'models', pipeline['language_model']),
            'sources': sources,
        }

        assert app.pipeline_templates[template_name]
        template_file = app.pipelines_template_path / (template_name + '.yml')

        with open(template_file) as f:
            content = yaml.safe_load(f)
            port_names = content.get('ports', [])
            configured_ports = pipeline.get('ports', [])
            ports = configured_ports + find_free_ports(len(port_names) - len(configured_ports))
            render_context['ports'] = dict(zip(port_names, ports))

        storage_path = Path(pipeline['storage_path'])

        try:
            project = start_project(name, template_file, storage_path, render_context)
            if project and project.running:
                flash(pgettext('flash', 'Started the pipeline'))
            else:
                flash(pgettext('flash', 'Failed to start the pipeline'), 'error')
        except Exception as e:
            app.logger.exception(e)
            app.logger.error('Could not start the pipeline')
            flash(pgettext('flash', 'Could not start the pipeline: %(message)s', message=e), 'error')

        # TODO "load" the corresponding Ollama model

    @app.post('/pipelines/<name>')
    def pipeline_action(name):
        pipeline = learn2rag.data.get_entry(app.instance_path, 'pipelines', name)
        if pipeline is None:
            flash(pgettext('flash', 'The requested pipeline is not found'), 'error')
        elif request.form['action'] == 'delete':
            return 'Not implemented'
        elif request.form['action'].startswith('start:'):
            start_pipeline(name, pipeline, request.form['action'].split(':', 2)[1])
        elif request.form['action'] == 'stop':
            try:
                stop_project(name)
                flash(pgettext('flash', 'Stopped the pipeline'))
            except Exception as e:
                app.logger.exception(e)
                app.logger.error('Could not stop the pipeline')
                flash(pgettext('flash', 'Could not stop the pipeline: %(message)s', message=e), 'error')
        return redirect(url_for('pipelines_list'))

    @app.get('/ps')
    def ps_list():
        projects = Project.get_all()
        return render_template('ps_list.html', projects=projects)

    app.logger.info('App creation complete')
    return app


def atexit_handler():
    logging.info('Stopping Ollama')
    project = Project.get('ollama')
    if project is not None:
        try:
            project.stop()
        except Exception as e:
            logging.error(e)
        try:
            project.remove()
        except Exception as e:
            logging.error(e)


atexit.register(atexit_handler)

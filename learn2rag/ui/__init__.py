from pathlib import Path
import atexit
import logging
import os
import socket
import urllib

from flask import Flask, flash, redirect, render_template, request, url_for
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
    storage_path = storage_path.absolute()
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


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

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
        }

    @app.context_processor
    def inject_data():
        return {
            'models': learn2rag.data.get_all(app.instance_path, 'models'),
            'sources': learn2rag.data.get_all(app.instance_path, 'sources'),
            'pipelines': learn2rag.data.get_all(app.instance_path, 'pipelines'),
        }

    @app.context_processor
    def inject_current_year():
        return {'current_year': datetime.now().year}

    # TODO: let the user configure the directory for ollama data before starting it?
    try:
        project = start_project('ollama', app.components_template_path / 'ollama.yml', Path(app.instance_path) / 'ollama')
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
        ollama_available = False
        ollama_models = []
        try:
            if hasattr(ollama, "list"):
                ollama_models = ollama.list()
                ollama_available = True
                app.logger.info('Ollama models: %s', ollama_models)
                if hasattr(ollama_models, "models"):
                    ollama_models = ollama_models.models
                else:
                    app.logger.warning('ollama list response has no models attribute')
        except Exception as e:
            app.logger.warning("Ollama not available: %s", e)

        return render_template(
            'models_list.html',
            ollama_available=ollama_available,
            ollama_models=ollama_models,
        )

    @app.post('/models')
    def model_create():
        label = request.form['label']
        model = request.form['model']
        if 'ollama' in request.form:
            api = 'ChatOllama'
            # TODO allow to customize Ollama port?
            url = 'http://127.0.0.1:11434/'
            # TODO setup tokens
            token = ''
            if request.form['ollama'] == 'pull':
                # TODO download in background
                ollama.pull(model)
                flash('Model downloaded')
        else:
            api = 'ChatOpenAI'
            url = request.form['url']
            token = request.form['token']
        learn2rag.data.create_entry(app.instance_path, 'models', {
            'label': label,
            'url': url,
            'token': token,
            'model': model,
            'api': api,
        })
        flash('New model added')
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
        ports = [int(port) for port in request.form.getlist("ports") if port]
        learn2rag.data.create_entry(app.instance_path, 'pipelines', {
            'label': request.form['label'],
            'storage_path': request.form['storage_path'],
            'language_model': request.form['language_model'],
            'sources': request.form.getlist('sources'),
            'ports': ports,
        })
        return redirect(url_for('pipelines_list'))

    @app.post('/pipelines/<name>')
    def pipeline_action(name):
        pipeline = learn2rag.data.get_entry(app.instance_path, 'pipelines', name)
        if pipeline is None:
            flash('Pipeline not found', 'error')
        elif request.form['action'] == 'delete':
            return 'Not implemented'
        elif request.form['action'].startswith('start:'):
            url = urllib.parse.urlparse(request.base_url)
            render_context = {
                'learn2rag_hostname': url.hostname,
                'pipeline': pipeline,
                'language_model': learn2rag.data.get_entry(app.instance_path, 'models', pipeline['language_model']),
                'sources': learn2rag.data.get_entries(app.instance_path, 'sources', pipeline['sources']),
            }

            app.logger.debug('Starting: %s', name)
            template_name = request.form['action'].split(':', 2)[1]
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
                    flash('Pipeline started')
                else:
                    flash('Pipeline failed to start', 'error')
            except Exception as e:
                app.logger.exception(e)
                app.logger.error('Could not start the pipeline')
                flash(f'Could not start the pipeline: {e}', 'error')

            # "load" the corresponding Ollama model
        elif request.form['action'] == 'stop':
            try:
                stop_project(name)
                flash('Pipeline stopped')
            except Exception as e:
                app.logger.exception(e)
                app.logger.error('Could not stop the pipeline')
                flash(f'Could not stop the pipeline: {e}', 'error')
        return redirect(url_for('pipelines_list'))

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

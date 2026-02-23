from pathlib import Path
import atexit
import importlib
import logging
import math
import os
import platform
import xdg.BaseDirectory
import shutil
import signal
import socket
import threading
import time
import urllib

from flask import Flask, flash, redirect as flask_redirect, render_template, request, make_response, url_for
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

# for now hardcode here , we can change it
DEFAULT_PIPELINE_PORTS = [9001, 9002, 9003, 9004, 9005]

def expand_path(path):
    return Path(path).expanduser().absolute()


def redirect(url):
    if 'HX-Boosted' in request.headers:
        response = make_response('', 204)
        response.headers['HX-Redirect'] = url
    else:
        response = flask_redirect(url)
    return response


def start_project(name, template_file, storage_path, render_context={}):
    logging.debug('UI starting project: %s', name)
    storage_path = expand_path(storage_path)
    logging.debug('Storage path: %s', storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    project_file = storage_path / 'compose.yml'

    template = jinja2.Template(template_file.read_text())
    project_file.write_text(template.render(render_context | {
        'is_windows': platform.system() == 'Windows',
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

def find_free_ports(n, preferred_ports=None):
    """
    Finds n free ports. Prioritizes preferred_ports if provided.
    """
    ports = []
    preferred_ports = preferred_ports or []

    # 1. Try preferred ports first
    for p in preferred_ports:
        if len(ports) >= n:
            break
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Set REUSEADDR to handle ports in TIME_WAIT state
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', p))
                ports.append(p)
        except OSError:
            continue  # Port is taken, skip to next or fallback

    # 2. Fallback to OS-assigned random ports if we still need more
    remaining = n - len(ports)
    if remaining > 0:
        temp_sockets = []
        for _ in range(remaining):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            ports.append(s.getsockname()[1])
            temp_sockets.append(s)

        for s in temp_sockets:
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
    default_instance_path = os.getenv('LOCALAPPDATA') + '/Learn2RAG/instance' if platform.system() == 'Windows' else xdg.BaseDirectory.save_data_path('Learn2RAG/instance')
    example_local_path = r'C:\Users\User\Documents' if platform.system() == 'Windows' else '/home/user/Documents'
    app = Flask(
        __name__,
        instance_path=config.get('flask', {}).get('instance_path', default_instance_path),
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
            'default_storage_prefix': app.instance_path + '/storage/',
            'firststeps_storage_path': app.instance_path + '/storage/example',
        }

    @app.context_processor
    def inject_data():
        suggested_models = app.config.get('SUGGESTED_MODELS')
        return {
            'suggested_models': suggested_models,
            'firststeps_model': suggested_models.get('gemma3_27b'),
            'models': learn2rag.data.get_all(app.instance_path, 'models'),
            'sources': learn2rag.data.get_all(app.instance_path, 'sources'),
            'pipelines': learn2rag.data.get_all(app.instance_path, 'pipelines'),
        }

    @app.context_processor
    def inject_current_year():
        return {'current_year': datetime.now().year}

    atexit.register(atexit_handler)

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

    def list_ollama_models():
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
        return ollama_models

    @app.get('/models')
    def models_list():
        return render_template(
            'models_list.html',
            ollama_models=list_ollama_models(),
        )

    @app.post('/models')
    def model_create():
        ok = True
        model = request.form['model']
        api = request.form['api']
        if api == 'ChatOllama':
            url = request.form.get('url') or 'http://127.0.0.1:' + str(app.config['OLLAMA']['port']) + '/'
            # TODO setup tokens for locally running ollama
            token = request.form.get('token') or ''
            if request.form.get('ollama') == 'pull':
                if model.find(':') == -1:
                    model += ':latest'
                start_project('ollama_download', app.components_template_path / 'ollama-download.yml', Path(), {'model': model})
                return flask_redirect(url_for('model_pulling', model=model))
        elif api == 'ChatOpenAI':
            url = request.form['url']
            token = request.form['token']
        else:
            ok = False
            flash(gettext('API is not supported: %(api)s', api=api), 'error')
        if ok:
            label = request.form.get('label', model)
            learn2rag.data.create_entry(app.instance_path, 'models', {
                'label': label,
                'url': url,
                'token': token,
                'model': model,
                'api': api,
            })
            flash(pgettext('flash', 'Added a new language model configuration: %(label)s', label=label))
        return redirect(url_for('models_list'))

    @app.get('/models/download')
    def model_pulling():
        model = request.args['model']
        ollama_downloader = Project.get('ollama_download')
        if ollama_downloader is not None and not ollama_downloader.running:
            ollama_downloader.remove()
            ollama_models = list_ollama_models()
            if any(ollama_model.model == model for ollama_model in ollama_models):
                status = 'success'
                learn2rag.data.create_entry(app.instance_path, 'models', {
                    'label': model,
                    'url': 'http://127.0.0.1:' + str(app.config['OLLAMA']['port']) + '/',
                    'token': '',
                    'model': model,
                    'api': 'ChatOllama',
                })
                flash(pgettext('flash', 'Downloaded a language model: %(model)s', model=model))
                res = make_response(render_template('model_pulling_success.html'))
                res.headers['HX-Redirect'] = url_for('models_list')
                return res
            else:
                flash(pgettext('flash', 'Failed to download a language model: %(model)s', model=model), 'error')
                res = make_response(render_template('model_pulling_failure.html'))
                res.headers['HX-Redirect'] = url_for('models_list')
                return res
        elif ollama_downloader is None:
            raise Exception('Unexpected downloader state')
        return render_template(
            'model_pulling.html',
        )

    @app.post('/models/<name>')
    def model_action(name):
        model = learn2rag.data.get_entry(app.instance_path, 'models', name)
        if model is None:
            flash(pgettext('flash', 'The requested language model configuration is not found'), 'error')
        else:
            pipelines = learn2rag.data.get_all(app.instance_path, 'pipelines')
            if any(True for p in pipelines.values() if name == p['language_model']):
                flash(pgettext('flash', 'Some configured pipelines use this language model configuration, remove them first'), 'error')
            else:
                # TODO if the model is on a local Ollama instance, remove it from there as well
                learn2rag.data.delete_entry(app.instance_path, 'models', name)
                flash(pgettext('flash', 'Removed language model configuration: %(label)s', label=model['label']))
        return redirect(url_for('models_list'))

    @app.get('/sources')
    def sources_list():
        return render_template('sources_list.html', example_local_path=example_local_path)

    @app.post('/sources')
    def source_create():
        label = request.form['label']
        learn2rag.data.create_entry(app.instance_path, 'sources', {
            'label': label,
            'path': request.form['path'],
        })
        flash(pgettext('flash', 'Added a new data source configuration: %(label)s', label=label))
        return redirect(url_for('sources_list'))

    @app.post('/sources/<name>')
    def source_action(name):
        source = learn2rag.data.get_entry(app.instance_path, 'sources', name)
        if source is None:
            flash(pgettext('flash', 'The requested data source configuration is not found'), 'error')
        else:
            pipelines = learn2rag.data.get_all(app.instance_path, 'pipelines')
            if any(True for p in pipelines.values() if name in p['sources']):
                flash(pgettext('flash', 'Some configured pipelines use this data source configuration, remove them first'), 'error')
            else:
                learn2rag.data.delete_entry(app.instance_path, 'sources', name)
                flash(pgettext('flash', 'Removed data source configuration: %(label)s', label=source['label']))
        return redirect(url_for('sources_list'))

    @app.get('/pipelines')
    def pipelines_list():
        projects = Project.get_all()
        if request.headers.get('HX-Request'):
            return render_template(
                '_pipelines_list_table.html',
                projects=projects,
                current_timestamp=math.floor(time.time())
            )

        return render_template('pipelines_list.html', projects=projects, current_timestamp=math.floor(time.time()))

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
        for path_name, source in sources.items():
            source['path'] = str(expand_path(source['path']))

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

            ports = configured_ports + find_free_ports(len(port_names) - len(configured_ports), DEFAULT_PIPELINE_PORTS)
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
            ok = True
            try:
                storage_path = expand_path(pipeline['storage_path'])
                shutil.rmtree(storage_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                app.logger.error('Failed to remove directory: %s, %s', storage_path, e)
                flash(pgettext('flash', 'Failed to remove directory: %(path)s', path=storage_path), 'error')
                ok = False
            if ok:
                learn2rag.data.delete_entry(app.instance_path, 'pipelines', name)
                flash(pgettext('flash', 'Removed pipeline: %(label)s', label=pipeline['label']))
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

    @app.post('/ps/stop')
    def ps_stop():
        try:
            stop_project(request.form['name'])
            flash('Stopped')
        except Exception as e:
            flash(f'Failed: {e}', 'error')
        return redirect(url_for('ps_list'))

    @app.post('/shutdown')
    def shutdown_request():
        threading.Thread(target=shutdown).start()
        return pgettext('shutdown', 'Bye!')

    app.logger.info('App creation complete')
    return app


def atexit_handler():
    logging.debug('Exit handler')
    logging.info('Stopping Ollama...')
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
    logging.info('Done')


def shutdown():
    time.sleep(1)
    logging.debug('Shutdown...')
    atexit_handler()
    os.kill(os.getpid(), signal.SIGTERM)

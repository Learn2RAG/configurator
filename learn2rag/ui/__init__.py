from pathlib import Path
import atexit
import logging
import os
import socket
import urllib
import platform

from flask import Flask, flash, redirect, render_template, request, url_for
import flask.logging
import jinja2
import ollama
import yaml

from learn2rag.compose import Project
import learn2rag.data

from flask_babel import Babel

from datetime import datetime


def normalize_path(path: Path) -> str:
    return str(path).replace('\\', '/')


logging.getLogger().addHandler(flask.logging.default_handler)
logging.getLogger().setLevel(logging.DEBUG)


def start_project(name, template_file, storage_path, render_context={}):
    storage_path = storage_path.absolute()
    storage_path.mkdir(parents=True, exist_ok=True)
    project_file = storage_path / 'compose.yml'

    full_context = render_context | {
        'learn2rag_path': normalize_path(Path('.').absolute()),
        'storage_path': normalize_path(storage_path),
        'is_windows': platform.system() == "Windows",
        'qdrant_bin': "C:/qdrant/qdrant.exe"  # Inject native Qdrant for Windows
    }

    logger = logging.getLogger(__name__)
    logger.debug("Template context: %s", full_context)

    template = jinja2.Template(template_file.read_text())
    project_file.write_text(template.render(full_context))

    project = Project.get(name)
    if project:
        if project.running:
            project.stop()
            project.running = False
            project.save()
        project.remove()

    project = Project.create(project_file, name)
    assert project is not None, 'project should not be None'
    project.start()
    return project


def stop_project(name):
    project = Project.get(name)
    assert project is not None, 'project should not be None'
    project.stop()
    project.running = False
    project.save()


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
    ROOT_PATH = Path(__file__).parent

    app = Flask(
        __name__,
        instance_relative_config=True,
        static_folder=str(ROOT_PATH / "static"),
        template_folder=str(ROOT_PATH / "templates")
    )

    # Babel setup
    babel = Babel(app)

    def get_locale():
        translations = map(str, babel.list_translations())
        return request.accept_languages.best_match(translations)

    babel.init_app(app, locale_selector=get_locale)

    app.config.from_mapping(SECRET_KEY='dev')
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

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
    pipeline_template_path = Path(__file__).parent / "templates" / "compose" / "pipelines"

    # Parse Jinja2 .yml.j2 templates
    app.pipeline_templates = {}
    for item in pipeline_template_path.glob("*.yml.j2"):
        template = jinja2.Template(item.read_text())
        rendered = template.render({
            'learn2rag_path': normalize_path(Path('.').absolute()),
            'storage_path': normalize_path(Path('.').absolute()),
            'pipeline': {'storage_path': normalize_path(Path('.').absolute())},
            'language_model': {
                'model': 'llama2:latest',
                'api': 'ChatOllama',
                'url': 'http://127.0.0.1:11434/',
                'token': ''
            },
            'sources': [],
            'ports': {
                'ui': 52000,
                'qdrant_http': 52001,
                'pipeline': 52002,
                'open_webui_pipelines': 52003
            },
            'is_windows': platform.system() == "Windows"
        })
        app.pipeline_templates[item.stem + ".yml"] = yaml.safe_load(rendered)

    app.logger.debug(
        'Loaded %i pipeline_templates: %s',
        len(app.pipeline_templates),
        list(app.pipeline_templates.keys())
    )

    @app.context_processor
    def inject_current_year():
        return {'current_year': datetime.now().year}

    ollama_bin_path = Path(os.environ.get("OLLAMA_PATH", "ollama"))
    app.logger.info("OLLAMA_PATH = %s (%s)", ollama_bin_path, platform.system())

    should_start_ollama = not (
        platform.system() == "Windows" and ollama_bin_path.suffix.lower() == ".exe"
    )

    if should_start_ollama:
        try:
            start_project(
                'ollama',
                app.components_template_path / 'ollama.yml',
                Path(app.instance_path) / 'ollama',
                {'ollama_bin': normalize_path(ollama_bin_path)}
            )
        except Exception as e:
            app.logger.exception(e)
            app.logger.warning('Ollama is already running or failed to start')
    else:
        app.logger.info("Skipping Ollama Docker startup on Windows (native binary detected)")

    # ---- Routes ----

    @app.get('/')
    def start():
        models = learn2rag.data.get_all(app.instance_path, 'models')
        sources = learn2rag.data.get_all(app.instance_path, 'sources')
        pipelines = learn2rag.data.get_all(app.instance_path, 'pipelines')
        projects = Project.get_all()
        running_pipelines = sum(1 for k, p in projects.items() if k in pipelines and p.running)
        return render_template(
            'start.html',
            models=models,
            sources=sources,
            pipelines=pipelines,
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
        models = learn2rag.data.get_all(app.instance_path, 'models')
        ollama_available, ollama_models = False, []
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
            models=models,
            ollama_available=ollama_available,
            ollama_models=ollama_models
        )

    @app.post('/models')
    def model_create():
        label = request.form['label']
        model = request.form['model']
        if 'ollama' in request.form:
            api, url, token = 'ChatOllama', 'http://127.0.0.1:11434/', ''
            if request.form['ollama'] == 'pull':
                ollama.pull(model)
                flash('Model downloaded')
        else:
            api = 'ChatOpenAI'
            url, token = request.form['url'], request.form['token']

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
        sources = learn2rag.data.get_all(app.instance_path, 'sources')
        return render_template('sources_list.html', sources=sources)

    @app.post('/sources')
    def source_create():
        learn2rag.data.create_entry(app.instance_path, 'sources', {
            'label': request.form['label'],
            'path': request.form['path'],
        })
        return redirect(url_for('sources_list'))

    @app.post('/sources/<source>')
    def source_action(source):
        if request.form['action'] == 'delete':
            try:
                learn2rag.data.delete_entry(app.instance_path, 'sources', source)
                flash(f"Source '{source}' removed successfully.")
            except Exception as e:
                app.logger.warning("Could not delete source %s: %s", source, e)
                flash(f"Failed to remove source '{source}': {e}", 'error')
        return redirect(url_for('sources_list'))

    @app.get('/pipelines')
    def pipelines_list():
        pipelines = learn2rag.data.get_all(app.instance_path, 'pipelines')
        language_models = learn2rag.data.get_all(app.instance_path, 'models')
        sources = learn2rag.data.get_all(app.instance_path, 'sources')
        projects = Project.get_all()
        return render_template(
            'pipelines_list.html',
            pipelines=pipelines,
            language_models=language_models,
            models=language_models,  # alias for template compatibility
            sources=sources,
            compose_templates=app.pipeline_templates,
            projects=projects
        )

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
        def escape_path(path):
            return str(path).replace('\\', '\\\\')

        pipeline = learn2rag.data.get_entry(app.instance_path, 'pipelines', name)
        if pipeline is None:
            flash('Pipeline not found', 'error')

        elif request.form['action'] == 'delete':
            try:
                project = Project.get(name)
                if project:
                    if project.running:
                        project.stop()
                    project.remove()
            except Exception as e:
                app.logger.warning("Could not stop/remove project %s: %s", name, e)
                flash(f"Could not stop/remove pipeline '{name}': {e}", 'error')

            try:
                learn2rag.data.delete_entry(app.instance_path, 'pipelines', name)
                flash(f"Pipeline '{name}' removed successfully.")
            except Exception as e:
                app.logger.warning("Could not delete pipeline entry %s: %s", name, e)
                flash(f"Failed to remove pipeline '{name}': {e}", 'error')

        elif request.form['action'].startswith('start:'):
            url = urllib.parse.urlparse(request.base_url)
            template_name = request.form['action'].split(':', 2)[1]
            if not template_name.endswith('.yml'):
                template_name += '.yml'

            assert app.pipeline_templates[template_name]
            template_base = template_name.replace('.yml', '').replace('.j2', '')
            template_file = app.pipelines_template_path / f"{template_base}.yml.j2"

            template = jinja2.Template(template_file.read_text())
            rendered = template.render({
                'pipeline': dict(pipeline, storage_path=escape_path(Path(pipeline['storage_path']))),
                'language_model': learn2rag.data.get_entry(app.instance_path, 'models', pipeline['language_model']),
                'sources': [
                    {
                        **learn2rag.data.get_entry(app.instance_path, 'sources', name),
                        'path': escape_path(Path(learn2rag.data.get_entry(app.instance_path, 'sources', name)['path']))
                    }
                    for name in pipeline['sources']
                ],
                'ports': {}
            })

            content = yaml.safe_load(rendered)
            port_names = content.get('ports', [])
            configured_ports = pipeline.get('ports', [])
            ports = configured_ports + find_free_ports(len(port_names) - len(configured_ports))

            storage_path = Path(pipeline['storage_path']).absolute()
            app.logger.debug("Pipeline config = %s", pipeline)

            render_context = {
                'learn2rag_hostname': url.hostname,
                'pipeline': dict(pipeline, storage_path=escape_path(Path(pipeline['storage_path']))),
                'language_model': learn2rag.data.get_entry(app.instance_path, 'models', pipeline['language_model']),
                'sources': [
                    {
                        **learn2rag.data.get_entry(app.instance_path, 'sources', name),
                        'path': escape_path(Path(learn2rag.data.get_entry(app.instance_path, 'sources', name)['path']))
                    }
                    for name in pipeline['sources']
                ],
                'ports': dict(zip(port_names, ports))
            }

            try:
                project = start_project(name, template_file, storage_path, render_context)
                if project and project.running:
                    flash('Pipeline started')
                else:
                    flash('Pipeline failed to start', 'error')
            except Exception as e:
                app.logger.exception(e)
                app.logger.error('Could not start the pipeline')
                flash(f"Could not start the pipeline: {e}", 'error')

        elif request.form['action'] == 'stop':
            try:
                stop_project(name)
                flash('Pipeline stopped')
            except Exception as e:
                app.logger.exception(e)
                app.logger.error('Could not stop the pipeline')
                flash(f"Could not stop the pipeline: {e}", 'error')

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

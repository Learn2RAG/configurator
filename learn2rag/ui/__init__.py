from pathlib import Path
import logging
import os
import urllib

from flask import Flask, flash, redirect, render_template, request, url_for
import flask.logging
import jinja2
import yaml

from learn2rag.compose import Project
import learn2rag.data


logging.getLogger().addHandler(flask.logging.default_handler)
logging.getLogger().setLevel(logging.DEBUG)


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

    app.logger.debug('cwd: %s', os.getcwd())
    app.logger.debug('root_path: %s', app.root_path)
    app.compose_template_path = Path(app.root_path) / app.template_folder / 'compose'
    app.compose_templates = {str(item.stem): yaml.safe_load(item.open()) for item in app.compose_template_path.glob('*.yml')}
    app.logger.debug('Loaded %i compose_templates: %s', len(app.compose_templates), list(app.compose_templates.keys()))

    @app.get('/')
    def start():
        return render_template('start.html')

    @app.get('/models')
    def models_list():
        models = learn2rag.data.get_all(app.instance_path, 'models')
        return render_template('models_list.html', models=models)

    @app.post('/models')
    def model_create():
        learn2rag.data.create_entry(app.instance_path, 'models', {
            'label': request.form['label'],
            'url': request.form['url'],
            'token': request.form['token'],
            'model': request.form['model'],
        })
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
        return 'Not implemented'

    @app.get('/pipelines')
    def pipelines_list():
        pipelines = learn2rag.data.get_all(app.instance_path, 'pipelines')

        # FIXME
        for pipeline in pipelines.values():
            url = urllib.parse.urlparse(request.base_url)
            pipeline['ui_url'] = url.scheme + '://' + url.hostname + ':5001'

        language_models = learn2rag.data.get_all(app.instance_path, 'models')
        sources = learn2rag.data.get_all(app.instance_path, 'sources')
        projects = Project.get_all()
        return render_template('pipelines_list.html', pipelines=pipelines, language_models=language_models, sources=sources, compose_templates=app.compose_templates, projects=projects)

    @app.post('/pipelines')
    def pipeline_create():
        learn2rag.data.create_entry(app.instance_path, 'pipelines', {
            'label': request.form['label'],
            'storage_path': request.form['storage_path'],
            'language_model': request.form['language_model'],
            'sources': request.form.getlist('sources'),
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
            app.logger.debug('Starting: %s', name)
            template_name = request.form['action'].split(':', 2)[1]
            assert app.compose_templates[template_name]
            template = jinja2.Template((app.compose_template_path / (template_name + '.yml')).read_text())
            language_model = learn2rag.data.get_entry(app.instance_path, 'models', pipeline['language_model'])
            sources = learn2rag.data.get_entries(app.instance_path, 'sources', pipeline['sources'])
            content = template.render(learn2rag_path=Path('.').absolute(), pipeline=pipeline, language_model=language_model, sources=sources)
            storage_path = Path(pipeline['storage_path']).absolute()
            app.logger.debug('Storage path: %s', storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            project_file = storage_path / 'pipeline.yml'
            project_file.write_text(content)
            try:
                project = None
                if project := Project.get(name):
                    assert not project.running
                    project.remove()
                project = Project.create(project_file, name)
                assert project is not None, 'project should not be None'
                project.start()
                flash('Pipeline started')
            except Exception as e:
                app.logger.error('Could not start the pipeline: %s', e)
                flash(f'Could not start the pipeline: {e}', 'error')
            if project and not project.running:
                flash('Pipeline failed to start', 'error')
        elif request.form['action'] == 'stop':
            project = Project.get(name)
            try:
                assert project is not None, 'project should not be None'
                project.stop()
                flash('Pipeline stopped')
            except Exception as e:
                app.logger.error('Could not stop the pipeline: %s', e)
                flash(f'Could not stop the pipeline: {e}', 'error')
        return redirect(url_for('pipelines_list'))

    return app

import os

from flask import Flask, redirect, render_template, request, url_for

import learn2rag.data


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
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

    @app.get('/')
    def start():
        return render_template('start.html')

    @app.get('/models')
    def models_list():
        models = learn2rag.data.get_entries(app.instance_path, 'models')
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
        sources = learn2rag.data.get_entries(app.instance_path, 'sources')
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
        pipelines = learn2rag.data.get_entries(app.instance_path, 'pipelines')
        language_models = learn2rag.data.get_entries(app.instance_path, 'models')
        sources = learn2rag.data.get_entries(app.instance_path, 'sources')
        return render_template('pipelines_list.html', pipelines=pipelines, language_models=language_models, sources=sources)

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
        if request.form['action'] == 'delete':
            return 'Not implemented'
        if request.form['action'] == 'build':
            pipeline = learn2rag.data.get_entry(app.instance_path, 'pipelines', name)
            return 'Not implemented'
        return redirect(url_for('pipelines_list'))

    return app

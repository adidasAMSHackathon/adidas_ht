import os
from flask import Flask, render_template
from adihack import db, running, api

def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'adiback.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.update(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    try:
        os.makedirs(app.config['UPLOAD_DIRECTORY'])
    except OSError:
        pass

    db.init_app(app)

    @app.route('/')
    def root():
        return render_template('index.html')

    app.register_blueprint(running.bp)
    app.register_blueprint(api.bp)

    return app

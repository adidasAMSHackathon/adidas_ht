import functools
import os
import json
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app as app
)
from werkzeug import secure_filename
import adihack.df_function as dfs

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/content', methods=['GET'])
def content():
    videos = [f for f in os.listdir(app.config['UPLOAD_DIRECTORY']) if os.path.isfile(os.path.join(app.config['UPLOAD_DIRECTORY'], f))]

    return '\n'.join(videos)

@bp.route('/content/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename[-3:] == 'mp4':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_DIRECTORY'], filename))
            return "ok"
    elif request.method == 'GET':
        return render_template('test_upload.html')

    return "err"

@bp.route('/process/<string:video>', methods=['GET'])
def process(video):
    path = os.path.join(app.config['UPLOAD_DIRECTORY'], video)
    analysis = dfs.walk_analysis("not used path because test", framerate = 24, fetched = True, path_csv = "/Users/svenhoelzel/Desktop/hackathon/sample.csv")
    res = analysis.create_walking_profile()

    with open(os.path.join(app.config['UPLOAD_DIRECTORY'], 'data.json'), 'w+') as f:
        f.write(json.dumps(res))
    return "ok"

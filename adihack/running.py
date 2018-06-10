import functools
import os
import json
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app as app
)

bp = Blueprint('running', __name__, url_prefix='/running')

@bp.route('/', methods=['GET'])
def index():
    data = None
    try:
        with open(os.path.join(app.config['UPLOAD_DIRECTORY'], 'data.json')) as f:
            data = json.load(f)
    except:
        pass
   
    if data:
        return render_template('running.html',
            analysis_chart_labels=str(data['walk_data']['raw_data']),
            analysis_chart_data=str(data['walk_data']['raw_data']),
            analysis_chart_data2=str(data['walk_data']['idealised_walk']),
            tread_stability=str(int(100*data['profiles']['distance_profile']['Movement_consistency']['std']/ data['profiles']['distance_profile']['Movement_consistency']['mean'])) + ' %',
            rom_consistency=str(int(100*data['profiles']['degree_profile']['Movement_consistency']['std']/ data['profiles']['degree_profile']['Movement_consistency']['mean'])) + ' %',
            step_width_consistency=str(int(100*data['profiles']['distance_profile']['Movement_pace']['std']/ data['profiles']['distance_profile']['Movement_pace']['mean'])) + ' %',
            cyclical_consistency=str(int(100*data['profiles']['degree_profile']['Movement_pace']['std']/ data['profiles']['degree_profile']['Movement_pace']['mean'])) + ' %')
    else:
        return render_template('running.html',
            analysis_chart_labels=[1, 2, 3, 4, 5],
            analysis_chart_data=[-4, 0, 4, 0, -4],        analysis_chart_data2=[0, -4, 0, 4, 0],
            tread_stability='No data',
            rom_consistency='No data',
            step_width_consistency='No data',
            cyclical_consistency='No data')
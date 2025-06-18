from flask import Blueprint, render_template, request
import logging
from web.config import CONFIG
from web.forecast_bp import sale_periods

import os
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
forecast_bp = Blueprint('forecast_bp', __name__, template_folder=template_dir)

@forecast_bp.route('/')
def home():
    try:
        markets = sorted(CONFIG.keys())
        selected_market = request.args.get('market', markets[0] if markets else '')
        fruits = sorted(CONFIG[selected_market].keys()) if selected_market in CONFIG else []
        selected_fruit = request.args.get('fruit', fruits[0] if fruits else '')
        varieties = sorted(CONFIG[selected_market][selected_fruit].keys()) if selected_fruit in CONFIG[selected_market] else []
        selected_variety = request.args.get('variety', varieties[0] if varieties else '')
        grades = sorted(CONFIG[selected_market][selected_fruit][selected_variety].keys()) if selected_variety in CONFIG[selected_market][selected_fruit] else []
        selected_grade = request.args.get('grade', grades[0] if grades else '')

        sale_periods_json = {"|".join(k): v for k, v in sale_periods.items()}

        return render_template('predict.html',
                               config=CONFIG,
                               sale_periods=sale_periods_json,
                               markets=markets,
                               fruits=fruits,
                               varieties=varieties,
                               grades=grades,
                               selected_market=selected_market,
                               selected_fruit=selected_fruit,
                               selected_variety=selected_variety,
                               selected_grade=selected_grade)

    except Exception as e:
        logging.exception("Error rendering forecast home:")
        return f"Template error: {str(e)}", 500

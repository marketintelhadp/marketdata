import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask
from web.forecast_bp import forecast_bp
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Flask
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Use secret key from .env
app.secret_key = os.getenv('SECRET_KEY', 'fallback_key')

app.register_blueprint(forecast_bp, url_prefix="/")

# Note: Removed the app.run() block to allow mounting in FastAPI

#!/bin/bash

# Activate the virtual environment (optional)
# source venv/bin/activate

# Navigate to the directory containing your Flask app (if needed)
# cd path_to_flask_app_directory

# Start the Flask app using Gunicorn
gunicorn app:app -w 4 -b 0.0.0.0:8000


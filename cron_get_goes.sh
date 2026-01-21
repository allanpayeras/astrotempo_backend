#!/bin/bash
set -e  # exit if any command fails

echo "=== Starting script ==="
date

# Activate virtual environment
source /home/payeras/personal_projects/astro-weather/env/bin/activate

# Run scripts in order
cd /home/payeras/personal_projects/astro-weather/
python -u /home/payeras/personal_projects/astro-weather/download_goes.py
python -u /home/payeras/personal_projects/astro-weather/create_goes_frames.py

echo "Finished execution at"
date
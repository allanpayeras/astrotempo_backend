#!/bin/bash
set -e  # exit if any command fails

echo "=== Starting script ==="
date

# Activate virtual environment
source /home/payeras/personal_projects/astro-weather/env/bin/activate

# Run scripts in order
cd /home/payeras/personal_projects/astro-weather/
python -u /home/payeras/personal_projects/astro-weather/downloader_wrf.py
python -u /home/payeras/personal_projects/astro-weather/create_wrf_db.py

echo "Finished execution at"
date
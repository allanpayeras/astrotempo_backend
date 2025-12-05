#!/bin/bash
set -e  # exit if any command fails

echo "=== Starting script ==="
date
whoami
pwd

# Activate virtual environment
source /home/payeras/personal_projects/astro-weather/env/bin/activate

which python
python --version

# Run scripts in order
cd /home/payeras/personal_projects/astro-weather/
pwd
python -u /home/payeras/personal_projects/astro-weather/downloader_wrf.py
python -u /home/payeras/personal_projects/astro-weather/process_wrf_files.py
python -u /home/payeras/personal_projects/astro-weather/extract_data.py

echo "Finished execution at"
date
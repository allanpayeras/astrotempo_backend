import os
import asyncio
import aiohttp
import requests

from pathlib import Path
from datetime import datetime, timezone, timedelta


# Configuration parameters
MAX_CONCURRENT = 12  # number of parallel downloads
CHUNK_SIZE = 1024 * 64  # 64 KB chunks
RETRIES = 3  # retries per file
TIMEOUT = aiohttp.ClientTimeout(
    total=None,  # no limit on total download time
    sock_connect=30,  # how long to wait to connect
    sock_read=60,  # how long to wait for each chunk
)


# functions


# Download a single file
async def download_file(session, url, out_path):
    for attempt in range(1, RETRIES + 1):
        try:
            async with session.get(url, timeout=TIMEOUT) as resp:

                # File does not exist on server
                if resp.status == 404:
                    print(f"❌ Not found (404): {url}")
                    return

                # Other HTTP errors
                if resp.status != 200:
                    print(f"❌ HTTP {resp.status} for {url} (attempt {attempt})")
                    continue

                # Save file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    print(f"⬇️ Downloading: {out_path.name}")
                    async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)

                print(f"✅ Downloaded: {out_path.name}")
                return

        except asyncio.TimeoutError:
            print(f"⚠️ Timeout downloading {url} (attempt {attempt})")
        except Exception as e:
            print(f"⚠️ Error downloading {url} (attempt {attempt}): {e}")
            print(f"Removing incomplete file {out_path.name}")
            os.remove(out_path) if out_path.exists() else None

    print(f"❌ Failed after {RETRIES} attempts: {url}")


# Download many files async
async def download_many(urls, output_dir):

    output_dir = Path(output_dir)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for url in urls:
            filename = url.split("/")[-1]
            out_path = output_dir / filename

            # Skip if file already exists
            if out_path.exists():
                print(f"⏩ Skipped (exists): {filename}")
                continue

            tasks.append(download_file(session, url, out_path))

        if not tasks:
            print("All files already exist.")
            return

        await asyncio.gather(*tasks)


# get latest available forecast run hour (in UTC) for a given date
def get_latest_forecast_run_hour(date: datetime):
    latest = None
    for run_hour in range(24):
        url = (
            f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
            f"{date.year}/{date.month:02d}/{date.day:02d}/{run_hour:02d}/"
        )
        r = requests.head(url)
        if r.status_code in [200, 403]:
            latest = run_hour
    if latest is not None:
        return latest
    raise ValueError(
        f"No available forecast found for date {date.year}-{date.month:02d}-{date.day:02d}"
    )


async def update_forecast_data():
    utc_now = datetime.now(timezone.utc)
    try:
        latest_run_hour = get_latest_forecast_run_hour(utc_now)
        initial_date_time = datetime(
            utc_now.year, utc_now.month, utc_now.day, latest_run_hour
        )
    except ValueError as e:
        print(e)
        yesterday = utc_now - timedelta(days=1)
        print(
            f"Trying yesterday's date ({yesterday.year}-{yesterday.month:02d}-{yesterday.day:02d})"
        )
        latest_run_hour = get_latest_forecast_run_hour(yesterday)
        initial_date_time = datetime(
            yesterday.year, yesterday.month, yesterday.day, latest_run_hour
        )

    files_to_download = []
    for hour_delta in range(1, (24 * 7 + 1)):
        forecast_date_time = initial_date_time + timedelta(hours=hour_delta)

        url_path = (
            f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
            f"{initial_date_time.year}/{initial_date_time.month:02d}/{initial_date_time.day:02d}/{initial_date_time.hour:02d}/"
        )
        url_file_name = (
            f"WRF_cpt_07KM_{initial_date_time.year}{initial_date_time.month:02d}{initial_date_time.day:02d}{initial_date_time.hour:02d}_"
            f"{forecast_date_time.year}{forecast_date_time.month:02d}{forecast_date_time.day:02d}{forecast_date_time.hour:02d}.grib2"
        )
        url = url_path + url_file_name

        files_to_download.append(url)

    if os.path.exists("./data_tmp"):
        current_downloaded_files = os.listdir("./data_tmp")
    else:
        current_downloaded_files = []

    await download_many(files_to_download, "./data_tmp")

    # delete all 'old forecast files' in ./data_tmp
    for file in current_downloaded_files:
        full_path = os.path.join("./data_tmp", file)
        if full_path not in [
            "./data_tmp/" + url.split("/")[-1] for url in files_to_download
        ]:
            print("Removing old file:", full_path)
            os.remove(full_path)


if __name__ == "__main__":
    asyncio.run(update_forecast_data())

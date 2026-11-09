import os
import sys
import shutil
import asyncio
import aiohttp
import logging
import tempfile
import requests

from pathlib import Path
from datetime import datetime, timezone, timedelta

# logging configuration
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wrf_downloader")

# configuration parameters
MAX_CONCURRENT = 12  # number of parallel downloads
CHUNK_SIZE = 1024 * 64  # 64 KB chunks
RETRIES = 3  # retries per file
TIMEOUT = aiohttp.ClientTimeout(
    total=None,  # no limit on total download time
    sock_connect=30,  # how long to wait to connect
    sock_read=60,  # how long to wait for each chunk
)


async def download_file(session: aiohttp.ClientSession, url: str, out_path: Path):
    """Download a single file asynchronously with retry logic."""
    for attempt in range(1, RETRIES + 1):
        try:
            async with session.get(url, timeout=TIMEOUT) as resp:
                # File does not exist on server
                if resp.status == 404:
                    logger.warning("File not found on server (404): %s", url)
                    return

                # Other HTTP errors
                if resp.status != 200:
                    logger.warning(
                        "HTTP error status %d for %s (attempt %d/%d)",
                        resp.status,
                        url,
                        attempt,
                        RETRIES,
                    )
                    continue

                # Save file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug("Starting download stream: %s", out_path.name)

                with open(out_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)

                logger.info("Successfully downloaded: %s", out_path.name)
                return

        except asyncio.TimeoutError:
            logger.warning(
                "Timeout downloading %s (attempt %d/%d)", url, attempt, RETRIES
            )
        except Exception as e:
            logger.error(
                "Error downloading %s (attempt %d/%d): %s",
                url,
                attempt,
                RETRIES,
                e,
                exc_info=True,
            )
            if out_path.exists():
                logger.debug("Cleaning up partial file: %s", out_path.name)
                try:
                    os.remove(out_path)
                except OSError as cleanup_err:
                    logger.error(
                        "Failed to remove partial file %s: %s",
                        out_path.name,
                        cleanup_err,
                    )

    logger.error("Failed to download %s after %d attempts", url, RETRIES)


async def download_many(urls, output_dir):
    """Download multiple files concurrently using aiohttp."""
    output_dir = Path(output_dir)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            download_file(session, url, output_dir / url.split("/")[-1]) for url in urls
        ]

        if not tasks:
            logger.info("No tasks to execute.")
            return

        logger.info("Initiating concurrent downloads for %d files...", len(tasks))
        await asyncio.gather(*tasks)


def get_latest_forecast_run_hour(date: datetime) -> int:
    """Get latest available forecast run hour (in UTC) for a given date."""
    latest = None
    for run_hour in range(24):
        url = (
            f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
            f"{date.year}/{date.month:02d}/{date.day:02d}/{run_hour:02d}/"
        )
        try:
            r = requests.head(url, timeout=10)
            if r.status_code in [200, 403]:
                latest = run_hour
        except requests.RequestException as e:
            logger.debug("Head request failed for run hour %02d: %s", run_hour, e)

    if latest is not None:
        return latest

    raise ValueError(
        f"No available forecast found for date {date.year}-{date.month:02d}-{date.day:02d}"
    )


def get_latest_forecast_urls() -> list:
    """Determine the latest forecast run URLs based on current UTC time."""
    utc_now = datetime.now(timezone.utc)
    try:
        latest_run_hour = get_latest_forecast_run_hour(utc_now)
        initial_date_time = datetime(
            utc_now.year,
            utc_now.month,
            utc_now.day,
            latest_run_hour,
            tzinfo=timezone.utc,
        )
    except ValueError as e:
        logger.warning("%s. Falling back to yesterday's date.", e)
        yesterday = utc_now - timedelta(days=1)
        latest_run_hour = get_latest_forecast_run_hour(yesterday)
        initial_date_time = datetime(
            yesterday.year,
            yesterday.month,
            yesterday.day,
            latest_run_hour,
            tzinfo=timezone.utc,
        )

    logger.info("Using forecast initial reference time (UTC): %s", initial_date_time)

    urls = []
    # Example: 7 days of hourly forecast steps (24 hours * 7 days)
    for hour_delta in range(1, (24 * 7) + 1):
        forecast_date_time = initial_date_time + timedelta(hours=hour_delta)

        url_path = (
            f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
            f"{initial_date_time.year}/{initial_date_time.month:02d}/{initial_date_time.day:02d}/{initial_date_time.hour:02d}/"
        )
        url_file_name = (
            f"WRF_cpt_07KM_{initial_date_time.year}{initial_date_time.month:02d}{initial_date_time.day:02d}{initial_date_time.hour:02d}_"
            f"{forecast_date_time.year}{forecast_date_time.month:02d}{forecast_date_time.day:02d}{forecast_date_time.hour:02d}.grib2"
        )
        urls.append(url_path + url_file_name)

    return urls


def get_current_latest_files() -> list:
    """Retrieve list of currently stored latest files."""
    target_dir = Path("./latest_wrf_files")
    if target_dir.exists():
        return [f.name for f in target_dir.iterdir() if f.is_file()]
    return []


async def update_forecast_data():
    """Main pipeline routine to check, download, and stage latest forecast files."""
    logger.info("Starting forecast data update check...")

    urls = get_latest_forecast_urls()
    current_latest_files = get_current_latest_files()

    files_to_download = [
        url for url in urls if url.split("/")[-1] not in current_latest_files
    ]

    if not files_to_download:
        logger.info("All forecast files are already up-to-date. No downloads required.")
        return

    logger.info("Found %d new files to download.", len(files_to_download))

    with tempfile.TemporaryDirectory(prefix="wrf_download_", dir="./") as temp_dir_str:
        tmp_dir = Path(temp_dir_str)

        await download_many(files_to_download, tmp_dir)

        latest_dir = Path("./latest_wrf_files")
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Staging downloaded files to %s", latest_dir)
        for file_path in tmp_dir.iterdir():
            if file_path.is_file():
                dest_path = latest_dir / file_path.name
                file_path.replace(dest_path)

        logger.info("Forecast data successfully updated.")


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info("WRF downloader started.")

    try:
        asyncio.run(update_forecast_data())
        elapsed = datetime.now() - start_time
        logger.info("WRF downloader execution completed successfully in %s.", elapsed)
    except Exception as e:
        logger.critical(
            "Pipeline execution failed with unhandled exception: %s", e, exc_info=True
        )
        sys.exit(1)

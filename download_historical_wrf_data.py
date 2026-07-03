import os
import requests
import duckdb

import datetime as dt
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlsplit
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from create_wrf_db import process_grib2, insert_batch

TMP_DIR = "./historical_wrf_data_tmp"
TMP_DB_FILENAME = "historical_wrf_data_tmp.duckdb"
DB_FILENAME = "historical_wrf_data.duckdb"


GRIB_VARIABLES = {
    "total_cloud": "Total Cloud Cover",
    "temp_C": "2 metre temperature",
    "dewpoint_C": "2 metre dewpoint temperature",
    "surface_pressure_Pa": "Surface pressure",
    "precipitable_water_kg_m2": "Precipitable water",
    "low_cloud_cover": "Low cloud cover",
    "mid_cloud_cover": "Medium cloud cover",
    "high_cloud_cover": "High cloud cover",
    "visibility_m": "Visibility",
    "boundary_layer_height_m": "Boundary layer height",
}


def init_database(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast (
            run_time VARCHAR,
            forecast_hour INTEGER,
            lat_idx INTEGER,
            lon_idx INTEGER,
            lat DOUBLE,
            lon DOUBLE,
            total_cloud DOUBLE,
            temp_C DOUBLE,
            dewpoint_C DOUBLE,
            surface_pressure_Pa DOUBLE,
            precipitable_water_kg_m2 DOUBLE,
            low_cloud_cover DOUBLE,
            mid_cloud_cover DOUBLE,
            high_cloud_cover DOUBLE,
            visibility_m DOUBLE,
            boundary_layer_height_m DOUBLE
        );
        """
    )


def process_extracted_data(data: dict):
    df = pd.DataFrame(data)

    # convert temperatures from K to C
    df["temp_C"] = df["temp_C"] - 273.15
    df["dewpoint_C"] = df["dewpoint_C"] - 273.15

    return df


# add relevant data from grib2 file to duckdb
def ingest_grib2(file_path: str):
    file_path = Path(file_path)
    data = process_grib2(file_path)
    data = process_extracted_data(data)
    conn = duckdb.connect(TMP_DB_FILENAME)
    init_database(conn)
    insert_batch(conn, data)
    conn.close()
    print(f"✓ Inserted {file_path.name} into DuckDB")


def download_grib2(
    url: str,
    dest_dir: str | Path,
    filename: str | None = None,
    chunk_size: int = 1024 * 512,  # 512 KB
    timeout: tuple = (5, 30),  # (connect, read)
    max_retries: int = 5,
):
    """
    Download a file from a URL to a destination directory efficiently and reliably.

    Parameters
    ----------
    url : str
        HTTP or HTTPS URL of the file.
    dest_dir : str or Path
        Destination directory.
    filename : str, optional
        Output filename. If None, inferred from URL.
    chunk_size : int
        Streaming chunk size in bytes.
    timeout : tuple
        (connect_timeout, read_timeout) in seconds.
    max_retries : int
        Number of automatic retries for transient failures.

    Returns
    -------
    Path
        Path to the downloaded file.

    Raises
    ------
    requests.HTTPError
        For non-recoverable HTTP errors.
    """

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = os.path.basename(urlsplit(url).path)
        if not filename:
            raise ValueError("Could not infer filename from URL")

    final_path = dest_dir / filename
    temp_path = final_path.with_suffix(final_path.suffix + ".part")

    if final_path.exists():
        print(f"File already exists: {final_path}")
        return

    # Configure retries
    retries = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)

    headers = {}
    resume_byte_pos = 0

    # Resume support
    if temp_path.exists():
        resume_byte_pos = temp_path.stat().st_size
        headers["Range"] = f"bytes={resume_byte_pos}-"

    with requests.Session() as session:
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        with session.get(
            url, stream=True, headers=headers, timeout=timeout
        ) as response:
            response.raise_for_status()

            # Determine total size for progress bar
            total_size = response.headers.get("Content-Length")
            total_size = (
                int(total_size) + resume_byte_pos if total_size is not None else None
            )

            mode = "ab" if resume_byte_pos else "wb"

            with open(temp_path, mode) as f, tqdm(
                total=total_size,
                initial=resume_byte_pos,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
            ) as pbar:

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    # Atomic move
    temp_path.replace(final_path)


def download_and_ingest_grib2(url: str):
    download_grib2(url, TMP_DIR)
    file = os.path.join(TMP_DIR, url.split("/")[-1])
    if os.path.exists(file):
        ingest_grib2(file)
        os.remove(file)


def main():
    # create a duck DB to store historical WRF data
    # download 3 years of historical WRF data
    # one full day of for each week of every month of the past 3 years
    now = dt.datetime.now(tz=dt.timezone.utc)
    for week_offset in range(0, 52 * 3):
        day = now - dt.timedelta(weeks=week_offset)
        day = day.replace(hour=0, minute=0, second=0, microsecond=0)
        url_base = f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/{day.year:04d}/{day.month:02d}/{day.day:02d}/00/"
        url_file = f"WRF_cpt_07KM_{day.year:04d}{day.month:02d}{day.day:02d}00_{day.year:04d}{day.month:02d}{day.day:02d}"
        for fh in range(0, 24):
            fh_str = f"{fh:02d}"
            full_url = url_base + url_file + fh_str + ".grib2"
            download_and_ingest_grib2(full_url)

    # urls = [
    #     # list of URLs to historical WRF grib2 files
    #     "https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/2024/01/01/00/WRF_cpt_07KM_2024010100_2024010101.grib2",
    # ]
    # for url in urls:
    #     download_and_ingest_grib2(url)

    if os.path.exists(TMP_DB_FILENAME):
        Path(TMP_DB_FILENAME).replace(DB_FILENAME)

    if os.path.exists(TMP_DIR):
        os.rmdir(TMP_DIR)


if __name__ == "__main__":
    main()

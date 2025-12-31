import duckdb
import pygrib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timezone


# ==============================
# Configuration
# ==============================

GRIB_VARIABLES = {
    "cloud": "Total Cloud Cover",
    "temp": "2 metre temperature",
    "dewpoint": "2 metre dewpoint temperature",
}

DB_PATH = "forecast.duckdb"
TMP_DB_PATH = "forecast_tmp.duckdb"


def parse_filename(file_path: Path):
    parts = file_path.stem.split("_")
    run_time = parts[-2]
    forecast_hour = parts[-1]
    return run_time, forecast_hour


# ==============================
# Process a single GRIB2 file
# ==============================
def process_grib2(file_path: Path):
    print(f"Processing {file_path.name} ...")

    run_time, forecast_hour = parse_filename(file_path)

    grbs = pygrib.open(str(file_path))

    # Storage for extracted arrays
    extracted = {}
    grid_lats = None
    grid_lons = None

    # Extract each variable
    for var_key, var_name in GRIB_VARIABLES.items():
        try:
            msg = grbs.select(name=var_name)[0]
            # forecast_hour = msg.validDate.replace(tzinfo=timezone.utc).strftime(
            #     "%Y-%m-%d %H:%M:%S"
            # )
        except:
            print(f"⚠ Variable {var_name} not found in {file_path.name}")
            continue

        values = msg.values  # 2D array
        if grid_lats is None:
            grid_lats, grid_lons = msg.latlons()

        extracted[var_key] = values

    if grid_lats is None:
        raise RuntimeError(f"No variables extracted from {file_path}")

    nlat, nlon = grid_lats.shape

    # Flatten coordinates and add grid indices for fast querying
    lat_idx, lon_idx = np.indices(grid_lats.shape)

    # Build a long dict column by column
    data = {
        "run_time": np.full(grid_lats.size, run_time, dtype="object"),
        "forecast_hour": np.full(grid_lats.size, forecast_hour, dtype=np.int32),
        "lat_idx": lat_idx.ravel().astype(np.int32),
        "lon_idx": lon_idx.ravel().astype(np.int32),
        "lat": grid_lats.ravel().astype(np.float64),
        "lon": grid_lons.ravel().astype(np.float64),
    }

    # Add extracted variables
    for var_key, arr in extracted.items():
        data[var_key] = arr.ravel().astype(np.float32)

    return data


# ==============================
# Insert the extracted dictionary into DuckDB
# ==============================


def insert_into_duckdb(conn, data):
    # Convert dict → DuckDB relation (via Arrow)
    rel = conn.from_df(pd.DataFrame(data))


# ==============================
# Create DB schema if not exists
# ==============================


def init_database(conn):
    conn.execute("DROP TABLE IF EXISTS forecast")
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS forecast (
        run_time VARCHAR,
        forecast_hour INTEGER,
        lat_idx INTEGER,
        lon_idx INTEGER,
        lat DOUBLE,
        lon DOUBLE,
        cloud DOUBLE,
        temp DOUBLE,
        dewpoint DOUBLE
    );
    """
    )


# ==============================
# Insert a batch of rows
# Faster than row by row
# ==============================


def insert_batch(conn, data):
    conn.register("tmp", data)
    conn.execute(
        """
        INSERT INTO forecast
        SELECT * FROM tmp;
    """
    )
    conn.unregister("tmp")


# ==============================
# Main ingestion routine
# ==============================


def ingest_folder(folder: str):
    conn = duckdb.connect(TMP_DB_PATH)
    init_database(conn)

    folder = Path(folder)
    grib_files = sorted(folder.glob("*.grib2"))

    for file in grib_files:
        data = process_grib2(file)
        insert_batch(conn, data)
        print(f"✓ Inserted {file.name} into DuckDB")

    conn.close()
    print("🎉 Ingestion complete!")


# ==============================
# Run the script
# ==============================
if __name__ == "__main__":
    ingest_folder("latest_wrf_files")
    # Move temp DB to final location
    Path(TMP_DB_PATH).rename(DB_PATH)

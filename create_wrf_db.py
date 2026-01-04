import duckdb
import pygrib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timezone


# ==============================
# Configuration
# ==============================

DB_PATH = "forecast.duckdb"
TMP_DB_PATH = "forecast_tmp.duckdb"

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


def add_transparency_index(
    df: pd.DataFrame,
):
    """
    Compute atmospheric transparency index (0–1) and add it to the DataFrame.
    """
    df_tmp = df.copy()

    sp = df["surface_pressure_Pa"]
    pwat = df["precipitable_water_kg_m2"]
    tcc = df["total_cloud"]
    lcc = df["low_cloud_cover"]
    mcc = df["mid_cloud_cover"]
    hcc = df["high_cloud_cover"]
    vis = df["visibility_m"]
    blh = df["boundary_layer_height_m"]

    # ---------------------------
    # 1. Rayleigh scattering
    # ---------------------------
    tau_rayleigh = 0.097 * (sp / 101325.0)

    # # ---------------------------
    # # 2. Gas absorption (water vapor)
    # # ---------------------------
    # # pwat units: 1 kg/m2 = 1 mm
    # k_wv = 0.015  # mm^-1
    # tau_gas = k_wv * pwat

    # # ---------------------------
    # # 3. Cloud optical depth
    # # ---------------------------
    # tau_cloud = 20.0 * lcc + 5.0 * mcc + 1.0 * hcc
    tau_cloud = pd.Series(np.zeros(len(df_tmp)))
    tau_cloud.where(tcc < 0.3, 10.0, inplace=True)

    # ---------------------------
    # 4. Hydrometeor / fog extinction
    # ---------------------------
    # tau_hydro = np.where(vis > 0, (3.912 / vis) * blh, 5.0)
    # tau_hydro = np.clip(tau_hydro, 0.0, 5.0)
    # tau_hydro = pd.Series(tau_hydro)

    # # ---------------------------
    # # 5. Aerosol proxy (clear sky only)
    # # ---------------------------
    # tau_aerosol = np.where((tcc < 0.2) & (vis > 0), 0.15 / (vis / 1000.0), 0.0)
    # tau_aerosol = pd.Series(tau_aerosol)
    tau_aerosol = np.where(vis > 0, (3.0 / (vis / 1000)) * (blh / 1000), 5.0)
    tau_aerosol = np.clip(tau_aerosol, 0.0, 5.0)
    tau_aerosol = pd.Series(tau_aerosol)

    # ---------------------------
    # 6. Total optical depth
    # ---------------------------
    # tau_total = tau_rayleigh + tau_gas + tau_aerosol + tau_cloud + tau_hydro
    tau_total = tau_cloud + tau_rayleigh + tau_aerosol + 0.03

    # ---------------------------
    # 7. Transparency index
    # ---------------------------
    transparency = tau_total.apply(lambda x: np.exp(-x))
    transparency.clip(0.0, 1.0, inplace=True)

    df_tmp["transparency_index"] = transparency

    return df_tmp


def parse_filename(file_path: Path):
    parts = file_path.stem.split("_")
    run_time = parts[-2]
    forecast_hour = parts[-1]
    return run_time, forecast_hour


def process_extracted_data(data: dict):
    df = pd.DataFrame(data)

    # convert temperatures from K to C
    df["temp_C"] = df["temp_C"] - 273.15
    df["dewpoint_C"] = df["dewpoint_C"] - 273.15

    # Compute transparency index
    df = add_transparency_index(df)

    # drop unneeded columns
    df.drop(
        columns=[
            "surface_pressure_Pa",
            "precipitable_water_kg_m2",
            "low_cloud_cover",
            "mid_cloud_cover",
            "high_cloud_cover",
            "visibility_m",
            "boundary_layer_height_m",
        ],
        inplace=True,
    )
    return df


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


# # ==============================
# # Insert the extracted dictionary into DuckDB
# # ==============================
# def insert_into_duckdb(conn, data):
#     # Convert dict → DuckDB relation (via Arrow)
#     rel = conn.from_df(pd.DataFrame(data))


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
        dewpoint DOUBLE,
        transparency DOUBLE
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
        data = process_extracted_data(data)
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

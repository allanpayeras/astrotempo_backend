import time
import duckdb
import logging

import numpy as np
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from skyfield import almanac
from skyfield.api import load, wgs84


DB_PATH = "forecast.duckdb"
TZ = ZoneInfo("America/Sao_Paulo")

app = FastAPI(title="Astro Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

TS = load.timescale()
EPH = load("de421.bsp")
SUN = EPH["sun"]
MOON = EPH["moon"]
EARTH = EPH["earth"]

LOGGER = logging.getLogger("uvicorn")


def get_forecast_for_location(lat: float, lon: float) -> pd.DataFrame:
    """
    Retrieve forecast data for a given latitude and longitude from
    the DuckDB database.
    """

    # Find nearest grid point
    conn = duckdb.connect(DB_PATH, read_only=True)

    grid = conn.execute("""
        SELECT lat_idx, lon_idx, lat, lon
        FROM grid
    """).fetchdf()

    GRID_LAT = grid["lat"].to_numpy()
    GRID_LON = grid["lon"].to_numpy()
    GRID_LAT_IDX = grid["lat_idx"].to_numpy()
    GRID_LON_IDX = grid["lon_idx"].to_numpy()

    dist2 = (GRID_LAT - lat) ** 2 + (GRID_LON - lon) ** 2
    i = np.argmin(dist2)

    lat_idx = int(GRID_LAT_IDX[i])
    lon_idx = int(GRID_LON_IDX[i])

    # Retrieve forecast
    df = conn.execute(
        """
        SELECT
            forecast_hour,
            cloud,
            temp,
            dewpoint,
            transparency
        FROM forecast
        WHERE lat_idx = ?
          AND lon_idx = ?
        ORDER BY forecast_hour
        """,
        (lat_idx, lon_idx),
    ).fetchdf()

    conn.close()

    return df


def get_sun_times(
    start: datetime, end: datetime, latitude: float, longitude: float, tz: ZoneInfo
):
    # Observer location
    observer = wgs84.latlon(latitude, longitude)

    # Build start/end datetime for this date
    start = datetime(start.year, start.month, start.day, 0, 0, tzinfo=tz)
    end = datetime(end.year, end.month, end.day, 23, 59, tzinfo=tz)

    t0 = TS.from_datetime(start)
    t1 = TS.from_datetime(end)

    # Compute sunrise/sunset
    f = almanac.sunrise_sunset(EPH, observer)
    times, events = almanac.find_discrete(t0, t1, f)

    sunrise = []
    sunset = []

    for t, e in zip(times, events):
        dt = t.utc_datetime().astimezone(tz)
        if e == 1:  # Sunrise
            sunrise.append(dt)
        elif e == 0:  # Sunset
            sunset.append(dt)

    # Compute astronomical twilight (-18 degrees)
    f2 = almanac.dark_twilight_day(EPH, observer)
    t2, ev2 = almanac.find_discrete(t0, t1, f2)

    astronomical_sunrise = []
    astronomical_sunset = []

    for i, (t, e) in enumerate(zip(t2, ev2)):
        dt = t.utc_datetime().astimezone(tz)

        # Night = 0, Astronomical twilight = 1
        # Astronomical sunrise → transition 0 → 1 in morning
        if e == 1 and (i == 0 or ev2[i - 1] == 0):
            astronomical_sunrise.append(dt)

        # Astronomical sunset → transition 1 → 0 in evening
        if e == 0 and (i > 0 and ev2[i - 1] == 1):
            astronomical_sunset.append(dt)

    return pd.DataFrame(
        {
            "sunrise": pd.Series(sunrise),
            "sunset": pd.Series(sunset),
            "astronomical_sunrise": pd.Series(astronomical_sunrise),
            "astronomical_sunset": pd.Series(astronomical_sunset),
        }
    )


def get_moon_illumination(date: datetime):
    t = TS.from_datetime(date)

    e = EARTH.at(t)
    m = e.observe(MOON).apparent()

    return 100.0 * m.fraction_illuminated(SUN)


def get_moon_up_illumination(
    start_dt: datetime,
    end_dt: datetime,
    latitude: float,
    longitude: float,
    tz: ZoneInfo,
    elevation_m: float = 0.0,
):
    """
    Returns a list of (datetime, event) tuples,
    where event is 'moonrise' or 'moonset'.
    """
    # Ensure timezone-awareness
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=tz)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=tz)

    # create an array of datetimes between start_dt and end_dt at 1-minute intervals
    times = [
        start_dt + timedelta(minutes=1 * i)
        for i in range(int((end_dt - start_dt).total_seconds() // 60))
    ]
    times = np.array(times)
    illumination = np.full(len(times), np.nan)

    # Skyfield setup
    observer = EARTH + wgs84.latlon(latitude, longitude, elevation_m)

    # Convert to Skyfield time
    t0 = TS.from_datetime(start_dt)
    t1 = TS.from_datetime(end_dt)

    t_r, _ = almanac.find_risings(observer, MOON, t0, t1)
    t_s, _ = almanac.find_settings(observer, MOON, t0, t1)

    if t_s[0] < t_r[0]:
        # Moon is already up at start_dt
        t_r = np.insert(t_r, 0, t0)

    for r, s in zip(t_r, t_s):
        r = r.utc_datetime().astimezone(tz)
        s = s.utc_datetime().astimezone(tz)
        ill = get_moon_illumination(r + (s - r) / 2)
        illumination = np.where((times >= r) & (times <= s), ill, illumination)

    # convert times from datetime to strings in ISO format
    times = [t.strftime("%Y-%m-%dT%H:%M:%S%z") for t in times]

    df_moon = pd.DataFrame({"time": times, "illumination": illumination})
    return df_moon.replace({np.nan: None})


# API endpoint
@app.get("/forecast")
def forecast(lat: float = Query(...), lon: float = Query(...)):
    t0 = time.perf_counter()

    df = get_forecast_for_location(lat, lon)
    LOGGER.info("DB: %.3fs", time.perf_counter() - t0)

    df["time"] = pd.to_datetime(df["forecast_hour"], format="%Y%m%d%H", utc=True)
    # df = df[df["time"] >= datetime.now(tz=ZoneInfo("UTC")) - timedelta(hours=4)]

    start = df["time"].iloc[0].tz_convert(TZ)
    end = df["time"].iloc[-1].tz_convert(TZ)

    t1 = time.perf_counter()
    df_sun = get_sun_times(start, end, lat, lon, TZ)
    LOGGER.info("Sun: %.3fs", time.perf_counter() - t1)

    t2 = time.perf_counter()
    df_moon = get_moon_up_illumination(start, end, lat, lon, TZ)
    LOGGER.info("Moon: %.3fs", time.perf_counter() - t2)

    LOGGER.info("Total: %.3fs", time.perf_counter() - t0)

    return {
        "forecast": df.drop(columns="time").to_dict(orient="records"),
        "sun": df_sun.to_dict(orient="records"),
        "moon": df_moon.to_dict(orient="records"),
    }
import os
import duckdb
import folium
import re

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from streamlit_js_eval import get_geolocation
from datetime import datetime, timezone, timedelta
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo

from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from skyfield.api import load, wgs84
from skyfield import almanac


blue_scale = [
    "#005bb5",
    "#0069D1",
    "#4EA4FB",
    "#B8DBFF",
    "#E6F2FF",
]
bounderies_cloud = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
colorscale_cloud = []
for i in range(len(blue_scale)):
    colorscale_cloud.append([bounderies_cloud[i], blue_scale[i]])
    colorscale_cloud.append([bounderies_cloud[i + 1], blue_scale[i]])

blue_scale_transparency = blue_scale[::-1]
bounderies_transparency = [0, 0.2, 0.4, 0.65, 0.85, 1.0]
colorscale_transparency = []
for i in range(len(blue_scale)):
    colorscale_transparency.append(
        [bounderies_transparency[i], blue_scale_transparency[i]]
    )
    colorscale_transparency.append(
        [bounderies_transparency[i + 1], blue_scale_transparency[i]]
    )


def get_sun_times(date: datetime, latitude: float, longitude: float, tz: ZoneInfo):
    # Load ephemeris and time scale
    ts = load.timescale()
    eph = load("de421.bsp")

    # Observer location
    observer = wgs84.latlon(latitude, longitude)

    # Build start/end datetime for this date
    start = datetime(date.year, date.month, date.day, 0, 0, tzinfo=tz)
    end = datetime(date.year, date.month, date.day, 23, 59, tzinfo=tz)

    t0 = ts.from_datetime(start)
    t1 = ts.from_datetime(end)

    # Compute sunrise/sunset
    f = almanac.sunrise_sunset(eph, observer)
    times, events = almanac.find_discrete(t0, t1, f)

    sunrise, sunset = None, None

    for t, e in zip(times, events):
        dt = t.utc_datetime().astimezone(tz)
        if e == 1:  # Sunrise
            sunrise = dt
        elif e == 0:  # Sunset
            sunset = dt

    # Compute astronomical twilight (-18 degrees)
    f2 = almanac.dark_twilight_day(eph, observer)
    t2, ev2 = almanac.find_discrete(t0, t1, f2)

    astronomical_sunrise, astronomical_sunset = None, None

    for t, e in zip(t2, ev2):
        dt = t.utc_datetime().astimezone(tz)

        # Night = 0, Astronomical twilight = 1
        # Astronomical sunrise → transition 0 → 1 in morning
        if e == 1 and astronomical_sunrise is None:
            astronomical_sunrise = dt

        # Astronomical sunset → transition 1 → 0 in evening
        if e == 0 and astronomical_sunrise is not None:
            astronomical_sunset = dt

    return sunrise, sunset, astronomical_sunrise, astronomical_sunset


def get_color_for_value(value: float) -> str:
    """Return appropriate hex color from blue_scale for cloud-cover percentage (0–100)."""
    v = value / 100
    for i in range(len(bounderies_cloud) - 1):
        if bounderies_cloud[i] <= v < bounderies_cloud[i + 1]:
            return blue_scale[i]
    return blue_scale[-1]


def build_cloud_cover_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame representing next 6 days with cloud-cover squares:
    Columns: ['day','0–6h','6–12h','12–18h','18–24h']
    """

    df = df.copy()
    df["dt"] = pd.to_datetime(df["forecast_hour"], format="%Y%m%d%H", utc=True)
    df["dt_local"] = df["dt"].dt.tz_convert(ZoneInfo("America/Sao_Paulo"))
    df["date"] = df["dt_local"].dt.date
    df["hour"] = df["dt_local"].dt.hour

    # bucket assignment
    def bucket(h):
        if 0 <= h < 6:
            return "0–6h"
        if 6 <= h < 12:
            return "6–12h"
        if 12 <= h < 18:
            return "12–18h"
        return "18–24h"

    df["bucket"] = df["hour"].apply(bucket)

    today = datetime.now(tz=ZoneInfo("America/Sao_Paulo")).date()
    next_days = sorted([d for d in df["date"].unique() if d >= today])[:6]

    rows = []
    for d in next_days:
        day_df = df[df["date"] == d]
        day = datetime(
            d.year, d.month, d.day, hour=20, tzinfo=ZoneInfo("America/Sao_Paulo")
        )

        row = {"day": pd.to_datetime(d).strftime("%a, %d %b")}
        for bucket_name in ["0–6h", "6–12h", "12–18h", "18–24h"]:
            subset = day_df[day_df["bucket"] == bucket_name]
            if subset.empty:
                row[bucket_name] = ""
            else:
                mean_cc = subset["cloud"].mean()
                color = get_color_for_value(mean_cc)
                row[bucket_name] = (
                    f'<div style="width:25px;height:25px;background:{color};border-radius:4px;"></div>'
                )
        row["moon"] = f"{get_moon_illumination(day):.0f}%"
        rows.append(row)
    rows[0]["day"] = "<b>Today</b>"

    return pd.DataFrame(rows)


def get_forecast_for_location(db_path: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Returns the hourly forecast for the grid point nearest to (lat, lon).

    Arguments:
        db_path (str): Path to the DuckDB file, e.g., 'forecast.duckdb'
        lat (float): Latitude of user location
        lon (float): Longitude of user location

    Returns:
        pandas.DataFrame: Columns = valid_time, cloud_cover, temp, dewpoint
    """

    conn = duckdb.connect(db_path, read_only=True)

    # 1. Find the nearest grid point stored in the DB
    nearest_query = """
        SELECT lat, lon
        FROM forecast
        ORDER BY (lat - ?) * (lat - ?) + (lon - ?) * (lon - ?) ASC
        LIMIT 1
    """
    nearest_lat, nearest_lon = conn.execute(
        nearest_query, (lat, lat, lon, lon)
    ).fetchone()

    # 2. Retrieve forecast for this grid point
    forecast_query = """
        SELECT 
            forecast_hour,
            cloud,
            temp,
            dewpoint,
            transparency
        FROM forecast
        WHERE lat = ? AND lon = ?
        ORDER BY forecast_hour ASC
    """

    df = conn.execute(forecast_query, (nearest_lat, nearest_lon)).fetchdf()
    conn.close()

    return df


def get_moon_illumination(date: datetime):
    ts = load.timescale()
    t = ts.from_datetime(date)

    eph = load("de421.bsp")
    sun, moon, earth = eph["sun"], eph["moon"], eph["earth"]

    e = earth.at(t)
    m = e.observe(moon).apparent()

    return 100.0 * m.fraction_illuminated(sun)


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
    # create a corresponding numpy array initialized with np.nan
    illumination = np.full(len(times), np.nan)

    # Skyfield setup
    ts = load.timescale()
    eph = load("de421.bsp")
    moon = eph["Moon"]
    earth = eph["Earth"]
    observer = earth + wgs84.latlon(latitude, longitude, elevation_m)

    # Convert to Skyfield time
    t0 = ts.from_datetime(start_dt)
    t1 = ts.from_datetime(end_dt)

    t_r, _ = almanac.find_risings(observer, moon, t0, t1)
    t_s, _ = almanac.find_settings(observer, moon, t0, t1)

    if t_s[0] < t_r[0]:
        # Moon is already up at start_dt
        t_r = np.insert(t_r, 0, t0)

    for r, s in zip(t_r, t_s):
        r = r.utc_datetime().astimezone(tz)
        s = s.utc_datetime().astimezone(tz)
        ill = get_moon_illumination(r + (s - r) / 2)
        illumination = np.where((times >= r) & (times <= s), ill, illumination)

    return times, illumination


def to_2d(values):
    return np.expand_dims(values, axis=0)


def get_update_time(filename: str) -> datetime:
    """Extract the forecast update time from a WRF grib filename.

    The filename contains an update time in UTC in the format YYYYMMDDHH.
    Example: WRF_cpt_07KM_2025122600_2025122601.grib2 -> 2025-12-26 00:00 UTC

    Args:
        filename: Path or filename string containing the timestamp.

    Returns:
        A timezone-aware datetime in UTC.

    Raises:
        ValueError: if no YYYYMMDDHH timestamp is found in the filename.
    """

    # Work on basename to be robust to paths
    name = os.path.basename(filename)

    # Find first occurrence of 10 consecutive digits (YYYYMMDDHH)
    m = re.search(r"(\d{10})", name)
    if not m:
        raise ValueError(f"No update timestamp found in filename: {filename}")

    ts = m.group(1)
    # Parse into datetime and set UTC tzinfo
    dt = datetime.strptime(ts, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return dt


def get_address_text(address: dict):
    # city_district, district, borough, suburb, subdivision
    values = []
    if "city_district" in address.keys():
        values.append(address["city_district"])
    elif "suburb" in address.keys():
        values.append(address["suburb"])
    elif "borough" in address.keys():
        values.append(address["borough"])
    elif "district" in address.keys():
        values.append(address["district"])
    elif "neighbourhood" in address.keys():
        values.append(address["neighbourhood"])
    # municipality, city, town, village
    if "city" in address.keys():
        values.append(address["city"])
    elif "town" in address.keys():
        values.append(address["town"])
    elif "village" in address.keys():
        values.append(address["village"])
    elif "municipality" in address.keys():
        values.append(address["municipality"])
    # region, state, state_district, county, ISO3166-2-lvl
    if "state" in address.keys():
        values.append(address["state"])
    elif "region" in address.keys():
        values.append(address["region"])

    if values:
        return ", ".join(values)
    else:
        return None


def main():
    st.set_page_config(page_title="Astro Forecast", layout="wide")

    ss = st.session_state
    if "lat" not in ss:
        ss.lat = None
    if "lon" not in ss:
        ss.lon = None

    grib_files = os.listdir("latest_wrf_files")
    update_time = get_update_time(grib_files[0])

    geolocator = Nominatim(user_agent="streamlit_reverse_geocoder")
    if ss.lat is None or ss.lon is None:
        loc = get_geolocation()
        if loc is not None:
            ss.lat = loc["coords"]["latitude"]
            ss.lon = 360 + loc["coords"]["longitude"]
            st.rerun()
        else:
            st.info("Requesting location permission from browser…")
            st.stop()

    col_map, col_geo_info = st.columns([0.4, 0.6])

    with col_map:
        m = folium.Map(location=[ss.lat, ss.lon], zoom_start=12)

        folium.Marker(
            [ss.lat, ss.lon],
            # popup=f"({ss.lat:.3f}, {ss.lon:.3f})",
            icon=folium.Icon(color="red"),
        ).add_to(m)

        # Enable clicking
        m.add_child(folium.LatLngPopup())

        map_data = st_folium(
            m, width=550, height=350, returned_objects=["last_clicked"]
        )

        if map_data and map_data.get("last_clicked"):
            ss.lat = map_data["last_clicked"]["lat"]
            ss.lon = map_data["last_clicked"]["lng"]
            st.rerun()
    with col_geo_info:
        try:
            location = geolocator.reverse((ss.lat, ss.lon), language="en")
            address = location.raw["address"] if location else "No address found"
            formated_address = get_address_text(address)
            if formated_address is not None:
                st.markdown(
                    f"<big>**Forecast for** {formated_address}</big>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                            <p style="line-height: 1.1;">
                                latitude: <code>{ss.lat:.3f}</code><br>
                                longitude: <code>{ss.lon:.3f}</code>
                            </p>
                            """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<small>Latest forecast update (UTC): {update_time.strftime('%Y-%m-%d %H:%M')}</small>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Error: {e}")

    with st.spinner("Loading forecast data..."):
        df = get_forecast_for_location("forecast.duckdb", ss.lat, ss.lon)
        df["time"] = pd.to_datetime(df["forecast_hour"], format="%Y%m%d%H", utc=True)
        df = df[df["time"] >= datetime.now(timezone.utc) - timedelta(hours=4)]

        hours = df["time"].to_list()
        hours = [t.astimezone(ZoneInfo("America/Sao_Paulo")) for t in hours]
        tcc = df["cloud"].to_numpy()
        transparency = df["transparency"].to_numpy()
        temp = df["temp"].to_numpy()
        dew = df["dewpoint"].to_numpy()

        time, moon_is_up = get_moon_up_illumination(
            hours[0],
            hours[-1],
            ss.lat,
            ss.lon,
            tz=ZoneInfo("America/Sao_Paulo"),
        )

    col_table, col_chart = st.columns([0.35, 0.65])
    with col_table:
        table_container = st.empty()
    with col_chart:
        chart_container = st.empty()

    with table_container:
        with st.spinner("Creating summary table..."):
            summary_df = build_cloud_cover_summary(df)
        st.write(
            summary_df.to_html(escape=False, justify="center", index=False),
            unsafe_allow_html=True,
        )

    with chart_container:
        with st.spinner("Creating forecast chart..."):
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.02, 0.02, 0.02, 0.10],
                specs=[
                    [{"type": "heatmap"}],
                    [{"type": "heatmap"}],
                    [{"type": "heatmap"}],
                    [{"type": "scatter"}],
                ],
            )

            # plot data
            fig.add_trace(
                go.Heatmap(
                    z=to_2d(tcc),
                    x=np.array(hours) + timedelta(minutes=30),
                    y=[""],
                    colorscale=colorscale_cloud,
                    zmin=0,
                    zmax=100,
                    xgap=8,
                    colorbar=None,
                    showscale=False,
                    opacity=1,
                    hoverinfo="skip",
                    name="Cloud Cover",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=to_2d(tcc),
                    x=np.array(hours) + timedelta(minutes=30),
                    y=[""],
                    colorscale=colorscale_cloud,
                    zmin=0,
                    zmax=100,
                    xgap=8,
                    colorbar=None,
                    showscale=False,
                    opacity=0,
                    hoverinfo="skip",
                    name="Cloud Cover",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=to_2d(moon_is_up),
                    x=time,
                    y=[""],
                    colorscale=colorscale_cloud,
                    zmin=0,
                    zmax=100,
                    colorbar=None,
                    showscale=False,
                    opacity=1,
                    hoverinfo="skip",
                    name="Moon",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=to_2d(transparency),
                    x=np.array(hours) + timedelta(minutes=30),
                    y=[""],
                    colorscale=colorscale_transparency,
                    zmin=0,
                    zmax=1,
                    xgap=8,
                    colorbar=None,
                    showscale=False,
                    opacity=1,
                    hoverinfo="skip",
                    name="Transparency",
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=temp,
                    mode="lines+text",
                    text=[f"{t:.0f}" for t in temp],
                    textposition="top right",
                    opacity=1,
                    name="Temperature",
                    hoverinfo="skip",
                ),
                row=4,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=dew,
                    mode="lines+text",
                    text=[f"{d:.0f}" for d in dew],
                    textposition="bottom right",
                    opacity=1,
                    name="Dew Point",
                    hoverinfo="skip",
                ),
                row=4,
                col=1,
            )

            # add vertical lines for now and shading for day/night
            now = datetime.now(tz=ZoneInfo("America/Sao_Paulo"))
            fig.add_shape(
                type="line",
                x0=now,
                x1=now,  # 0–1 in paper coordinates
                y0=0,
                y1=1,  # full height of figure
                xref="x1",
                yref="paper",
                line=dict(color="red", width=3, dash="solid"),
            )
            max_date = np.max(hours)
            max_day_delta = (max_date - now).days + 2
            for day_delta in range(max_day_delta):
                d = now.replace(hour=0, minute=0, second=0) + timedelta(days=day_delta)
                sunrise, sunset, sr_astronomical, ss_astronomical = get_sun_times(
                    d,
                    ss.lat,
                    ss.lon,
                    tz=ZoneInfo("America/Sao_Paulo"),
                )
                # daytime shading
                fig.add_shape(
                    type="rect",
                    x0=sunrise,
                    x1=sunset,
                    y0=-1,
                    y1=1.2,
                    xref="x",
                    yref="paper",
                    fillcolor="#37376D",
                    line_width=0,
                    opacity=0.4,
                    layer="below",
                )
                # twilight shading
                fig.add_shape(
                    type="rect",
                    x0=sr_astronomical,
                    x1=sunrise,
                    y0=-1,
                    y1=1.2,
                    xref="x",
                    yref="paper",
                    fillcolor="#27274E",
                    line_width=0,
                    opacity=0.5,
                    layer="below",
                )
                fig.add_shape(
                    type="rect",
                    x0=sunset,
                    x1=ss_astronomical,
                    y0=-1,
                    y1=1.2,
                    xref="x",
                    yref="paper",
                    fillcolor="#27274E",
                    line_width=0,
                    opacity=0.5,
                    layer="below",
                )

            # configure axes and layout
            fig.update_yaxes(
                title_text="Clouds",
                ticks="",
                row=1,
                col=1,
                fixedrange=True,
                showgrid=False,
            )
            fig.update_yaxes(
                title_text="Moon",
                ticks="",
                row=2,
                col=1,
                fixedrange=True,
                showgrid=False,
            )
            fig.update_yaxes(
                title_text="Transp",
                ticks="",
                row=3,
                col=1,
                fixedrange=True,
                showgrid=False,
            )
            fig.update_yaxes(
                title_text="Temperature (°C)",
                ticks="",
                showgrid=False,
                showticklabels=False,
                row=4,
                col=1,
                fixedrange=True,
            )
            fig.update_xaxes(
                ticks="",
                showticklabels=True,
                ticklabelposition="outside",
                ticklabelmode="period",
                dtick=(60 * 60000),
                tickformat="%H",
                row=1,
                col=1,
                side="top",
                autorange=False,
                range=[
                    datetime.now(tz=ZoneInfo("America/Sao_Paulo")) - timedelta(hours=3),
                    datetime.now(tz=ZoneInfo("America/Sao_Paulo")) + timedelta(days=1),
                ],
            )
            dates = df["time"].dt.date.unique()
            noons = [pd.Timestamp(day) + pd.Timedelta(hours=12) for day in dates]
            ticktexts = [d.strftime("%a %d %b") for d in noons]
            fig.update_layout(
                xaxis5=dict(
                    overlaying="x",  # draw on top of x-axis
                    matches="x",
                    side="top",
                    tickvals=noons,
                    ticktext=ticktexts,
                    ticklabelposition="outside",
                    ticks="",
                    ticklabelstandoff=20,
                    tickfont_size=16,
                ),
                margin=dict(t=60, b=10),
            )
            fig.data[1].update(xaxis="x5")
            fig.update_xaxes(ticks="", row=2, col=1)
            fig.update_xaxes(showticklabels=False, row=4, col=1)
            fig.update_layout(
                paper_bgcolor="#181830",
                plot_bgcolor="#181830",
                showlegend=False,
                height=400,
                width=None,
                autosize=False,
                dragmode="pan",
            )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "scrollZoom": False,  # disable wheel zoom
                "doubleClick": False,  # disable reset on double-click
                "displayModeBar": False,  # hide tool bar
                "editable": False,  # disable edit tools
                "staticPlot": False,  # must be False or pan won't work
            },
        )


if __name__ == "__main__":
    main()

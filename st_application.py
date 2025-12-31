import os
import duckdb
import folium
import time as time_pkg
import re

import pygrib as pg
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import astropy.units as u

from streamlit_js_eval import get_geolocation
from datetime import datetime, timezone, timedelta
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_moon
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim


blue_scale = [
    "#005bb5",
    "#0069D1",
    "#2E96FF",
    "#B8DBFF",
    "#E6F2FF",
]
bounderies = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
blue_colorscale = []
for i in range(len(blue_scale)):
    blue_colorscale.append([bounderies[i], blue_scale[i]])
    blue_colorscale.append([bounderies[i + 1], blue_scale[i]])


def get_color_for_value(value: float) -> str:
    """Return appropriate hex color from blue_scale for cloud-cover percentage (0–100)."""
    v = value / 100
    for i in range(len(bounderies) - 1):
        if bounderies[i] <= v < bounderies[i + 1]:
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

        row = {"day": pd.to_datetime(d).strftime("%a %d %b")}
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
            dewpoint
        FROM forecast
        WHERE lat = ? AND lon = ?
        ORDER BY forecast_hour ASC
    """

    df = conn.execute(forecast_query, (nearest_lat, nearest_lon)).fetchdf()
    conn.close()

    return df


def get_sunrise_sunset(
    date: datetime, lat, lon, elevation=0, timezone=timezone.utc, type="civil"
):
    """
    Compute sunrise and sunset for given date and location.
    """
    location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=elevation)

    # Sample times every 2 minutes across the whole day
    date = date.strftime("%Y-%m-%d")
    times = Time(f"{date} 00:00:00") + np.linspace(0, 24, 24 * 300) * u.hour

    # Sun altitude for each sample
    altaz = get_sun(times).transform_to(
        AltAz(obstime=times, location=location, pressure=1000 * u.hPa)
    )
    altitudes = altaz.alt.deg

    if type == "civil":
        altitudes_corrected = (
            altitudes + 0.833
        )  # atmospheric refraction + solar radius for sunrise
    elif type == "astronomical":
        altitudes_corrected = altitudes + 18.0  # astronomical twilight

    # Find index where altitude crosses 0 degrees (sunrise/set)
    sign = np.sign(altitudes_corrected)
    crossings = np.where(np.diff(sign))[0]

    if len(crossings) < 2:
        return None, None  # Polar day or night

    sunrise_time = times[crossings[0]].to_datetime(timezone=timezone)
    sunset_time = times[crossings[1]].to_datetime(timezone=timezone)

    return sunrise_time, sunset_time


def get_moon_altitudes(time_range, lat, lon, elevation=0, timezone=timezone.utc):
    """
    Compute moonrise and moonset for given date and location using altitude crossings.
    """
    location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=elevation)

    # Sample times every 30 seconds across the time range
    start = time_range[0].astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    end = time_range[1].astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
    duration_hours = (time_range[1] - time_range[0]).total_seconds() / 3600
    times = (
        Time(start) + np.linspace(0, duration_hours, int(duration_hours * 120)) * u.hour
    )
    # Compute moon altitude
    altaz = get_moon(times).transform_to(
        AltAz(obstime=times, location=location, pressure=1000 * u.hPa)
    )
    times = [t.to_datetime(timezone=timezone) for t in times]

    return times, altaz.alt.deg


# Convert values to 2D arrays (1 row heatmaps)
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

    # ---------- SESSION STATE INITIALIZATION ----------
    if "lat" not in ss:
        ss.lat = None
    if "lon" not in ss:
        ss.lon = None
    # if "clear_next_run" not in ss:
    #     ss.clear_next_run = True

    # Create placeholders ONCE
    # if "location_container" not in ss:
    #     ss.location_container = st.empty()
    # # if "map_container" not in ss:
    # #     ss.map_container = st.empty()
    # # if "chart_container" not in ss:
    # #     ss.chart_container = st.empty()
    # if "summary_container" not in ss:
    #     ss.summary_container = st.empty()
    # if "info_container" not in ss:
    #     ss.info_container = st.empty()

    # If a new render cycle is requested, clear everything ONCE
    # if ss.clear_next_run:
    #     # ss.location_container.empty()
    #     ss.summary_container.empty()
    #     ss.info_container.empty()
    #     # Reset flag and continue execution
    #     ss.clear_next_run = False

    # get update time from forecast file
    grib_files = os.listdir("latest_wrf_files")
    update_time = get_update_time(grib_files[0])

    geolocator = Nominatim(user_agent="streamlit_reverse_geocoder")
    if ss.lat is None or ss.lon is None:
        loc = get_geolocation()
        if loc is not None:
            ss.lat = loc["coords"]["latitude"]
            ss.lon = 360 + loc["coords"]["longitude"]
            # ss.clear_next_run = True
            st.rerun()
        else:
            st.info("Requesting location permission from browser…")
            st.stop()

    col_map, col_geo_info = st.columns([0.4, 0.6])

    # containers created inside columns – this guarantees side-by-side display
    with col_map:
        m = folium.Map(location=[ss.lat, ss.lon], zoom_start=11)

        folium.Marker(
            [ss.lat, ss.lon],
            # popup=f"({ss.lat:.3f}, {ss.lon:.3f})",
            icon=folium.Icon(color="red"),
        ).add_to(m)

        # Enable clicking
        m.add_child(folium.LatLngPopup())

        # ---------------------------
        # Step 4: Render map and capture clicks
        # ---------------------------
        map_data = st_folium(
            m, width=550, height=350, returned_objects=["last_clicked"]
        )

        if map_data and map_data.get("last_clicked"):
            ss.lat = map_data["last_clicked"]["lat"]
            ss.lon = map_data["last_clicked"]["lng"]
            # ss.clear_next_run = True
            st.rerun()
    with col_geo_info:
        try:
            print("lat:", ss.lat, "lon:", ss.lon)
            location = geolocator.reverse((ss.lat, ss.lon), language="en")
            print(location.raw)
            address = location.raw["address"] if location else "No address found"
            formated_address = get_address_text(address)
            print("formated_address:", formated_address)
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
        except Exception as e:
            st.error(f"Error: {e}")

    col_table, col_chart = st.columns([0.3, 0.7])

    # containers created inside columns – this guarantees side-by-side display
    with col_table:
        table_container = st.empty()
    with col_chart:
        chart_container = st.empty()

    # store them in session_state only once so that they survive reruns
    # if "map_container" not in ss:
    #     ss.map_container = map_container
    # if "chart_container" not in ss:
    #     ss.chart_container = chart_container

    # if ss.clear_next_run:
    #     ss.map_container.empty()
    #     ss.chart_container.empty()
    #     ss.clear_next_run = False

    with table_container:
        with st.spinner("Loading forecast..."):
            df = get_forecast_for_location("forecast.duckdb", ss.lat, ss.lon)
            df["time"] = pd.to_datetime(
                df["forecast_hour"], format="%Y%m%d%H", utc=True
            )
            hours = df["time"].to_list()
            hours = [t.astimezone(ZoneInfo("America/Sao_Paulo")) for t in hours]
            tcc = df["cloud"].to_numpy()
            temp = (df["temp"] - 273.15).to_numpy()
            dew = (df["dewpoint"] - 273.15).to_numpy()
        summary_df = build_cloud_cover_summary(df)
        st.write(
            summary_df.to_html(escape=False, justify="center", index=False),
            unsafe_allow_html=True,
        )

    with chart_container:
        with st.spinner("Loading forecast..."):
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.04, 0.02, 0.10],
                specs=[
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
                    colorscale=blue_colorscale,
                    zmin=0,
                    zmax=100,
                    colorbar=None,
                    showscale=False,
                    opacity=0.9,
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
                    colorscale=blue_colorscale,
                    zmin=0,
                    zmax=100,
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
                row=3,
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
                row=3,
                col=1,
            )

            time, moon_alt = get_moon_altitudes(
                [hours[0], hours[-1]],
                ss.lat,
                ss.lon,
                timezone=ZoneInfo("America/Sao_Paulo"),
            )
            moon_is_up = np.where(moon_alt > 0, 1, np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=to_2d(moon_is_up),
                    x=time,
                    y=[""],
                    colorscale="Greys_r",
                    zmin=0,
                    zmax=1,
                    colorbar=None,
                    showscale=False,
                    opacity=0.9,
                    hoverinfo="skip",
                    name="Moon",
                ),
                row=2,
                col=1,
            )

            # add vertical lines for now, midnight, sunrise/sunset
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
            max_day_delta = (max_date - now).days + 1
            for day_delta in range(max_day_delta):
                d = now.replace(hour=0, minute=0, second=0) + timedelta(days=day_delta)
                sunset, sunrise = get_sunrise_sunset(
                    d,
                    ss.lat,
                    ss.lon,
                    timezone=ZoneInfo("America/Sao_Paulo"),
                )
                ss_astronomical, sr_astronomical = get_sunrise_sunset(
                    d,
                    ss.lat,
                    ss.lon,
                    timezone=ZoneInfo("America/Sao_Paulo"),
                    type="astronomical",
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
                title_text="Temperature (°C)",
                ticks="",
                showgrid=False,
                showticklabels=False,
                row=3,
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
                    datetime.now() - timedelta(hours=5),
                    datetime.now() + timedelta(days=1),
                ],
            )
            dates = df["time"].dt.date.unique()
            noons = [pd.Timestamp(day) + pd.Timedelta(hours=12) for day in dates]
            ticktexts = [d.strftime("%a %d %b") for d in noons]
            fig.update_layout(
                xaxis4=dict(
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
            fig.data[1].update(xaxis="x4")
            fig.update_xaxes(ticks="", row=2, col=1)
            fig.update_xaxes(showticklabels=False, row=3, col=1)
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

    # --- cloud cover summary table container ---
    # with ss.summary_container:

    # Show last update time
    # with ss.info_container:
    st.markdown(
        f"<small>Latest forecast update (UTC): {update_time.strftime('%Y-%m-%d %H:%M')}</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

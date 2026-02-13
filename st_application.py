import os
import folium
import re
import json
import requests

import time as tm
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_js_eval import streamlit_js_eval

from streamlit_js_eval import get_geolocation
from datetime import datetime, timezone, timedelta
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo

from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from streamlit.components.v1 import html

GOES_COG_DIR = "./goes_cog"
# FORECAST_API_URL = "http://localhost:8001/forecast"
# TILE_URLS = {
#     "cod": "http://localhost:8002/tiles/cod/{timestamp}/{z}/{x}/{y}.png",
#     "acha": "http://localhost:8002/tiles/acha/{timestamp}/{z}/{x}/{y}.png",
#     "acht": "http://localhost:8002/tiles/acht/{timestamp}/{z}/{x}/{y}.png",
# }
FORECAST_API_URL = "https://forecast.astrotempo.org/forecast"
TILE_URLS = {
    "cod": "https://sttiles.astrotempo.org/tiles/cod/{timestamp}/{z}/{x}/{y}.png",
    "acha": "https://sttiles.astrotempo.org/tiles/acha/{timestamp}/{z}/{x}/{y}.png",
    "acht": "https://sttiles.astrotempo.org/tiles/acht/{timestamp}/{z}/{x}/{y}.png",
}
DEFAULT_LAT = -25.427
DEFAULT_LON = 310.758

ss = st.session_state
if "lat" not in ss:
    ss.lat = DEFAULT_LAT
if "lon" not in ss:
    ss.lon = DEFAULT_LON
if "is_first_load" not in ss:
    ss.is_first_load = True
if "location_requested" not in ss:
    ss.location_requested = False
if "location_request_timer" not in ss:
    ss.location_request_timer = None
if "goes_hours_back" not in ss:
    ss.goes_hours_back = 3
if "screen_width" not in ss:
    width = streamlit_js_eval(
        js_expressions="window.innerWidth",
        key="WIDTH",
    )
    if width is None:
        st.stop()
    ss.screen_width = width

if ss.screen_width < 600:
    IS_SCREEN_SMALL = True
else:
    IS_SCREEN_SMALL = False

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


def get_user_location(timeout: float = 10.0):
    info_container = st.container()
    info_container.empty()

    if ss.location_request_timer is None:
        ss.location_request_timer = tm.time()

    loc = get_geolocation()
    if loc is not None:
        ss.lat = loc["coords"]["latitude"]
        ss.lon = loc["coords"]["longitude"] + 360
        ss.location_requested = False
        ss.location_request_timer = None
        st.rerun()
    elapsed_time = tm.time() - ss.location_request_timer
    if elapsed_time < timeout:
        info_container.info("Getting location for forecast...")
        tm.sleep(2)
        st.rerun()
    else:
        info_container.warning(
            "It was not possible to retrieve your location. Using default."
        )
        tm.sleep(3)
        ss.location_requested = False
        ss.location_request_timer = None
        st.rerun()


def get_color_for_value(value: float) -> str:
    """Return appropriate hex color from blue_scale for cloud-cover percentage (0–100)."""
    v = value / 100
    for i in range(len(bounderies_cloud) - 1):
        if bounderies_cloud[i] <= v < bounderies_cloud[i + 1]:
            return blue_scale[i]
    return blue_scale[-1]


@st.cache_data(show_spinner=False)
def build_cloud_cover_summary(df: pd.DataFrame, df_moon: pd.DataFrame) -> pd.DataFrame:
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
        moon_illumination_mode = (
            df_moon[df_moon["time"].dt.date == d]["illumination"].dropna().mode()
        )
        if moon_illumination_mode.shape[0] > 0:
            moon_illumination = moon_illumination_mode[0]
        row["moon"] = f"{moon_illumination:.0f}%"
        rows.append(row)
    rows[0]["day"] = "<b>Today</b>"

    return pd.DataFrame(rows)


def to_2d(values):
    return np.expand_dims(values, axis=0)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def get_formated_address(lat: float, lon: float) -> str:
    geolocator = Nominatim(user_agent="streamlit_reverse_geocoder")
    try:
        location = geolocator.reverse((lat, lon), language="en")
        address = location.raw["address"] if location else None
        return get_address_text(address) if address else None
    except Exception as e:
        # st.error(f"Error during reverse geocoding: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_forecast_data(lat: float, lon: float, update_time: datetime):
    resp = requests.get(
        FORECAST_API_URL,
        params={"lat": lat, "lon": lon},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    st.set_page_config(page_title="Astro Forecast", layout="centered")

    if ss.location_requested or ss.is_first_load:
        if ss.is_first_load:
            ss.is_first_load = False
            ss.location_requested = True
        get_user_location()

    grib_files = os.listdir("latest_wrf_files")
    update_time = get_update_time(grib_files[0])

    try:
        formated_address = get_formated_address(ss.lat, ss.lon)
        if formated_address is not None:
            st.markdown(
                f"<big>**Forecast for** {formated_address}</big>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<big>**Forecast for**</big>",
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

        if st.button("Get my location"):
            ss.location_requested = True
            st.rerun()

        st.markdown(
            f"<small>Latest forecast update (UTC): {update_time.strftime('%Y-%m-%d %H:%M')}</small>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error: {e}")

    m = folium.Map(location=[ss.lat, ss.lon], zoom_start=12, zoom_control=False)

    folium.Marker(
        [ss.lat, ss.lon],
        icon=folium.Icon(color="red"),
    ).add_to(m)

    # Enable clicking
    m.add_child(folium.LatLngPopup())

    location_map_width = "100%" if IS_SCREEN_SMALL else 450
    map_data = st_folium(
        m, width=location_map_width, height=350, returned_objects=["last_clicked"]
    )

    if map_data and map_data.get("last_clicked"):
        ss.lat = map_data["last_clicked"]["lat"]
        ss.lon = map_data["last_clicked"]["lng"]
        st.rerun()

    with st.spinner("Loading forecast data..."):
        data = get_forecast_data(ss.lat, ss.lon, update_time)

        df = pd.DataFrame(data["forecast"])
        df["time"] = pd.to_datetime(df["forecast_hour"], format="%Y%m%d%H", utc=True)
        hours = df["time"].dt.tz_convert(ZoneInfo("America/Sao_Paulo")).to_list()

        tcc = df["cloud"].to_numpy()
        transparency = df["transparency"].to_numpy()
        temp = df["temp"].to_numpy()
        dew = df["dewpoint"].to_numpy()

        df_sun = pd.DataFrame(data["sun"])

        df_moon = pd.DataFrame(data["moon"])
        df_moon["time"] = pd.to_datetime(df_moon["time"], format="%Y-%m-%dT%H:%M:%S%z")

    st.markdown("#### Cloud cover overview")

    with st.spinner("Creating summary table..."):
        summary_df = build_cloud_cover_summary(df, df_moon)
    st.write(
        summary_df.to_html(escape=False, justify="center", index=False),
        unsafe_allow_html=True,
    )

    st.markdown("#### Hourly forecast")

    with st.spinner("Creating forecast chart..."):
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.06, 0.06, 0.06, 0.22],
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
                xgap=1,
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
                xgap=1,
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
                z=to_2d(df_moon["illumination"].to_numpy()),
                x=df_moon["time"].to_numpy(),
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
                xgap=1,
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

        sunrise_times = df_sun["sunrise"]
        sunset_times = df_sun["sunset"]
        sr_astronomical_times = df_sun["astronomical_sunrise"]
        ss_astronomical_times = df_sun["astronomical_sunset"]
        sun_times = zip(
            sunrise_times,
            sunset_times,
            sr_astronomical_times,
            ss_astronomical_times,
        )
        for sunrise, sunset, sr_astronomical, ss_astronomical in sun_times:
            # daytime shading
            fig.add_shape(
                type="rect",
                x0=sunrise,
                x1=sunset,
                y0=-1,
                y1=2.0,
                xref="x",
                yref="paper",
                fillcolor="#605EE4",
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
                y1=2.0,
                xref="x",
                yref="paper",
                fillcolor="#34346E",
                line_width=0,
                opacity=0.5,
                layer="below",
            )
            fig.add_shape(
                type="rect",
                x0=sunset,
                x1=ss_astronomical,
                y0=-1,
                y1=2.0,
                xref="x",
                yref="paper",
                fillcolor="#34346E",
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
        # visible hours is 12 when in mobile phone and 24 in desktop
        VISIBLE_HOURS = 12 if IS_SCREEN_SMALL else 24
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
                datetime.now(tz=ZoneInfo("America/Sao_Paulo"))
                + timedelta(hours=(VISIBLE_HOURS - 3)),
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
            height=350,
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

    st.markdown("#### Satellite view")

    # ------------------------------------------------------------
    # Extract timestamps from filenames
    # ------------------------------------------------------------
    timestamps = []
    for file in sorted(os.listdir(GOES_COG_DIR)):
        parts = file.split("_")
        if len(parts) >= 3:
            ts = parts[1] + "_" + parts[2]  # YYYYMMDD_HHMMSS
            timestamps.append(ts)

    def parse_ts(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)

    dt_list = [parse_ts(ts) for ts in timestamps]

    latest_dt = max(dt_list)
    earliest_dt = min(dt_list)
    max_hours_back = int((latest_dt - earliest_dt).total_seconds() // 3600)

    cutoff_dt = latest_dt - timedelta(hours=int(ss.goes_hours_back))

    filtered = [ts for ts, dt in zip(timestamps, dt_list) if dt >= cutoff_dt]

    if not filtered:
        st.error("No GOES frames found for the selected time range.")
        st.stop()

    slider_max = len(filtered) - 1

    hours_back = st.selectbox(
        label="Hours back from most recent frame:",
        options=[i for i in range(1, max_hours_back + 1)],
        width=220,
        index=(2 if max_hours_back >= 3 else 0),
    )
    if hours_back != ss.goes_hours_back:
        ss.goes_hours_back = hours_back
        st.rerun()

    sat_map_height = 450

    # Satellite map
    leaflet_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8" />
            <meta name="viewport"
                content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />

            <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
            <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>

            <style>
            body {{ margin: 0; padding: 0; }}
            #map {{ width: 100%; max-width: 100vw; height: {sat_map_height}px; }}

            .controls {{
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(255,255,255,0.96);
                padding: 8px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                font-family: sans-serif;
                z-index: 10;
            }}

            .control-row {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            .control-row input[type=range] {{
                flex: 1;
            }}

            .time-label {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.75);
                color: white;
                padding: 6px 10px;
                font-family: monospace;
                font-size: 14px;
                z-index: 10;
            }}

            @media (max-width: 600px) {{
                #map {{ height: 86vh; }}
                .controls {{ padding: 6px; }}
                .time-label {{ font-size: 12px; }} }}
            </style>
            </head>

            <body>

            <div id="map"></div>
            <div class="time-label" id="timeLabel"></div>

            <div class="controls">
                <div class="control-row">
                    <select id="layerSel">
                        <option value="cod">Optical Depth</option>
                        <option value="ctt">Cloud Temp</option>
                        <option value="ch">Cloud Height</option>
                    </select>

                    <input id="opacitySlider" type="range" min="0" max="1" step="0.05" value="0.3" />
                </div>

                <div class="control-row">
                    <button id="playBtn">▶</button>

                    <select id="speedSel">
                        <option value="1600">0.5×</option>
                        <option value="800" selected>1×</option>
                        <option value="400">2×</option>
                        <option value="200">4×</option>
                    </select>

                    <input id="slider" type="range" min="0" max="{slider_max}" step="1" />
                </div>
            </div>

            <script>
            const timestamps = {json.dumps(filtered)};
            const tileTemplates = {{
                cod: "{TILE_URLS['cod']}",
                ch:  "{TILE_URLS['acha']}",
                ctt: "{TILE_URLS['acht']}"
            }};

            let index = {slider_max};
            let activeLayer = "cod";
            let opacity = 0.3;
            let playing = false;
            let frameDelay = 800;
            let timer = null;

            // ---- USER LOCATION ----
            // GOES longitude is 0–360 → convert to -180…180 for MapLibre
            const goesLon = {ss.lon};
            const mapLon  = goesLon > 180 ? goesLon - 360 : goesLon;
            const mapLat  = {ss.lat};

            // UI refs
            const slider = document.getElementById("slider");
            const playBtn = document.getElementById("playBtn");
            const speedSel = document.getElementById("speedSel");
            const opacitySlider = document.getElementById("opacitySlider");
            const layerSel = document.getElementById("layerSel");

            slider.value = index;

            // ---- seed raster source (CRITICAL) ----
            const initialTimestamp = timestamps[index];
            const initialTileURL =
                tileTemplates[activeLayer].replace("{{timestamp}}", initialTimestamp);

            const map = new maplibregl.Map({{
                container: "map",
                style: {{
                    version: 8,
                    sources: {{
                        basemap: {{
                            type: "raster",
                            tiles: [
                                "https://a.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}.png",
                                "https://b.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}.png"
                            ],
                            tileSize: 256,
                            bounds: [
                                -90,
                                -45,
                                -20,
                                20
                            ]
                        }},
                        goes: {{
                            type: "raster",
                            tiles: [initialTileURL],
                            tileSize: 256,
                            bounds: [
                                -90, // west
                                -45, // south
                                -20, // east
                                20   // north
                            ]
                        }},
                        labels: {{
                            type: "raster",
                            tiles: [
                                "https://a.basemaps.cartocdn.com/light_only_labels/{{z}}/{{x}}/{{y}}.png",
                                "https://b.basemaps.cartocdn.com/light_only_labels/{{z}}/{{x}}/{{y}}.png"
                            ],
                            tileSize: 256,
                            bounds: [
                                -90,
                                -45,
                                -20,
                                20
                            ]
                        }}
                    }},
                    layers: [
                        {{ id: "basemap", type: "raster", source: "basemap" }},
                        {{
                            id: "goes",
                            type: "raster",
                            source: "goes",
                            paint: {{
                                "raster-opacity": opacity,
                                "raster-fade-duration": 0,
                                "raster-resampling": "nearest"
                            }}
                        }},
                        {{ id: "labels", type: "raster", source: "labels" }}
                    ]
                }},
                center: [mapLon, mapLat],
                zoom: 5,
                minZoom: 3,
                maxZoom: 9,
                dragRotate: false,
                pitchWithRotate: false
            }});

            function createGeoCircle(center, radiusInMeters, points = 64) {{
                const coords = [];
                const distanceX = radiusInMeters / (111320 * Math.cos(center[1] * Math.PI / 180));
                const distanceY = radiusInMeters / 110540;

                for (let i = 0; i < points; i++) {{
                    const theta = (i / points) * (2 * Math.PI);
                    const x = distanceX * Math.cos(theta);
                    const y = distanceY * Math.sin(theta);

                    coords.push([
                        center[0] + x,
                        center[1] + y
                    ]);
                }}

                coords.push(coords[0]); // close polygon

                return {{
                    type: "Feature",
                    geometry: {{
                        type: "Polygon",
                        coordinates: [coords]
                    }}
                }};
            }}

            const lowClCircle = createGeoCircle([mapLon, mapLat], 5500);
            const midClCircle = createGeoCircle([mapLon, mapLat], 19200);
            const highClCircle = createGeoCircle([mapLon, mapLat], 35500);

            // ---- frame update ----
            function setFrame(i) {{
                index = i;
                const ts = timestamps[index];
                const url = tileTemplates[activeLayer].replace("{{timestamp}}", ts);
                map.getSource("goes").setTiles([url]);
                map.setPaintProperty("goes", "raster-opacity", opacity);
                slider.value = index;
                updateTimeLabel(ts);
            }}

            function updateTimeLabel(ts) {{
                const [d, t] = ts.split("_");
                const utc = new Date(
                    d.slice(0,4)+"-"+d.slice(4,6)+"-"+d.slice(6,8)+"T"+
                    t.slice(0,2)+":"+t.slice(2,4)+":"+t.slice(4,6)+"Z"
                );
                document.getElementById("timeLabel").innerText =
                    utc.toLocaleString("pt-BR", {{
                        timeZone: "America/Sao_Paulo",
                        hour12: false
                    }}) + " (BRT)";
            }}

            // ---- controls wiring ----
            playBtn.onclick = () => {{
                if (playing) {{
                    clearInterval(timer);
                    playing = false;
                    playBtn.innerText = "▶";
                    return;
                }}
                playing = true;
                playBtn.innerText = "⏸";
                timer = setInterval(() => {{
                    setFrame((index + 1) % timestamps.length);
                }}, frameDelay);
            }};

            speedSel.onchange = () => {{
                frameDelay = parseInt(speedSel.value);
                if (playing) {{
                    clearInterval(timer);
                    playBtn.onclick();
                }}
            }};

            opacitySlider.oninput = () => {{
                opacity = parseFloat(opacitySlider.value);
                map.setPaintProperty("goes", "raster-opacity", opacity);
            }};

            slider.oninput = () => {{
                setFrame(parseInt(slider.value));
            }};

            layerSel.onchange = () => {{
                activeLayer = layerSel.value;
                setFrame(index);
            }};

            map.on("load", () => {{
                map.setRenderWorldCopies(false);
                updateTimeLabel(initialTimestamp);
                
                // ----- RANGE CIRCLES ---------
                map.addSource("range-circle-low", {{
                    type: "geojson",
                    data: lowClCircle
                }});
                map.addLayer({{
                    id: "range-circle-low-fill",
                    type: "fill",
                    source: "range-circle-low",
                    paint: {{
                        "fill-color": "#ff0000",
                        "fill-opacity": 0.0
                    }}
                }});
                map.addLayer({{
                    id: "range-circle-outline-low",
                    type: "line",
                    source: "range-circle-low",
                    paint: {{
                        "line-color": "#000000",
                        "line-width": 1.5
                    }}
                }});

                map.addSource("range-circle-mid", {{
                    type: "geojson",
                    data: midClCircle
                }});
                map.addLayer({{
                    id: "range-circle-mid-fill",
                    type: "fill",
                    source: "range-circle-mid",
                    paint: {{
                        "fill-color": "#ff0000",
                        "fill-opacity": 0.0
                    }}
                }});
                map.addLayer({{
                    id: "range-circle-outline-mid",
                    type: "line",
                    source: "range-circle-mid",
                    paint: {{
                        "line-color": "#000000",
                        "line-width": 1.5
                    }}
                }});

                map.addSource("range-circle-high", {{
                    type: "geojson",
                    data: highClCircle
                }});
                map.addLayer({{
                    id: "range-circle-high-fill",
                    type: "fill",
                    source: "range-circle-high",
                    paint: {{
                        "fill-color": "#ff0000",
                        "fill-opacity": 0.0
                    }}
                }});
                map.addLayer({{
                    id: "range-circle-outline-high",
                    type: "line",
                    source: "range-circle-high",
                    paint: {{
                        "line-color": "#000000",
                        "line-width": 1.5
                    }}
                }});

                // ---- USER LOCATION CIRCLE ----
                map.addSource("user-location", {{
                    type: "geojson",
                    data: {{
                        type: "Feature",
                        geometry: {{
                            type: "Point",
                            coordinates: [mapLon, mapLat]
                        }}
                    }}
                }});

                map.addLayer({{
                    id: "user-location-circle",
                    type: "circle",
                    source: "user-location",
                    paint: {{
                        "circle-radius": 2,
                        "circle-color": "#263FCF",
                        "circle-stroke-color": "#263FCF",
                        "circle-stroke-width": 2
                    }}
                }});

            }});
            </script>

            </body>
            </html>
            """
    html(leaflet_html, height=sat_map_height, width="100%")


if __name__ == "__main__":
    main()

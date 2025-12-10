import os
import duckdb
import folium

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


def main():
    st.set_page_config(page_title="Astro Forecast", layout="wide")

    geolocator = Nominatim(user_agent="streamlit_reverse_geocoder")

    # get data
    loc = get_geolocation()
    if loc is None:
        st.info("Requesting location… please allow location permissions.")
        st.stop()
    else:
        target_lat = loc["coords"]["latitude"]
        target_lon = 360 + loc["coords"]["longitude"]
        try:
            location = geolocator.reverse((target_lat, target_lon), language="en")
            print(location.raw)
            address = location.raw["address"] if location else "No address found"
            st.markdown(
                f"Forecast for {address['suburb']}, {address['city']}, {address['state']}"
            )
            st.markdown(f"Lat: {target_lat:.2f} Lon: {target_lon:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
    # --- 2. Create the Folium map centered on user ---
    m = folium.Map(location=[target_lat, target_lon], zoom_start=12)

    # Marker showing user's location
    folium.Marker(
        [target_lat, target_lon], popup="You are here", icon=folium.Icon(color="blue")
    ).add_to(m)

    # # Allow user to click and get coordinates
    # m.add_child(folium.LatLngPopup())

    # # --- 3. Display the map and capture click ---
    # map_data = st_folium(m, width=700, height=500)

    # # --- 4. Show clicked coordinates ---
    # if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
    #     target_lat = map_data["last_clicked"]["lat"]
    #     target_lon = map_data["last_clicked"]["lng"]
    #     st.success(f"Selected Point → Lat: {target_lat:.6f}, Lon: {target_lon:.6f}")

    df = get_forecast_for_location("forecast.duckdb", target_lat, target_lon)
    df["time"] = pd.to_datetime(df["forecast_hour"], format="%Y%m%d%H", utc=True)
    hours = df["time"].to_list()
    hours = [t.astimezone(ZoneInfo("America/Sao_Paulo")) for t in hours]
    tcc = df["cloud"].to_numpy()
    temp = (df["temp"] - 273.15).to_numpy()
    dew = (df["dewpoint"] - 273.15).to_numpy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
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
        target_lat,
        target_lon,
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
            d, target_lat, target_lon, timezone=ZoneInfo("America/Sao_Paulo")
        )
        ss_astronomical, sr_astronomical = get_sunrise_sunset(
            d,
            target_lat,
            target_lon,
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
        title_text="Clouds", ticks="", row=1, col=1, fixedrange=True, showgrid=False
    )
    fig.update_yaxes(
        title_text="Moon", ticks="", row=2, col=1, fixedrange=True, showgrid=False
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
        height=430,
        width=1200,
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

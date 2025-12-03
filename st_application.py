import os

import pygrib as pg
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import astropy.units as u

from datetime import datetime, timezone, timedelta
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_moon


blue_scale = [
    [0.00, "#005bb5"],
    [0.25, "#3f8fdb"],
    [0.50, "#7fb2e6"],
    [0.75, "#b8d4f0"],
    [1.00, "#e8f1fa"],
]
cloud_bins = [0, 20, 40, 60, 80, 100]


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

    # get data
    target_lat, target_lon = -25.39, 310.79
    df = pd.read_csv("extracted_data/extracted_wrf.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    hours = df["time"].to_list()
    hours = [t.astimezone(ZoneInfo("America/Sao_Paulo")) for t in hours]
    tcc = df["total_cloud_cover"].to_numpy()
    temp = df["temperature_C"].to_numpy()
    dew = df["dew_point_C"].to_numpy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.05, 0.03, 0.10],
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
            colorscale="Blues_r",
            zmin=0,
            zmax=100,
            colorbar=None,
            showscale=False,
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
            colorscale="Blues_r",
            zmin=0,
            zmax=100,
            colorbar=None,
            showscale=False,
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
        margin=dict(t=60, b=20),
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

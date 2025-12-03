import os

import pygrib as pg
import pandas as pd
import numpy as np

from datetime import datetime, timezone

target_lat = -25.39
target_lon = 310.79


def interp_value(msg, lat, lon):
    vals, lats, lons = msg.data()
    dist = (lats - lat) ** 2 + (lons - lon) ** 2
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    return vals[i, j]


def main():
    hour = []
    tcc = []
    tmp = []
    dew = []

    forecast_files = sorted(
        [
            os.path.join("./processed_wrf_files", f)
            for f in os.listdir("./processed_wrf_files")
        ]
    )
    for file in forecast_files:
        # print(f"Processing file: {file}")
        try:
            grbs = pg.open(file)

            # Select the messages of interest
            tcc_msg = grbs.select(name="Total Cloud Cover")[0]
            tmp_msg = grbs.select(name="2 metre temperature")[0]
            dew_msg = grbs.select(name="2 metre dewpoint temperature")[0]

            # Extract the value at that grid point
            tcc.append(interp_value(tcc_msg, target_lat, target_lon))
            tmp.append(
                interp_value(tmp_msg, target_lat, target_lon) - 273.15
            )  # Convert from K to °C
            dew.append(interp_value(dew_msg, target_lat, target_lon) - 273.15)
            hour.append(tcc_msg.validDate.replace(tzinfo=timezone.utc))

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    df = pd.DataFrame(
        {
            "time": hour,
            "total_cloud_cover": tcc,
            "temperature_C": tmp,
            "dew_point_C": dew,
        }
    )
    df.to_csv("extracted_data/extracted_wrf.csv", index=False)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Data extraction completed in: {end_time - start_time}")

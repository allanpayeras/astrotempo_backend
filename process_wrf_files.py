import os
import shutil
import pygrib as pg
from datetime import datetime

KEEP_VARS = [
    "Visibility",
    "Wind speed (gust)",
    "2 metre temperature",
    "2 metre relative humidity",
    "2 metre dewpoint temperature",
    "Surface pressure",
    "Total Precipitation",
    "Low cloud cover",
    "Medium cloud cover",
    "High cloud cover",
    "Total Cloud Cover",
]

OUTPUT_FOLDER = "./processed_wrf_files"


def main():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    files = sorted(os.listdir("./latest_wrf_files"))
    for file in files:
        output_file = os.path.join(OUTPUT_FOLDER, f"processed_{file}.grib2")
        try:
            if os.path.exists(output_file):
                print(f"✅ Processed file already exists: {output_file}")
                continue
            with pg.open(os.path.join("./latest_wrf_files", file)) as grbs, open(
                output_file, "wb"
            ) as out_f:
                for grb in grbs:
                    if grb.name in KEEP_VARS:
                        out_f.write(grb.tostring())
            print(f"Processed file saved to: {output_file}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            continue


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Processing completed in: {end - start}")

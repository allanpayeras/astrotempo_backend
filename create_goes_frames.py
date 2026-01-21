import rasterio
import subprocess

import xarray as xr
import numpy as np

from rasterio.crs import CRS
from pathlib import Path
from datetime import datetime, timezone
from goes2go.accessors import rgbAccessor
from rasterio.transform import from_bounds


GOES_DATA_DIR = "./goes_data"
GOES_COG_DIR = "./goes_cog"

# [min_lon, max_lat, max_lon, min_lat]
BRAZIL_EXTENT = [-90.0, 20.0, -20.0, -45.0]


def float_rgb_to_uint8(rgb):
    """
    Convert float32 RGB [0–1] to uint8 [0–255]
    """
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def run_cmd(cmd: list[str]):
    """Run a command-line program and fail loudly if it errors."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{result.stderr}")


def create_goes_cog(goes_nc: Path, output_dir: Path):
    """
    Create a Brazil-cropped, Web Mercator COG from a GOES NetCDF file
    using GDAL-native reprojection.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Timestamp for naming
    # ------------------------------------------------------------------
    ds = xr.open_dataset(goes_nc)

    ts = datetime.strptime(ds.time_coverage_start[:-3], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    base = f"goes_{ts:%Y%m%d_%H%M%S}"
    goes_tif = output_dir / f"{base}.tif"
    warped = output_dir / f"{base}_wrp.tif"
    cog = output_dir / f"{base}_cog.tif"

    if cog.exists():
        print(f"COG exists: {cog.name}")
        return

    print(f"Processing {goes_nc.name}")

    rgb = rgbAccessor(ds).TrueColor().values
    rgb = float_rgb_to_uint8(rgb)
    rows, cols, ndim = rgb.shape

    # Set projection CRS
    lon_origin = ds.goes_imager_projection.attrs["longitude_of_projection_origin"]
    height = ds.goes_imager_projection.attrs["perspective_point_height"]
    semi_major = ds.goes_imager_projection.attrs["semi_major_axis"]
    semi_minor = ds.goes_imager_projection.attrs["semi_minor_axis"]

    crs = CRS.from_proj4(
        f"+proj=geos +lon_0={lon_origin:.0f} +h={height:.0f} "
        f"+a={semi_major:.0f} +b={semi_minor:.5f} +sweep=x +no_defs"
    )

    # set geotransform
    x_min = ds.x_image_bounds[0] * height
    x_max = ds.x_image_bounds[1] * height
    y_min = ds.y_image_bounds[1] * height
    y_max = ds.y_image_bounds[0] * height

    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)

    with rasterio.open(
        goes_tif,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
        photometric="RGB",
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(rgb[:, :, 0], 1)  # Red
        dst.write(rgb[:, :, 1], 2)  # Green
        dst.write(rgb[:, :, 2], 3)  # Blue

    ds.close()

    # Reproject to Web Mercator
    cmd_warp = [
        "gdalwarp",
        "-t_srs",
        "EPSG:3857",
        "-te",
        str(BRAZIL_EXTENT[0]),
        str(BRAZIL_EXTENT[1]),
        str(BRAZIL_EXTENT[2]),
        str(BRAZIL_EXTENT[3]),
        "-te_srs",
        "EPSG:4326",
        "-r",
        "bilinear",
        "-multi",
        str(goes_tif),
        str(warped),
    ]
    run_cmd(cmd_warp)

    # ------------------------------------------------------------------
    # 3. Convert to COG (Google Maps compatible)
    # ------------------------------------------------------------------
    cmd_cog = [
        "gdal_translate",
        str(warped),
        str(cog),
        "-of",
        "COG",
        "-ot",
        "Byte",
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "TILING_SCHEME=GoogleMapsCompatible",
        "-co",
        "BLOCKSIZE=256",
    ]
    run_cmd(cmd_cog)

    goes_tif.unlink()
    warped.unlink()

    print(f"Created COG: {cog.name}")


def main():
    goes_path = Path(GOES_DATA_DIR)
    if not goes_path.exists():
        raise RuntimeError("GOES data directory does not exist.")

    goes_files = sorted(goes_path.glob("**/*.nc"))

    output_dir = Path(GOES_COG_DIR)

    for file in goes_files:
        create_goes_cog(file, output_dir)


if __name__ == "__main__":
    main()

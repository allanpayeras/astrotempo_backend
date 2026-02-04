import rasterio
import subprocess
import matplotlib

import xarray as xr
import numpy as np

from matplotlib import cm, colors
from rasterio.crs import CRS
from pathlib import Path
from datetime import datetime, timezone

# from goes2go.accessors import rgbAccessor
from rasterio.transform import from_bounds


GOES_PRODUCTS = [
    "ABI-L2-COD",  # Cloud Optical Depth
    "ABI-L2-ACHT",  # Cloud Top Temperature
    "ABI-L2-ACHA",  # Cloud Top Height
]
GOES_DATA_DIR = "./goes_data"
GOES_COG_DIR = "./goes_cog"
# [min_lon, max_lat, max_lon, min_lat]
BRAZIL_EXTENT = [-90.0, 20.0, -20.0, -45.0]


def make_image(ds: xr.Dataset, suffix: str) -> np.ndarray:
    if suffix == "cod":
        return cod_to_rgba(ds.COD)
    elif suffix == "acht":
        return ctt_to_rgba(ds.TEMP)
    if suffix == "acha":
        return ch_to_rgba(ds.HT)
    else:
        raise ValueError(f"Unsupported suffix: {suffix}")


def cod_to_rgba(cod: np.ndarray) -> np.ndarray:
    """
    Convert a 2D cloud optical depth array to a classified RGBA image.

    Alpha = 0 for COD <= 0 or NaN
    Alpha = 255 for valid COD
    """
    if cod.ndim != 2:
        raise ValueError("Input array must be 2D (rows, cols)")

    h = np.asarray(cod, dtype=np.float32)

    rgba = np.zeros((*h.shape, 4), dtype=np.uint8)

    valid = np.isfinite(h) & (h > 0)

    good = valid & (h <= 0.16)
    fair = valid & (h > 0.16) & (h <= 0.35)
    poor = valid & (h > 0.35) & (h <= 0.69)
    very_poor = valid & (h > 0.69) & (h <= 1.0)
    blocked = valid & (h > 1.0)

    rgba[good, :3] = (40, 85, 175)
    rgba[fair, :3] = (110, 155, 215)
    rgba[poor, :3] = (205, 150, 90)
    rgba[very_poor, :3] = (180, 115, 70)
    rgba[blocked, :3] = (117, 38, 27)

    # Alpha channel
    rgba[valid, 3] = 255  # visible
    rgba[~valid, 3] = 0  # fully transparent

    return rgba


def ctt_to_rgba(
    cloud_temp_k: np.ndarray, min_temp_c: float = -90.0, max_temp_c: float = 30.0
) -> np.ndarray:
    """
    Convert a 2D cloud top temperature array (Kelvin) to a viridis scale RGBA image.

    Parameters
    ----------
    cloud_temp_k : np.ndarray
        2D array (rows, cols) of cloud top temperature values in Kelvin.
    min_temp_c : float
        Minimum cloud top temperature (Celcius) used as lower bound for cold clouds.
    max_temp_c : float
        Maximum cloud top temperature (Celcius) used as upper bound for warm clouds.

    Returns
    -------
    np.ndarray
        3D uint8 array (rows, cols, 4) representing an RGBA image.
    """
    if cloud_temp_k.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Kelvin → Celsius
    t_celsius = cloud_temp_k - 273.15

    # Mask no-data
    mask = np.isnan(t_celsius)

    # Normalize
    norm = colors.Normalize(vmin=min_temp_c, vmax=max_temp_c, clip=True)

    # Apply colormap (returns float RGBA in [0,1])
    cmap = matplotlib.colormaps["viridis"]
    rgba = cmap(norm(t_celsius))

    # Convert to uint8
    rgba = (rgba * 255).astype(np.uint8)

    # Set alpha channel
    rgba[..., 3] = np.where(mask, 0, 255)

    return rgba


def ch_to_rgba(cloud_height_m: np.ndarray, max_height_m: float = 20000.0) -> np.ndarray:
    """
    Convert a 2D cloud height array (meters) to a classified RGB image.

    Parameters
    ----------
    cloud_height_m : np.ndarray
        2D array (rows, cols) of cloud height values in meters.
    max_height_m : float
        Maximum cloud height (meters) used as upper bound for high clouds.

    Returns
    -------
    np.ndarray
        3D uint8 array (rows, cols, 3) representing an RGB image.
    """
    if cloud_height_m.ndim != 2:
        raise ValueError("Input array must be 2D (rows, cols)")

    h = np.asarray(cloud_height_m, dtype=np.float32)

    # Initialize output as all transparent
    rgba = np.zeros((*h.shape, 4), dtype=np.uint8)

    # Masks
    valid = np.isfinite(h) & (h > 0)
    low = valid & (h < 2000)
    mid = valid & (h >= 2000) & (h < 6000)
    high = valid & (h >= 6000) & (h <= max_height_m)

    # Assign colors
    rgba[low, :3] = (86, 180, 233)  # light blue
    rgba[mid, :3] = (213, 94, 0)  # orange
    rgba[high, :3] = (0, 158, 115)  # green

    # Alpha channel
    rgba[valid, 3] = 255  # visible
    rgba[~valid, 3] = 0  # fully transparent

    return rgba


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


def create_goes_cog(goes_nc: Path, product: str, output_dir: Path):
    """
    Create Brazil-cropped, Web Mercator COG from a GOES NetCDF file
    using GDAL-native reprojection.
    """

    if product == "ABI-L2-COD":
        suffix = "cod"
        resampling_warp = "near"
        resampling_translate = "NEAREST"
    elif product == "ABI-L2-ACHT":
        suffix = "acht"
        resampling_warp = "bilinear"
        resampling_translate = "BILINEAR"
    elif product == "ABI-L2-ACHA":
        suffix = "acha"
        resampling_warp = "near"
        resampling_translate = "NEAREST"
    else:
        raise ValueError(f"Unsupported product: {product}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Timestamp for naming
    # ------------------------------------------------------------------
    ds = xr.open_dataset(goes_nc)

    ts = datetime.strptime(ds.time_coverage_end[:-3], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    base = f"goes_{ts:%Y%m%d_%H%M%S}_{suffix}"
    goes_tif = output_dir / f"{base}.tif"
    warped = output_dir / f"{base}_wrp.tif"
    cog = output_dir / f"{base}_cog.tif"

    if cog.exists():
        print(f"COG exists: {cog.name}")
        return

    print(f"Processing {goes_nc.name}")

    image = make_image(ds, suffix)
    rows, cols, ndim = image.shape

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
        count=4,
        dtype="uint8",
        crs=crs,
        transform=transform,
        photometric="RGB",
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(image[:, :, 0], 1)  # R
        dst.write(image[:, :, 1], 2)  # G
        dst.write(image[:, :, 2], 3)  # B
        dst.write(image[:, :, 3], 4)  # Alpha

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
        # "near",
        resampling_warp,
        # "-multi",
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
        "-co",
        # "RESAMPLING=NEAREST",
        f"RESAMPLING={resampling_translate}",
    ]
    run_cmd(cmd_cog)

    goes_tif.unlink()
    warped.unlink()

    print(f"Created COG: {cog.name}")


def clean_up_cog_dir(goes_files: list[Path], product: str, output_dir: Path):
    """
    Remove COGs that do not have a corresponding GOES file.
    """
    if product == "ABI-L2-COD":
        suffix = "cod"
    elif product == "ABI-L2-ACHT":
        suffix = "acht"
    elif product == "ABI-L2-ACHA":
        suffix = "acha"
    else:
        raise ValueError(f"Unsupported product: {product}")

    keep_cog_files = set()
    for file in goes_files:
        ts = datetime.strptime(file.stem.split("_e")[1][:13], "%Y%j%H%M%S").replace(
            tzinfo=timezone.utc
        )
        base = f"goes_{ts:%Y%m%d_%H%M%S}_{suffix}_cog.tif"
        keep_cog_files.add(base)

    # all_cog_files = sorted(output_dir.glob("goes_*_cth_cog.tif"))
    all_cog_files = sorted(output_dir.glob(f"goes_*_{suffix}_cog.tif"))
    for cog_file in all_cog_files:
        if cog_file.name not in keep_cog_files:
            print(f"Removing obsolete COG: {cog_file.name}")
            cog_file.unlink()


def main():
    goes_path = Path(GOES_DATA_DIR)
    if not goes_path.exists():
        raise RuntimeError("GOES data directory does not exist.")

    for product in GOES_PRODUCTS:
        print(f"Processing product: {product}")
        goes_files = sorted(list(goes_path.glob(f"**/*{product}*.nc")))

        output_dir = Path(GOES_COG_DIR)

        for file in goes_files:
            create_goes_cog(file, product, output_dir)

        clean_up_cog_dir(goes_files, product, output_dir)


if __name__ == "__main__":
    main()

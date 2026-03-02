import rasterio
import subprocess
import matplotlib
import shutil
import tempfile
import os

import xarray as xr
import numpy as np
import matplotlib.colors as mcolors

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
TILE_ROOT = Path("/var/www/tiles")
# [min_lon, min_lat, max_lon, max_lat]
BRAZIL_EXTENT = [-90.0, -45.0, -20.0, 20.0]
ZOOM_RANGE = "0-8"


def make_image(ds: xr.Dataset, suffix: str) -> np.ndarray:
    if suffix == "cod":
        return cod_to_rgba(ds.COD)
    elif suffix == "acht":
        return ctt_to_rgba(ds.TEMP)
    if suffix == "acha":
        return ch_to_rgba(ds.HT)
    else:
        raise ValueError(f"Unsupported suffix: {suffix}")


def cod_to_rgba(
    cod: np.ndarray,
    min_log_cod: float = -2.0,
    transition_log_cod: float = 0.176,
    max_log_cod: float = 2.0,
) -> np.ndarray:
    """
    Convert a 2D cloud optical depth array to a viridis scale RGBA image.

    Parameters
    ----------
    cod : np.ndarray
        2D array (rows, cols) of cloud optical depth values.
    min_log_cod : float
        Minimum cloud optical depth (Celcius) used as lower bound for cold clouds.
    max_log_cod : float
        Maximum cloud optical depth (Celcius) used as upper bound for warm clouds.
    Returns
    -------
    np.ndarray
        3D uint8 array (rows, cols, 4) representing an RGBA image.
    """
    if cod.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Mask no-data
    mask = np.isnan(cod) | (cod <= 0)

    cod = np.clip(cod, 10**min_log_cod, 10**max_log_cod)
    cod = np.log10(cod)

    # Get base colormaps
    blues = matplotlib.colormaps["Blues_r"](np.linspace(0.0, 0.5, 128))
    reds = matplotlib.colormaps["OrRd"](np.linspace(0.5, 1, 128))

    # Combine them
    colors = np.vstack((blues, reds))  # Dark → light blue -> Light → dark red
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("CloudOpticalDepth", colors)

    # Define normalization with a transition point
    norm = mcolors.TwoSlopeNorm(
        vmin=min_log_cod, vcenter=transition_log_cod, vmax=max_log_cod
    )

    # Apply colormap (returns float RGBA in [0,1])
    rgba = custom_cmap(norm(cod))

    # Convert to uint8
    rgba = (rgba * 255).astype(np.uint8)

    # Set alpha channel
    rgba[..., 3] = np.where(mask, 0, 255)

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
    Convert a 2D cloud height array (meters) to a continuous RGBA image
    using matplotlib Oranges, Greens, and Purples colormaps.
    """

    if cloud_height_m.ndim != 2:
        raise ValueError("Input array must be 2D (rows, cols)")

    cth = np.asarray(cloud_height_m, dtype=np.float32)

    rgba = np.zeros((*cth.shape, 4), dtype=np.uint8)

    valid = np.isfinite(cth) & (cth > 0)

    low = valid & (cth < 2000)
    mid = valid & (cth >= 2000) & (cth < 7000)
    high = valid & (cth >= 7000) & (cth <= max_height_m)

    # Use matplotlib colormaps
    cmap_low = matplotlib.colormaps["Oranges"]
    cmap_mid = matplotlib.colormaps["Greens"]
    cmap_high = matplotlib.colormaps["Purples"]

    # Controlled sampling range (avoids washed-out tones)
    cmin, cmax = 0.4, 0.75

    # LOW CLOUDS
    if np.any(low):
        t = cth[low] / 2000.0
        t = cmin + (cmax - cmin) * t
        colors = cmap_low(t)
        rgba[low] = (colors * 255).astype(np.uint8)

    # MID CLOUDS
    if np.any(mid):
        t = (cth[mid] - 2000.0) / (7000.0 - 2000.0)
        t = cmin + (cmax - cmin) * t
        colors = cmap_mid(t)
        rgba[mid] = (colors * 255).astype(np.uint8)

    # HIGH CLOUDS
    if np.any(high):
        t = (cth[high] - 7000.0) / (max_height_m - 7000.0)
        t = np.clip(t, 0, 1)
        t = cmin + (cmax - cmin) * t
        colors = cmap_high(t)
        rgba[high] = (colors * 255).astype(np.uint8)

    # Transparency
    rgba[~valid, 3] = 0

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
    # Timestamp for naming
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
        "-multi",
        "-srcalpha",
        "-dstalpha",
        str(goes_tif),
        str(warped),
    ]
    run_cmd(cmd_warp)

    # ------------------------------------------------------------------
    # Convert to COG (Google Maps compatible)
    # ------------------------------------------------------------------
    cmd_cog = [
        "gdal_translate",
        str(warped),
        str(cog),
        "-of",
        "COG",
        "-b",
        "1",
        "-b",
        "2",
        "-b",
        "3",
        "-b",
        "4",
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
        "-co",
        "OVERVIEWS=AUTO",
    ]
    run_cmd(cmd_cog)

    # generate static tiles
    # ------------------------------------------------------------------
    # Generate tiles safely in temporary location
    # ------------------------------------------------------------------

    parts = base.split("_")
    timestamp = parts[1] + "_" + parts[2]
    product = parts[3]

    final_dir = TILE_ROOT / product / timestamp
    parent_dir = TILE_ROOT / product
    parent_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating tiles for {cog.name}")

    # Create temp directory on SAME filesystem to ensure os.replace is atomic
    with tempfile.TemporaryDirectory(dir=parent_dir, prefix="tmp_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        result = subprocess.run(
            [
                "gdal2tiles.py",
                "-z",
                ZOOM_RANGE,
                "--xyz",
                "--processes=4",
                "-w",
                "none",
                str(cog),
                str(tmp_path),
            ]
        )

        if result.returncode != 0:
            raise RuntimeError("Tile generation failed")

        # Remove old version if exists
        if final_dir.exists():
            shutil.rmtree(final_dir)

        # Atomic rename (same filesystem)
        os.replace(tmp_path, final_dir)

    print(f"Tiles ready: {final_dir}")

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

    all_cog_files = sorted(output_dir.glob(f"goes_*_{suffix}_cog.tif"))
    for cog_file in all_cog_files:
        if cog_file.name not in keep_cog_files:
            print(f"Removing obsolete COG: {cog_file.name}")
            cog_file.unlink()


def clean_up_tiles_dir(goes_files, product):
    if product == "ABI-L2-COD":
        suffix = "cod"
    elif product == "ABI-L2-ACHT":
        suffix = "acht"
    elif product == "ABI-L2-ACHA":
        suffix = "acha"
    else:
        raise ValueError(f"Unsupported product: {product}")

    keep_tile_dirs = set()
    for file in goes_files:
        ts = datetime.strptime(file.stem.split("_e")[1][:13], "%Y%j%H%M%S").replace(
            tzinfo=timezone.utc
        )
        dir_path = TILE_ROOT / suffix / f"{ts:%Y%m%d_%H%M%S}"
        keep_tile_dirs.add(dir_path)

    for product_dir in (TILE_ROOT / suffix).iterdir():
        if product_dir.is_dir() and product_dir not in keep_tile_dirs:
            print(f"Removing obsolete tile folder: {product_dir}")
            shutil.rmtree(product_dir)


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
        clean_up_tiles_dir(goes_files, product)


if __name__ == "__main__":
    main()

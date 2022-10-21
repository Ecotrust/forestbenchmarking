# %%
from pathlib import Path
import os
import sys
from datetime import datetime

import numpy as np
import geopandas as gpd
import ee

root = [p for p in Path(__file__).parents if p.name == "forestbenchmarking"][0]
sys.path.append(os.path.abspath(root / "src/data"))

from utils import create_directory_tree
from gee_utils import multithreaded_download, GEEImageLoader

YEAR = 2011
QQ_SHP = "oregon_quarter_quads_sample.geojson"
WORKERS = 20

# %%
def get_gflandsat(
    bbox,
    year,
    path,
    prefix=None,
    season="leafon",
    epsg=4326,
    scale=30,
):
    """
    Fetch Gap-Filled Landsat (GFL) image url from Google Earth Engine (GEE) using a bounding box.

    See https://www.ntsg.umt.edu/project/landsat/landsat-gapfill-reflect.php for more information.

    Parameters
    ----------
    month : int
        Month of year (1-12)
    year : int
        Year (e.g. 2019)
    bbox : list
        Bounding box in the form [xmin, ymin, xmax, ymax].

    Returns
    -------
    url : str
        GEE generated URL from which the raster will be downloaded.
    metadata : dict
        Image metadata.
    """
    if season == "leafoff":
        start_date = f"{year - 1}-10-01"
        end_date = f"{year}-03-31"
    elif season == "leafon":
        start_date = f"{year}-04-01"
        end_date = f"{year}-09-30"
    else:
        raise ValueError(f"Invalid season: {season}")

    collection = ee.ImageCollection("projects/KalmanGFwork/GFLandsat_V1").filterDate(
        start_date, end_date
    )

    ts_start = datetime.timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    ts_end = datetime.timestamp(datetime.strptime(end_date, "%Y-%m-%d"))

    bbox = ee.Geometry.BBox(*bbox)
    image = GEEImageLoader(collection.median().clip(bbox))
    # Set image metadata and params
    image.metadata_from_collection(collection)
    image.set_property("system:time_start", ts_start * 1000)
    image.set_property("system:time_end", ts_end * 1000)
    image.set_params("scale", scale)
    image.set_params("crs", f"EPSG:{epsg}")
    image.set_params("region", bbox)
    image.set_viz_params("min", 0)
    image.set_viz_params("max", 2000)
    image.set_viz_params("bands", ["B3_mean_post", "B2_mean_post", "B1_mean_post"])
    image.id = f"{prefix}Gap_Filled_Landsat_CONUS_{year}_{season}"

    # Download cog
    out_path = path / image.id
    out_path.mkdir(parents=True, exist_ok=True)

    image.save_metadata(out_path)
    image.to_geotif(out_path)
    image.save_preview(out_path)


if __name__ == "__main__":

    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # %%
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
    # %%
    # Load the USGS QQ shapefile for Oregon state.
    qq_shp = gpd.read_file(root / "data/external/usfs_stands" / QQ_SHP)
    gfl_path = create_directory_tree(root, f"data/processed", "GFLandsat_V1", str(YEAR))

    # Select a subset of qq cells to play with.
    # qq_shp = qq_shp[qq_shp.CELL_ID.isin(qq_shp.head(20).CELL_ID)].copy()
    params = np.array(
        [
            [geo.bounds for geo in qq_shp.geometry],
            [YEAR for _ in qq_shp.CELL_ID],
            [gfl_path for _ in qq_shp.CELL_ID],
            [f"{id}_" for id in qq_shp.CELL_ID],
            ["leafon" for _ in qq_shp.CELL_ID],
        ],
        dtype=object,
    ).T

    multithreaded_download(params, get_gflandsat)
    
    params[params == 'leafon'] = 'leafoff'
    multithreaded_download(params, get_gflandsat)

# %%
import concurrent.futures as cf
from pathlib import Path
import os
import sys
import json

import numpy as np
import geopandas as gpd
import ee 

root = [p for p in Path(__file__).parents if p.name == 'forestbenchmarking'][0] 
sys.path.append(os.path.abspath(root / 'src/data'))

from gee_utils import (
    get_landtrendr_download_url,
    download_from_url
)

MONTH = 3
YEAR = 2019
QQ_SHP = 'oregon_quarter_quads_sample.geojson'
WORKERS = 5

# %%
def create_directory_tree(*args):
    """Set the directory tree for the dataset.

    Parameters
    ----------
    args : str
        The directory names to be created.

    Returns
    -------
        PosixPath object  
    """
    filepath = Path(*args)

    if not filepath.exists():
        os.makedirs(filepath)
    
    return filepath

# %%
def download_landtrendr_img(bbox, year, path, thumbnail=False, prefix=None, save_metadata=True, epsg=4326, scale=30):
    """Downloads an 8-band raster generated by executing the LandTrendr
    algorithm on Google Earth Engine analyzing Landsat imagery from 1984 to
    the user-specified year.

    Parameters
    ----------
    bbox : list-like
      (xmin, ymin, xmax, ymax) coordinates for bounding box
    year : int
      year up to which Landtrendr time series will be calculated
    path : str
      path to directory where image will be saved
    prefix : str
      prefix to add to filename
    epsg : int
      EPSG code used to define projection
    scale : int
        scale of image in meters
    """
    url, metadata = get_landtrendr_download_url(bbox, year, False, epsg, scale)

    # Rename file to include prefix and file extension.
    img_name = f"{prefix}LandTrendr_8B_SWIR1-NBR_{year}-cog.tif"
    path = (path / img_name.replace('-cog.tif', ''))
    path.mkdir(parents=True, exist_ok=True)

    # Download cog image.
    download_from_url(url, img_name, False, path)  

    # Save metadata as json.
    filepath = path / f"{img_name.replace('-cog.tif', '-metadata.json')}"
    if save_metadata:
        with open(filepath, 'w') as f:
            f.write(json.dumps(metadata, indent=4))

    # Download thumbnail
    if thumbnail:
        url, _ = get_landtrendr_download_url(bbox, year, thumbnail)
        img_name = f"{prefix}LandTrendr_8B_SWIR1-NBR_{year}-sample.png"
        download_from_url(url, img_name, thumbnail, path)

# %%
def main():
    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # %%
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    ltr_path = create_directory_tree(root, 'data/raw', 'LandTrendr')

    # %%
    # Load the USGS QQ shapefile for Oregon state.
    qq_shp = gpd.read_file(root / 'data/external/usfs_stands' / QQ_SHP).to_crs('epsg:4326')
    
    # Select only 5 qq cells to play with.
    qq_shp = qq_shp[qq_shp.CELL_ID.isin(qq_shp.head(20).CELL_ID)]

    params = np.array([
        [geo.bounds for geo in qq_shp.geometry],
        [2019 for _ in qq_shp.CELL_ID],
        [ltr_path for _ in qq_shp.CELL_ID],
        [True for _ in qq_shp.CELL_ID],
        [f"{cell_id}_" for cell_id in qq_shp.CELL_ID]
    ], dtype=object).T

    # %%
    # for chunk in np.array_split(params, -(qq_shp.shape[0] // -WORKERS)):

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        
        # Fetch gee image url and metadata for each qq cell.
        future_url = {executor.submit(download_landtrendr_img, *p): p for p in params}

        for future in cf.as_completed(future_url):
            bbox = future_url[future]

            try:
                future.result()

            except Exception as exc:
                print('%r generated an exception: %s' % (bbox, exc))

if __name__ == '__main__':
    main()

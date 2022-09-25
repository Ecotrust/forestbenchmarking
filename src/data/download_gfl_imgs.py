# %%
import concurrent.futures as cf
from pathlib import Path
import os
import sys

import numpy as np
import geopandas as gpd
import ee

root = [p for p in Path(__file__).parents if p.name == 'forestbenchmarking'][0] 
sys.path.append(os.path.abspath(root / 'src/data'))

from gee_utils import (   
    download_gflandsat_img
)

MONTH = 3
YEAR = 2019
QQ_SHP = 'oregon_quarter_quads_sample.geojson'
WORKERS = 20

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
def main():
    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # %%
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    gfl_path = create_directory_tree(root, 'data/raw', 'GFLandsat_V2')

    # %%
    # Load the USGS QQ shapefile for Oregon state.
    qq_shp = gpd.read_file(root / 'data/external/usfs_stands' / QQ_SHP).to_crs('epsg:4326')
    
    # Select a subset of qq cells to play with.
    qq_shp = qq_shp[qq_shp.CELL_ID.isin(qq_shp.head(20).CELL_ID)].copy()

    params = np.array([
        [geo.bounds for geo in qq_shp.geometry],
        [3 for _ in qq_shp.CELL_ID],
        [2019 for _ in qq_shp.CELL_ID],
        [gfl_path for _ in qq_shp.CELL_ID],
        [True for _ in qq_shp.CELL_ID],
        [f'{id}_' for id in qq_shp.CELL_ID],
        [True for _ in qq_shp.CELL_ID]
    ], dtype=object).T

    # for chunk in np.array_split(params, -(qq_shp.shape[0] // -WORKERS)):

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        
        # Fetch gee image url and metadata for each qq cell.
        future_url = {executor.submit(download_gflandsat_img, *p): p for p in params}

        for future in cf.as_completed(future_url):
            bbox = future_url[future]

            try:
                future.result()

            except Exception as exc:
                print('%r generated an exception: %s' % (bbox, exc))

if __name__ == '__main__':
    main()

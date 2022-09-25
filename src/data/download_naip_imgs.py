"""
Class GEEImageLoader(ee.Image)

def __init__(image_id, bbox)
    self.bbox

def url():
    pass

- Set bbox
- 
"""

import os
import glob
import sys
from pathlib import Path

import ee 
import geopandas as gpd

root = [p for p in Path(__file__).parents if p.name == 'forestbenchmarking'][0] 
sys.path.append(os.path.abspath(root / 'src/data'))

from utils import split_bbox
from gee_utils import multithreaded_download


def get_naip(gdf, state, year, out_dir):
    """Iterates through features in a GeoDataFrame and exports NAIP images that 
    include the centroid of each feature.
    """
    print(f'Preparing download URLs for {len(gdf)} tiles for year {year}')
    to_download = []
    cell_ids = gdf['CELL_ID'].astype(str).values
    out_path = os.path.join(out_dir, 'NAIP', str(year))
    already_done = [
        os.path.basename(x)[:-6] 
        for x in glob.glob(f'{out_dir}/*') 
        if os.path.basename(x).split('_')[0] in cell_ids
    ]
    print(f'Existing tiles for {year}: {len(already_done)}')
    
    for idx, row in gdf.iterrows():
        cell_id = str(row['CELL_ID'])
        outfile = f'{cell_id}_{state}_NAIP_{year}'
        geom = ee.Geometry.Rectangle(row['geometry'].bounds)

        if len([f for f in already_done if f == outfile]) < 6:
            # get the tile bounding box for filtering images
            bbox = row['geometry'].bounds
            aoi_bboxes = split_bbox(3, bbox)
            aois = [
                ee.Geometry.Rectangle(
                    box.tolist(),
                    proj=f'EPSG:{gdf.crs.to_epsg()}',
                    evenOdd=True,
                    geodesic=True
                ) for box in aoi_bboxes
            ]

            # get the naip image collection for our aoi and timeframe
            collection = ee.ImageCollection('USDA/NAIP/DOQQ')\
                           .filterBounds(geom)\
                           .filterDate(f'{year}-01-01', f'{year}-12-31')
            coll_size = collection.size().getInfo()
            image = collection.mosaic()

            url_params = dict(
                name='',
                filePerBand=False,
                scale=1,
                crs=f'EPSG:{gdf.crs.to_epsg()}',
                formatOptions={'cloudOptimized':True}
            )

            # if there are no naip images in this timeframe
            if coll_size == 0:
                print(f'No NAIP images found for {cell_id} in {year}.')

            # if collection is not empty, collect image url and other info.
            if coll_size > 0:
                for tile, aoi in zip(range(1, len(aois) + 1), aois):
                    out_tile = f'{outfile}_{tile}'
                    url_params['name'] = out_tile
                    url = image.clip(aoi).getDownloadURL(url_params)
                    to_download.append((url, out_tile + '.tif', out_path))

        # report progress
        if idx % 100 == 0 and idx > 0:
            print()
        if (idx+1) % 10 == 0:
            print('{:,d}'.format(idx+1), end='')
        else:
            print('.', end='')

    return to_download

if "__main__" == __name__:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    SHP_DIR = 'data/external/usfs_stands/'
    OUT_DIR = 'data/raw/'

    QQ_SHP = 'oregon_quarter_quads_sample.geojson'
    STATE = 'OR'
    YEARS = [2009, 2011, 2015, 2017, 2019]

    qq_shp = gpd.read_file(root / SHP_DIR / QQ_SHP)#.to_crs('EPSG:4326')

    for year in YEARS:
        to_download = get_naip(qq_shp, STATE, year, root / OUT_DIR)
        if len(to_download) > 0:
            multithreaded_download(to_download)
 
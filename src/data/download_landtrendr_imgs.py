# %%
from pathlib import Path
import os
import sys

import numpy as np
import geopandas as gpd
import ee

root = [p for p in Path(__file__).parents if p.name == "forestbenchmarking"][0]
sys.path.append(os.path.abspath(root / "src/data"))

from utils import create_directory_tree
from gee_utils import GEEImageLoader, get_landsat_collection, multithreaded_download

YEAR = 2011
QQ_SHP = "oregon_quarter_quads_sample.geojson"
WORKERS = 5


def parse_landtrendr_result(
    lt_result, current_year, flip_disturbance=False, big_fast=False, sieve=False
):
    """Parses a LandTrendr segmentation result, returning an image that
    identifies the years since the largest disturbance.

    Parameters
    ----------
    lt_result : image
      result of running ee.Algorithms.TemporalSegmentation.LandTrendr on an
      image collection
    current_year : int
       used to calculate years since disturbance
    flip_disturbance: bool
      whether to flip the sign of the change in spectral change so that
      disturbances are indicated by increasing reflectance
    big_fast : bool
      consider only big and fast disturbances
    sieve : bool
      filter out disturbances that did not affect more than 11 connected pixels
      in the year of disturbance

    Returns
    -------
    img : image
      an image with four bands:
        ysd - years since largest spectral change detected
        mag - magnitude of the change
        dur - duration of the change
        rate - rate of change
    """
    lt = lt_result.select("LandTrendr")
    is_vertex = lt.arraySlice(0, 3, 4)  # 'Is Vertex' row - yes(1)/no(0)
    verts = lt.arrayMask(is_vertex)  # vertices as boolean mask

    left, right = verts.arraySlice(1, 0, -1), verts.arraySlice(1, 1, None)
    start_yr, end_yr = left.arraySlice(0, 0, 1), right.arraySlice(0, 0, 1)
    start_val, end_val = left.arraySlice(0, 2, 3), right.arraySlice(0, 2, 3)

    ysd = start_yr.subtract(current_year - 1).multiply(-1)  # time since vertex
    dur = end_yr.subtract(start_yr)  # duration of change
    if flip_disturbance:
        mag = end_val.subtract(start_val).multiply(-1)  # magnitude of change
    else:
        mag = end_val.subtract(start_val)

    rate = mag.divide(dur)  # rate of change

    # combine segments in the timeseries
    seg_info = ee.Image.cat([ysd, mag, dur, rate]).toArray(0).mask(is_vertex.mask())

    # sort by magnitude of disturbance
    sort_by_this = seg_info.arraySlice(0, 1, 2).toArray(0)
    seg_info_sorted = seg_info.arraySort(
        sort_by_this.multiply(-1)
    )  # flip to sort in descending order
    biggest_loss = seg_info_sorted.arraySlice(1, 0, 1)

    img = ee.Image.cat(
        biggest_loss.arraySlice(0, 0, 1).arrayProject([1]).arrayFlatten([["ysd"]]),
        biggest_loss.arraySlice(0, 1, 2).arrayProject([1]).arrayFlatten([["mag"]]),
        biggest_loss.arraySlice(0, 2, 3).arrayProject([1]).arrayFlatten([["dur"]]),
        biggest_loss.arraySlice(0, 3, 4).arrayProject([1]).arrayFlatten([["rate"]]),
    )

    if big_fast:
        # get disturbances larger than 100 and less than 4 years in duration
        dist_mask = img.select(["mag"]).gt(100).And(img.select(["dur"]).lt(4))

        img = img.mask(dist_mask)

    if sieve:
        MAX_SIZE = 128  #  maximum map unit size in pixels
        # group adjacent pixels with disturbance in same year
        # create a mask identifying clumps larger than 11 pixels
        mmu_patches = (
            img.int16().select(["ysd"]).connectedPixelCount(MAX_SIZE, True).gte(11)
        )

        img = img.updateMask(mmu_patches)

    return img.round().toShort()


# %%
def get_landtrendr(bbox, year, path, prefix=None, epsg=4326, scale=30):

    aoi = ee.Geometry.Rectangle(bbox, proj=f"EPSG:{epsg}", evenOdd=True, geodesic=False)
    swir_coll = get_landsat_collection(aoi, 1984, year, band="SWIR1")
    nbr_coll = get_landsat_collection(aoi, 1984, year, band="NBR")

    LT_PARAMS = {
        "maxSegments": 6,
        "spikeThreshold": 0.9,
        "vertexCountOvershoot": 3,
        "preventOneYearRecovery": True,
        "recoveryThreshold": 0.25,
        "pvalThreshold": 0.05,
        "bestModelProportion": 0.75,
        "minObservationsNeeded": 6,
    }

    swir_result = ee.Algorithms.TemporalSegmentation.LandTrendr(swir_coll, **LT_PARAMS)
    nbr_result = ee.Algorithms.TemporalSegmentation.LandTrendr(nbr_coll, **LT_PARAMS)

    swir_img = parse_landtrendr_result(swir_result, year).set(
        "system:time_start", swir_coll.first().get("system:time_start")
    )
    nbr_img = parse_landtrendr_result(nbr_result, year, flip_disturbance=True).set(
        "system:time_start", nbr_coll.first().get("system:time_start")
    )

    lt_img = ee.Image.cat(
        swir_img.select(["ysd"], ["ysd_swir1"]),
        swir_img.select(["mag"], ["mag_swir1"]),
        swir_img.select(["dur"], ["dur_swir1"]),
        swir_img.select(["rate"], ["rate_swir1"]),
        nbr_img.select(["ysd"], ["ysd_nbr"]),
        nbr_img.select(["mag"], ["mag_nbr"]),
        nbr_img.select(["dur"], ["dur_nbr"]),
        nbr_img.select(["rate"], ["rate_nbr"]),
    ).set("system:time_start", swir_img.get("system:time_start"))

    image = GEEImageLoader(lt_img.clip(aoi))
    # Set image metadata and params
    image.metadata_from_collection(nbr_coll)
    image.set_params("scale", scale)
    image.set_params("crs", f"EPSG:{epsg}")
    image.set_viz_params("min", 200)
    image.set_viz_params("max", 800)
    image.set_viz_params("bands", ["mag_swir1"])
    image.set_viz_params(
        "palette",
        [
            "#9400D3",
            "#4B0082",
            "#0000FF",
            "#00FF00",
            "#FFFF00",
            "#FF7F00",
            "#FF0000",
        ],
    )
    image.id = f"{prefix}LandTrendr_8B_SWIR1-NBR_{year}"

    # Rename file to include prefix and file extension.
    out_path = path / image.id
    out_path.mkdir(parents=True, exist_ok=True)

    image.save_metadata(out_path)
    image.to_geotif(out_path)
    image.save_preview(out_path, overwrite=True)


# %%
if __name__ == "__main__":
    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # %%
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

    ltr_path = create_directory_tree(root, "data/processed", "LandTrendr", str(YEAR))

    # %%
    # Load the USGS QQ shapefile for Oregon state.
    qq_shp = gpd.read_file(root / "data/external/usfs_stands" / QQ_SHP)

    # Select only 5 qq cells to play with.
    # qq_shp = qq_shp[qq_shp.CELL_ID.isin(qq_shp.head(20).CELL_ID)]

    params = np.array(
        [
            [geo.bounds for geo in qq_shp.geometry],
            [YEAR for _ in qq_shp.CELL_ID],
            [ltr_path for _ in qq_shp.CELL_ID],
            [f"{cell_id}_" for cell_id in qq_shp.CELL_ID],
        ],
        dtype=object,
    ).T

    # %%
    # for chunk in np.array_split(params, -(qq_shp.shape[0] // -WORKERS)):
    multithreaded_download(params, get_landtrendr)

# Description: Downloads NAIP imagery for Oregon quarter quads

import os
import glob
import sys
from pathlib import Path

import ee
import geopandas as gpd

root = [p for p in Path(__file__).parents if p.name == "forestbenchmarking"][0]
sys.path.append(os.path.abspath(root / "src/data"))

from utils import split_bbox, create_directory_tree
from gee_utils import multithreaded_download, GEEImageLoader, download_from_url

SHP_DIR = "data/external/usfs_stands/"
QQ_SHP = "oregon_quarter_quads_sample.geojson"
OUT_DIR = "data/raw/"
STATE = "OR"
YEARS = [2011]


def get_naip(gdf, state, year, out_dir):
    """Iterates through features in a GeoDataFrame and exports NAIP images that
    include the centroid of each feature.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
    state : str
        Two-letter state abbreviation
    year : int
        Year of NAIP imagery
    out_dir : pathlib.Path
        Path to directory where images will be saved
    """
    print(f"Preparing download URLs for {len(gdf)} tiles for year {year}")
    to_download = []
    cell_ids = gdf["CELL_ID"].astype(str).values
    already_done = [
        os.path.basename(x)[:-6]
        for x in glob.glob(f"{out_dir}/*")
        if os.path.basename(x).split("_")[0] in cell_ids
    ]
    print(f"Existing tiles for {year}: {len(already_done)}")

    prev_to_download = []
    for idx, row in gdf.iterrows():
        cell_id = str(row["CELL_ID"])
        outfile = f"{cell_id}_{state}_NAIP_{year}"
        geom = ee.Geometry.Rectangle(row["geometry"].bounds)

        if len([f for f in already_done if f == outfile]) < 6:
            # get the tile bounding box for filtering images
            bbox = row["geometry"].bounds
            aoi_bboxes = split_bbox(3, bbox)
            aois = [
                ee.Geometry.Rectangle(
                    box.tolist(),
                    proj=f"EPSG:{gdf.crs.to_epsg()}",
                    evenOdd=True,
                    geodesic=True,
                )
                for box in aoi_bboxes
            ]

            # get the naip image collection for our aoi and timeframe
            collection = (
                ee.ImageCollection("USDA/NAIP/DOQQ")
                .filterBounds(geom)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
            )
            coll_size = collection.size().getInfo()
            image = collection.select("R", "G", "B").mosaic()

            img = GEEImageLoader(image)
            img.metadata_from_collection(collection)
            img.id = f"{cell_id}_{state}_NAIP_DOQQ_{year}"
            img.set_params("crs", f"EPSG:{gdf.crs.to_epsg()}")
            img.set_params("region", geom)
            img.set_viz_params("min", 0)
            img.set_viz_params("max", 255)
            img.set_viz_params("bands", ["R", "G", "B"])

            # if there are no naip images in this timeframe
            if coll_size == 0:
                print(f"No NAIP images found for {cell_id} in {year}.")

            # if collection is not empty, collect image url and other info.
            if coll_size > 0:
                out_path = create_directory_tree(out_dir / "NAIP" / str(year) / img.id)
                # Save metadata and img preview
                img.save_metadata(out_path)
                # set low res to generate preview
                img.set_params("scale", 30)
                prev_to_download.append(
                    (img.get_url(preview=True), f"{img.id}-preview.png", out_path, True)
                )

                # set the scale back to 1 for the full res download
                img.set_params("scale", 1)
                for tile, aoi in zip(range(1, len(aois) + 1), aois):
                    out_tile = f"{img.id}_{tile}"
                    img.set_params("name", out_tile)
                    img.set_params("scale", 1)
                    img.set_params("region", aoi)
                    img.image = image.clip(aoi)
                    to_download.append((img.get_url(), f"{out_tile}.tif", out_path))

        # report progress
        if idx % 100 == 0 and idx > 0:
            print()
        if (idx + 1) % 10 == 0:
            print("{:,d}".format(idx + 1), end="")
        else:
            print(".", end="")

    return to_download, prev_to_download


if "__main__" == __name__:
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

    qq_shp = gpd.read_file(root / SHP_DIR / QQ_SHP)

    for year in YEARS:
        to_download, prev_to_download = get_naip(qq_shp, STATE, year, root / OUT_DIR)
        if len(to_download) > 0:
            multithreaded_download(to_download, download_from_url)
            multithreaded_download(prev_to_download, download_from_url)

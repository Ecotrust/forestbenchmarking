# %% [markdown]
# # Fetching Elevation Data
# We have prepared shapefiles containing the USGS quarter quadrangles that have good coverage of forest stand delineations that we want to grab other data for. We'll fetch elevation data (a Digital Elevation Model) from The National Map for each tile, create  additional derivative products, and write our outputs as GeoTiffs to Google Drive. 
# 
# DEMs will be retrieved from a webservice hosted by The National Map using elevation data produced by the US Geological Survey's [3D Elevation Program](https://www.usgs.gov/core-science-systems/ngp/3dep/what-is-3dep?). 
# 
# In subsequent processing, we may generate other terrain-derived layers (e.g., slope, aspect, curvature) from these DEMs. For now, we'll just grab the raw DEM and generate a couple layers quantifying Topographic Position Index (TPI). We add TPI layers here because, as described below, the calculation of TPI involves a convolution that requires elevation data that extends beyond the footprint of the tile we will ultimately retain and export to Google Drive. 
# 
# ### Topographic Position Index (TPI)
# TPI characterizes the elevation of a point on the landscape relative to its surroundings. It is calculated as a convolution of a DEM where the size of the kernel used for the convolution can be adjusted to capture local-scale to regional-scale topographic features.  
# 
# <figure>
# <img src='http://drive.google.com/uc?export=view&id=1TY5OYyOA4n7ke-CtGR7LFIdRVpPIG5IX' width=500px>
# <figcaption align='left'>Illustration of TPI (credit: <a href='http://jennessent.com/arcview/TPI_jen_poster.htm'>Jeff Jenness</a>)</figcaption>
# </figure>
# 
# In this notebook, we follow the original description of TPI by [Weiss (2001)](http://www.jennessent.com/downloads/tpi-poster-tnc_18x22.pdf) by using an annular (donut-shaped) kernel that subtracts the average elevation of pixels in the donut from the elevation at the single pixel in the center of the donut hole. We implement TPI calculations at a range of 300m (annulus runs from 150-300m from the center pixel) and for 2000m (annulus runs 1850-2000m).

# %% [markdown]
# # Mount Google Drive 
# So we can access our files showing tile locations, and save the rasters we will generate from the elevation data.

# %%
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# %%
# ! pip install geopandas rasterio -q

# %% [markdown]
# The following functions will do the work to retrieve the DEM (or calculate a TPI raster from a DEM) from The National Map's web service.

# %%
import os
import sys
import requests
from pathlib import Path
import contextlib
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import MemoryFile
from rasterio import transform
from rasterio.warp import reproject, Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from scipy.ndimage import convolve
from skimage import filters
from skimage.morphology import disk
from skimage.util import apply_parallel
from tqdm import tqdm
from pyproj import CRS
import matplotlib.pyplot as plt


root = [p for p in Path(__file__).parents if p.name == "forestbenchmarking"][0]
sys.path.append(os.path.abspath(root / "src/data"))

from utils import create_directory_tree, degrees_to_meters, split_bbox
from gee_utils import multithreaded_download

def dem_from_tnm(bbox, res=10, inSR=4326, **kwargs):
    """
    Retrieves a Digital Elevation Model (DEM) image from The National Map (TNM)
    web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned DEM (grid cell size)
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    dem : numpy array
      DEM image as array
    """
    xmin, ymin, xmax, ymax = bbox

    if inSR == 4326:
        dx = degrees_to_meters(xmax - xmin)
        dy = degrees_to_meters(ymax - ymin, angle='lat')
    else:
        dx = xmax - xmin
        dy = ymax - ymin

    width = int(abs(dx) // res)  # type: ignore
    height = int(abs(dy) // res)  # type: ignore

    BASE_URL = ''.join([
        'https://elevation.nationalmap.gov/arcgis/rest/',
        'services/3DEPElevation/ImageServer/exportImage'
    ])

    params = dict(
        bbox=','.join([str(x) for x in bbox]),
        bboxSR=inSR,
        size=f'{width},{height}',
        imageSR=inSR,
        time=None,
        format='tiff',
        pixelType='F32',
        noData=None,
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression=None,
        compressionQuality=None,
        bandIds=None,
        mosaicRule=None,
        renderingRule=None,
        f='image'
    )
    
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    with MemoryFile(r.content) as memfile:
        src = memfile.open()
        dem = src.read(1)
    
    return dem


def download_dem(filepath, bbox, res=10, inSR=4326, **kwargs):
    """
    Retrieves a Digital Elevation Model (DEM) image from The National Map (TNM)
    web service.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    res : numeric
      spatial resolution to use for returned DEM (grid cell size)
    inSR : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    dem : numpy array
      DEM image as array
    """
    xmin, ymin, xmax, ymax = bbox

    if inSR == 4326:
        dx = degrees_to_meters(xmax - xmin)
        dy = degrees_to_meters(ymax - ymin, angle='lat')
    else:
        dx = xmax - xmin
        dy = ymax - ymin

    width = int(abs(dx) // res)  # type: ignore
    height = int(abs(dy) // res)  # type: ignore

    BASE_URL = ''.join([
        'https://elevation.nationalmap.gov/arcgis/rest/',
        'services/3DEPElevation/ImageServer/exportImage'
    ])

    params = dict(
        bbox=','.join([str(x) for x in bbox]),
        bboxSR=inSR,
        size=f'{width},{height}',
        imageSR=inSR,
        time=None,
        format='tiff',
        pixelType='F32',
        noData=None,
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression=None,
        compressionQuality=None,
        bandIds=None,
        mosaicRule=None,
        renderingRule=None,
        f='image'
    )

    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)

    # 
    profile = dict(
        driver='GTiff',
        interleave='band',
        tiled=True,
        crs=inSR,
        width=width, 
        height=height,
        transform=transform.from_bounds(*bbox, width, height),
        blockxsize=256,
        blockysize=256,
        compress='lzw',
        nodata=-9999,
        dtype=rasterio.float32,
        count=1,
    )

    with MemoryFile(r.content) as data:
        with rasterio.open(filepath, 'w', **profile) as dst:
            src = data.open()
            dst.write(src.read())
            dst.update_tags(**src.tags())
            del src


def download_quad(filepath, bbox, dim=1, num_threads=None, **kwargs):
    """Breaks user-provided bounding box into quadrants and retrieves data
    using `fetcher` for each quadrant in parallel using a ThreadPool.

    Parameters
    ----------
    """
    from gee_utils import multithreaded_download

    if dim > 1:
        if num_threads is None:
            num_threads = dim**2

        bboxes = split_bbox(dim, bbox)
        n = len(bboxes)

        params = np.array([
            [f'{filepath}_{i}.tif' for i in range(n)],
            bboxes.tolist(),
        ], dtype=object).T

        multithreaded_download(params, download_dem)

    else:
        download_dem(f'{filepath}.tif', bbox, **kwargs)


def quad_fetch(bbox, dim=1, num_threads=None, **kwargs):
    """Breaks user-provided bounding box into quadrants and retrieves data
    using `fetcher` for each quadrant in parallel using a ThreadPool.

    Parameters
    ----------
    fetcher : callable
      data-fetching function, expected to return an array-like object
    bbox : 4-tuple or list
      coordinates of x_min, y_min, x_max, and y_max for bounding box of tile
    num_threads : int
      number of threads to use for parallel executing of data requests
    qq : bool
      whether or not to execute request for quarter quads, which executes this
      function recursively for each quadrant
    *args
      additional positional arguments that will be passed to `fetcher`
    **kwargs
      additional keyword arguments that will be passed to `fetcher`

    Returns
    -------
    quad_img : array
      image returned with quads stitched together into a single array

    """
    if dim > 1:
        if num_threads is None:
            num_threads = dim**2

        bboxes = split_bbox(dim, bbox)
        n = len(bboxes)

        get_quads = partial(dem_from_tnm, **kwargs)
        with ThreadPool(num_threads) as p:
            quads = p.map(get_quads, bboxes)

        # Split quads list in tuples of size dim
        quad_list = [quads[x:x + dim] for x in range(0, len(quads), dim)]
        # Reverse order of rows to match rasterio's convention
        [x.reverse() for x in quad_list]
        return np.hstack([np.vstack(quad_list[x]) for x in range(0, len(quad_list))])
          
    else:
        dem = dem_from_tnm(bbox, **kwargs)
    
    return dem

# %%
def tpi(dem, irad=5, orad=10, res=10, norm=False):
    """
    Produces a raster of Topographic Position Index (TPI) by fetching a Digital
    Elevation Model (DEM) from The National Map (TNM) web service.

    TPI is the difference between the elevation at a location from the average
    elevation of its surroundings, calculated using an annulus (ring). This
    function permits the calculation of average surrounding elevation using
    a coarser grain, and return the TPI user a higher-resolution DEM.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    irad : numeric
      inner radius of annulus used to calculate TPI
    orad : numeric
      outer radius of annulus used to calculate TPI
    res : numeric
      spatial resolution of Digital Elevation Model (DEM)
    norm : bool
      whether to return a normalized version of TPI, with mean = 0 and SD = 1

    Returns
    -------
    tpi : ndarray
      TPI image as array
    """
    k_orad = orad // res
    k_irad = irad // res

    # dem = np.pad(dem, k_orad, mode='constant', constant_values=-9999)
    kernel = disk(k_orad) - np.pad(disk(k_irad), pad_width=(k_orad - k_irad))
    weights = kernel / kernel.sum()

    def conv(dem): return convolve(dem, weights, mode='nearest')

    convolved = apply_parallel(conv, dem, compute=True, depth=k_orad)
    tpi = dem - convolved

    # trim the padding around the dem used to calculate TPI
    # tpi = tpi[orad // res:-orad // res, orad // res:-orad // res]

    if norm:
        tpi_mean = (dem - convolved).mean()
        tpi_std = (dem - convolved).std()
        tpi = (tpi - tpi_mean) / tpi_std

    return tpi


# %%
def infer_utm(bbox):
    """Infer the UTM Coordinate Reference System (CRS) by determining 
    the UTM zone where a given lat/long bounding box is located.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)

    Returns
    -------
    crs : pyproj.CRS
      UTM crs for the bounding box
    """
    xmin, _, xmax, _ = bbox
    midpoint = (xmax - xmin) / 2

    if xmax <= -120 + midpoint:
        epsg = 32610
    elif (xmin + midpoint > -120) and (xmax <= -114 + midpoint):
        epsg = 32611
    elif (xmin  + midpoint > -114) and (xmax <= -108 + midpoint):
        epsg = 32612
    elif (xmin  + midpoint > -108) and (xmax <= -102 + midpoint):
        epsg = 32613
    elif (xmin  + midpoint > -102) and (xmax <= -96 + midpoint):
        epsg = 32614 
    elif (xmin  + midpoint > -96) and (xmax <= -90 + midpoint):
        epsg = 32615
    elif (xmin  + midpoint > -90) and (xmax <= -84 + midpoint):
        epsg = 32616
    elif (xmin  + midpoint > -84) and (xmax <= -78 + midpoint):
        epsg = 32617
    elif (xmin  + midpoint > -78) and (xmax <= -72 + midpoint):
        epsg = 32618
    elif (xmin + midpoint > -72):
        epsg = 32619

    return CRS.from_epsg(epsg)


# Supress output from c++ shared libs and python warnings
# so the progress bar doesn't get messed up
# See: 
# 1. https://stackoverflow.com/a/57677370/1913361
# 2. https://stackoverflow.com/a/28321717/1913361
# 3. https://stackoverflow.com/a/37243211/1913361
# TODO: integrate solutions into one decorator or context manager
class SuppressStream(object): 
    def __init__(self, file=None, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()
        self.file = file

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

# def supress_stdout(func):
#     def wrapper(*a, **ka):
#         with open(os.devnull, 'w') as devnull:
#             with contextlib.redirect_stdout(devnull):
#                 return func(*a, **ka)
#     return wrapper

@contextlib.contextmanager
def supress_stdout():
    save_stdout = sys.stdout
    sys.stdout = SuppressStream(sys.stdout)
    yield
    sys.stdout = save_stdout
# ------

# %%
# @supress_stdout
def slope(dem):
    """
    Produces a raster of slope.

    Parameters
    ----------
    dem : ndarray
        Digital Elevation Model (DEM) as array
    res : numeric
      spatial resolution of Digital Elevation Model (DEM)

    Returns
    -------
    slope : ndarray
      slope image as array
    """
    import richdem as rd
    # dx, dy = np.gradient(dem, res)
    # slope = np.arctan(np.sqrt(dx**2 + dy**2))

    # Convert ndarray to richdem.rdarray
    with SuppressStream():
        rd_dem = rd.rdarray(dem, no_data=-9999)
        slope = rd.TerrainAttribute(rd_dem, attrib='slope_riserun')

    return np.array(slope)


# @supress_stdout
def aspect(dem):
    """
    Produces a raster of aspect.

    Parameters
    ----------
    dem : ndarray
        Digital Elevation Model (DEM) as array
    res : numeric
      spatial resolution of Digital Elevation Model (DEM)

    Returns
    -------
    aspect : ndarray
      aspect image as array
    """
    import richdem as rd
  
    # Convert ndarray to richdem.rdarray
    with SuppressStream():
        rd_dem = rd.rdarray(dem, no_data=-9999)
        aspect = rd.TerrainAttribute(rd_dem, attrib='aspect')

    return np.array(aspect)


def flow_accumulation(dem):
    """
    Produces a raster of flow accumulation.

    Parameters
    ----------
    dem : ndarray
        Digital Elevation Model (DEM) as array
    res : numeric
      spatial resolution of Digital Elevation Model (DEM)

    Returns
    -------
    flow_accumulation : ndarray
      flow accumulation image as array
    """
    from pysheds.grid import Grid
    from pysheds.view import Raster, ViewFinder
  
    # Convert ndarray to pysheds.view.Raster
    r_dem = Raster(dem, viewfinder=ViewFinder(shape=dem.shape))
    grid = Grid.from_raster(r_dem)
    inflated_dem = grid.resolve_flats(r_dem)
    flow_direction = grid.flowdir(inflated_dem)
    flow_accumulation = grid.accumulation(flow_direction)
    return np.array(flow_accumulation)

# %%
def classify_slope_position(tpi, slope):
    """Classifies an image of normalized Topograhic Position Index into 6 slope
    position classes:

    =======  ============
    Slope #  Description
    =======  ============
    1        Valley
    2        Lower Slope
    3        Flat Slope
    4        Middle Slope
    5        Upper Slope
    6        Ridge
    =======  ============

    Classification following Weiss, A. 2001. "Topographic Position and
    Landforms Analysis." Poster presentation, ESRI User Conference, San Diego,
    CA.  http://www.jennessent.com/downloads/tpi-poster-tnc_18x22.pdf

    Parameters
    ----------
    tpi : array
      TPI values, assumed to be normalized to have mean = 0 and standard
      deviation = 1
    slope : array
      slope of terrain, in degrees
    """
    assert tpi.shape == slope.shape
    pos = np.empty(tpi.shape, dtype=int)

    pos[(tpi<=-1)] = 1
    pos[(tpi>-1)*(tpi<-0.5)] = 2
    pos[(tpi>-0.5)*(tpi<0.5)*(slope<=5)] = 3
    pos[(tpi>-0.5)*(tpi<0.5)*(slope>5)] = 4
    pos[(tpi>0.5)*(tpi<=1.0)] = 5
    pos[(tpi>1)] = 6

    return pos


def classify_landform(tpi_near, tpi_far, slope):
    """Classifies a landscape into 10 landforms given "near" and "far" values
    of Topographic Position Index (TPI) and a slope raster.

    ==========  ======================================
    Landform #   Description
    ==========  ======================================
    1           canyons, deeply-incised streams
    2           midslope drainages, shallow valleys
    3           upland drainages, headwaters
    4           U-shape valleys
    5           plains
    6           open slopes
    7           upper slopes, mesas
    8           local ridges, hills in valleys
    9           midslope ridges, small hills in plains
    10          mountain tops, high ridges
    ==========  ======================================

    Classification following Weiss, A. 2001. "Topographic Position and
    Landforms Analysis." Poster presentation, ESRI User Conference, San Diego,
    CA.  http://www.jennessent.com/downloads/tpi-poster-tnc_18x22.pdf

    Parameters
    ----------
    tpi_near : array
      TPI values calculated using a smaller neighborhood, assumed to be
      normalized to have mean = 0 and standard deviation = 1
    tpi_far : array
      TPI values calculated using a smaller neighborhood, assumed to be
      normalized to have mean = 0 and standard deviation = 1
    slope : array
      slope of terrain, in degrees
    """
    assert tpi_near.shape == tpi_far.shape == slope.shape
    lf = np.empty(tpi_near.shape, dtype=int)

    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far<1)*(tpi_far>-1)*(slope<=5)] = 5
    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far<1)*(tpi_far>-1)*(slope>5)] = 6
    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far>=1)] = 7
    lf[(tpi_near<1)*(tpi_near>-1)*(tpi_far<=-1)] = 4
    lf[(tpi_near<=-1)*(tpi_far<1)*(tpi_far>-1)] = 2
    lf[(tpi_near>=1)*(tpi_far<1)*(tpi_far>-1)] = 9
    lf[(tpi_near<=-1)*(tpi_far>=1)] = 3
    lf[(tpi_near<=-1)*(tpi_far<=-1)] = 1
    lf[(tpi_near>=1)*(tpi_far>=1)] = 10
    lf[(tpi_near>=1)*(tpi_far<=-1)] = 8

    return lf

# %%
# def fetch_dem(outfile, utm_bbox, utm_epsg, bbox, res=10, dim=3, overwrite=False):
#     """Fetch a Digital Elevation Model (DEM) for a given cell_id.
#     """
#     assert isinstance(outfile, Path), "outfile must be a pathlib.Path object"
#     if not os.path.exists(outfile) or overwrite:
#         create_directory_tree(outfile.parent)

#         PROFILE = {
#             'driver': 'GTiff',
#             'interleave': 'band',
#             'tiled': True,
#             'blockxsize': 256,
#             'blockysize': 256,
#             'compress': 'lzw',
#             'nodata': -9999,
#             'dtype': rasterio.float32
#         }    

#         p_trf = transform.from_bounds(*utm_bbox, utm_bbox[2]-utm_bbox[0], utm_bbox[-1]-utm_bbox[1]) 

#         # We'll need this to transform the data back to the original CRS
#         width = np.ceil(degrees_to_meters(bbox[2]-bbox[0]))
#         height = np.ceil(degrees_to_meters(bbox[-1]-bbox[1]))
#         trf = transform.from_bounds(*bbox, width, height) 

#         dem = quad_fetch(bbox=utm_bbox, dim=dim, res=res, inSR=utm_epsg, noData=-9999)
#         ## apply a smoothing filter to mitigate stitching/edge artifacts
#         dem = filters.gaussian(dem, 3)

#         # ---
#         # TODO: use multiprocessing to speed up this step
#         # TODO: generate maps with projected dem
#         topo_metrics = [
#             slope(dem),
#             aspect(dem),
#             flow_accumulation(dem),
#             tpi(dem, irad=15, orad=30, res=res),
#             tpi(dem, irad=180, orad=200, res=res),
#         ]
#         # ---

#         band_info = {
#             'dem': 'Digital Elevation Model',
#             'slope': 'Slope',
#             'aspect': 'Aspect',
#             'flowac': 'Flow Accumulation',
#             'tpi': 'Topographic Position Index',
#         }

#         ## Reproject, generate cog, and write the data to disk
#         crs = CRS.from_epsg(4326)
#         PROFILE.update(crs=crs, transform=trf, width=width, height=height, count=len(topo_metrics)+1)
#         cog_profile = cog_profiles.get("deflate")
#         # with rasterio.open(outfile, 'w', **PROFILE) as dst:
#         with MemoryFile() as memfile:
#             with memfile.open(mode='w', **PROFILE) as dst:
#                 dst_idx = 1
#                 for band, data in zip(band_info.keys(), *topo_metrics):
#                     output = np.zeros(dst.shape, rasterio.float32)
#                     reproject(
#                         source=data,
#                         destination=output,
#                         src_transform=p_trf,
#                         src_crs=crs,
#                         dst_transform=trf,
#                         dst_crs=CRS.from_epsg(utm_epsg),
#                         resampling=Resampling.nearest
#                     )
#                     dst.write(output, dst_idx)
#                     dst.set_band_description(dst_idx, band_info[band])
#                     dst_idx += 1
                
#                 cog_translate(
#                     dst,
#                     outfile,
#                     cog_profile,
#                     in_memory=True,
#                     quiet=True
#                 )

#         return True

#     else:
#         print(f'File {outfile.name} already exists, skipping.')
#         return False


# %%
# def fetch_dems(path_to_tiles, out_dir,  res=10):
#     """Loop through a GeoDataFrame, fetch the relevant data, and write GeoTiffs to disk 
#     in the appropriate formats.
#     """
#     gdf = gpd.read_file(path_to_tiles)

#     # Extract parameters from GeoDataFrame    
#     gdf['bbox'] = gdf.geometry.apply(lambda geom: geom.bounds)
#     gdf['epsg'] = gdf.geometry.apply(lambda geom: infer_utm(geom.bounds).to_epsg())
#     gdf['p_geom'] = [
#         gdf[gdf.CELL_ID == cell].geometry.to_crs(epsg) 
#         for cell, epsg in zip(gdf.CELL_ID, gdf.epsg)
#     ]
#     gdf['p_bbox'] = gdf.p_geom.apply(lambda geom: geom.values[0].bounds)
#     gdf['fileprefix'] = gdf.CELL_ID.apply(lambda cell: f'{cell}_3DEP_{res}mDEM')
#     gdf['outfile'] = gdf.fileprefix.apply(lambda p: root / out_dir / p / f'{p}-cog.tif')
    
#     params = np.array(gdf[['outfile', 'p_bbox', 'epsg', 'bbox']])
#     multithreaded_download(params, fetch_dem)


def center_crop_array(new_size, array):
    xpad, ypad = (np.subtract(array.shape, new_size)/2).astype(int)
    dx, dy = np.subtract(new_size, array[xpad:-xpad, ypad:-ypad].shape)
    return array[xpad:-xpad+dx, ypad:-ypad+dy]

# %%
def fetch_metadata(image_id, bands, res, out_dir='.'):
    # id
    # resolution
    # properties
    # - bands
    # - datetime start (in seconds)
    # crs
    # transform
    # bounds
    # license for each collection
    import requests
    import json
    import calendar 
    from datetime import datetime

    month_name = {month: index for index, month in enumerate(calendar.month_name) if month}

    URL = 'https://elevation.nationalmap.gov/arcgis/'\
          'rest/services/3DEPElevation/ImageServer?f=pjson'
    r = requests.get(URL)
    src_metadata = r.json()

    metadata = {}
    for key in src_metadata.keys():
        if key in ['currentVersion', 'description', 'copyrightText']:
            metadata[key] = src_metadata[key]

    m, d, y = src_metadata['copyrightText']\
                .replace(',', '')\
                  .replace('.', '')\
                    .split(' ')[-3:]
    timestamp = datetime(int(y), month_name[m], int(d)).timestamp()
    
    _bands = [{'id': key, 'name': bands[key]} for key in bands.keys()]

    metadata.update(
        id=image_id,
        name=' '.join(src_metadata['copyrightText'].split(' ')[:-3]),
        resolution=res,
        bands=_bands,
        properties={
            'system:time_start': int(timestamp * 1000),
        }
    )

    with open(os.path.join(out_dir, f"{image_id}-metadata.json"), "w") as f:
            f.write(json.dumps(metadata, indent=4))

    return True

# %%
# def save_preview(image_array, image_id, out_dir='.'):
#     from PIL import Image
#     import matplotlib.pyplot as plt

#     # cm_terrain = plt.get_cmap('terrain')
#     # img = cm_terrain(image_array)
#     # img = np.uint8(img * 255)
#     img = Image.fromarray(image_array, mode='L').convert('L')
#     img.thumbnail((512, 512))
#     img.save(os.path.join(out_dir, f'{image_id}-preview.png'))

def fetch_dems(path_to_tiles, out_dir, res=10, overwrite=False):
    """Loop through a GeoDataFrame, fetch the relevant data, and write GeoTiffs to disk 
    in the appropriate formats.
    """
    gdf = gpd.read_file(path_to_tiles)
    # epsg = gdf.crs.to_epsg()
    
    desc = f'Fetching DEMs'
    
    PROFILE = {
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'lzw',
        'nodata': -9999,
        'dtype': rasterio.float32,
        # 'count': 5 # set number of bands
    }

    ## loop through all the geometries in the geodataframe and fetch the DEM
    with tqdm(total=len(gdf), desc=desc, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}', 
              file=sys.stdout) as pbar:
        with supress_stdout():
            for idx in range(len(gdf)):
                ## don't bother fetching data if we already have processed this tile
                cell_id = gdf.loc[idx, 'CELL_ID']
                itemdir = f'{cell_id}_3DEP_{res}mDEM'
                filename = f'{itemdir}-cog.tif'
                outfile = root / out_dir / itemdir / filename
                
                if os.path.exists(outfile) and not overwrite:
                    pbar.write(f"File {itemdir} already exists, skipping...")
                    pbar.update(1)
                    continue

                pbar.write(f"Processing file {itemdir}")
                create_directory_tree(out_dir, itemdir)

                # We want to request the data in a planar coordinate system
                # so we can calculate topographic metrics. This is to avoid
                # distortions due to the curvature of the earth.
                # See discussion https://gis.stackexchange.com/q/7906/72937
                p_crs = infer_utm(gdf.loc[idx, 'geometry'].bounds)
                geom = gdf[gdf.CELL_ID == gdf.CELL_ID[idx]].geometry.to_crs(p_crs)
                p_bbox = geom[idx].bounds
                p_width = np.ceil((p_bbox[2]-p_bbox[0])/res).astype(int) 
                p_height = np.ceil((p_bbox[-1]-p_bbox[1])/res).astype(int)
                p_trf = transform.from_bounds(*p_bbox, p_width, p_height)  # type: ignore

                # Extend the AOI with a buffer to avoid edge effects
                # when calculating topographic metrics
                padding = int(round((p_width)//2/100))*100
                p_buffer = geom.buffer(padding * res, join_style=2)
                bbox_buff = p_buffer.bounds.values[0]

                # Fetch DEM and apply a smoothing filter to mitigate stitching/edge artifacts
                dem = quad_fetch(bbox=bbox_buff, dim=3, res=res, inSR=p_crs.to_epsg(), noData=-9999)
                dem = filters.gaussian(dem, 3)
                
                # We'll need this to transform the data back to the original CRS
                crs = CRS.from_epsg(4326)
                bbox = gdf.loc[idx, 'geometry'].bounds
                width = np.ceil(degrees_to_meters(bbox[2]-bbox[0])/res)
                height = np.ceil(degrees_to_meters(bbox[-1]-bbox[1])/res)
                trf = transform.from_bounds(*bbox, width, height)  # type: ignore

                # ---
                # TODO: use multiprocessing to speed up this step
                bands=[
                    dem,
                    slope(dem),
                    aspect(dem),
                    flow_accumulation(dem),
                    tpi(dem, irad=15, orad=30, res=res),
                    tpi(dem, irad=180, orad=200, res=res),
                ]
                bands.append(
                    classify_slope_position(bands[3], bands[0]) 
                )   
                bands.append(
                    classify_landform(bands[4], bands[3], bands[0])
                )

                bands = [center_crop_array((p_height, p_width), x) for x in bands]
                # ---

                band_info = {
                    'dem': 'Digital Elevation Model',
                    'slope': 'Slope',
                    'aspect': 'Aspect',
                    'flowacc': 'Flow Accumulation',
                    'tpi300': 'Topographic Position Index (300m)',
                    'tpi2000': 'Topographic Position Index (2000m)',
                    'spc300': 'Slope Position Class (300m)',
                    'landform': 'Landform Class',
                }

                fetch_metadata(itemdir, band_info, res, out_dir / itemdir)
                
                ## Reproject, generate cog, and write the data to disk
                PROFILE.update(crs=crs, transform=trf, width=width, height=height, count=len(bands)+1)
                cog_profile = cog_profiles.get("deflate")

                with MemoryFile() as memfile:
                    with memfile.open(**PROFILE) as dst:
                        dst_idx = 1
                        for band, data in zip(band_info.keys(), bands):
                            output = np.zeros(dst.shape, rasterio.float32)
                            reproject(
                                source=data,
                                destination=output,
                                src_transform=p_trf,
                                src_crs=p_crs,
                                dst_transform=trf,
                                dst_crs=crs,
                                resampling=Resampling.nearest
                            )
                            if band == 'dem':
                                imgname = f'{itemdir}-preview.png'
                                plt.imsave(out_dir / itemdir / imgname, output, cmap='gist_earth')
                            dst.write(output, dst_idx)
                            dst.set_band_description(dst_idx, band_info[band])
                            dst_idx += 1

                        with SuppressStream(sys.stdout):
                            cog_translate(
                                dst,
                                outfile,
                                cog_profile,
                                in_memory=True,
                                quiet=True
                            )

                pbar.update(1)


# %%
if __name__ == '__main__':
    WORK_DIR = 'data/external/usfs_stands'
    OUT_DIR = 'data/processed'
    OR_QUADS = 'oregon_quarter_quads_sample.geojson'
    out_path = create_directory_tree(OUT_DIR, '3DEP')

    fetch_dems(path_to_tiles=os.path.join(WORK_DIR, OR_QUADS), res=10,
               out_dir=out_path, overwrite=True)

# %%
# WORK_DIR = '../../data/external/usfs_stands/'
# OUT_DIR = 'data/processed'
# OR_QUADS = 'oregon_quarter_quads_sample.geojson'

# gdf = gpd.read_file(WORK_DIR + OR_QUADS)
# bbox = gdf.loc[0, 'geometry'].bounds
# p_crs = infer_utm(bbox)
# cell_id = gdf.loc[0, "CELL_ID"]
# p_geom = gdf[gdf.CELL_ID == 108243].geometry.to_crs(p_crs)[0]
# p_bbox = p_geom.bounds
# xmin, ymin, xmax, ymax = p_bbox
# res = 10
# padding = int(round((xmax-xmin)/res//2/100))*100
# w = int((xmax-xmin)//res)
# h = int((ymax-ymin)//res)
# p_buffer = p_geom.buffer(padding * res, join_style=2)
# bbox_buff = p_buffer.bounds
# dem = dem_from_tnm(bbox_buff, res=res, dim=3, inSR=p_crs.to_epsg())
# xpad, ypad = (np.subtract(dem.shape, (h, w))/2).astype(int)
# dx, dy = np.abs(np.subtract((h, w), dem[xpad:-xpad, ypad:-ypad].shape))
# dem[xpad:-xpad-dx, ypad:-ypad-dy].shape

# %%

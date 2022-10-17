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
from multiprocessing import Pool

import numpy as np
import geopandas as gpd

from functools import partial
from multiprocessing.pool import ThreadPool
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


root = [p for p in Path(__file__).parents if p.name == "forestbenchmarking"][0]
sys.path.append(os.path.abspath(root / "src/data"))

from utils import create_directory_tree, degrees_to_meters, split_bbox
from gee_utils import multithreaded_download

def dem_from_tnm(bbox, res=1, inSR=4326, **kwargs):
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


def download_dem(filepath, bbox, res=1, inSR=4326, **kwargs):
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
def tpi(dem, irad=5, orad=10, res=1, norm=False):
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

    kernel = disk(k_orad) - np.pad(disk(k_irad), pad_width=(k_orad - k_irad))
    weights = kernel / kernel.sum()

    def conv(dem): return convolve(dem, weights)

    convolved = apply_parallel(conv, dem, compute=True, depth=k_orad)
    tpi = dem - convolved

    # trim the padding around the dem used to calculate TPI
    tpi = tpi[orad // res:-orad // res, orad // res:-orad // res]

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
# TODO: integrate both solutions into one decorator and move to utils.py
class SuppressStream(object): 

    def __init__(self, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper
# ------

# %%
@supress_stdout
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
    rd_dem = rd.rdarray(dem, no_data=-9999)

    with SuppressStream():
        slope = rd.TerrainAttribute(rd_dem, attrib='slope_riserun')

    return np.array(slope)

@supress_stdout
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
    rd_dem = rd.rdarray(dem, no_data=-9999)

    with SuppressStream():
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


def fetch_dem(outfile, utm_bbox, utm_epsg, bbox, res=1, dim=3, overwrite=False):
    """Fetch a Digital Elevation Model (DEM) for a given cell_id.
    """
    assert isinstance(outfile, Path), "outfile must be a pathlib.Path object"
    if not os.path.exists(outfile) or overwrite:
        create_directory_tree(outfile.parent)

        PROFILE = {
            'driver': 'GTiff',
            'interleave': 'band',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'compress': 'lzw',
            'nodata': -9999,
            'dtype': rasterio.float32
        }    

        p_trf = transform.from_bounds(*utm_bbox, utm_bbox[2]-utm_bbox[0], utm_bbox[-1]-utm_bbox[1]) 

        # We'll need this to transform the data back to the original CRS
        width = np.ceil(degrees_to_meters(bbox[2]-bbox[0]))
        height = np.ceil(degrees_to_meters(bbox[-1]-bbox[1]))
        trf = transform.from_bounds(*bbox, width, height) 

        dem = quad_fetch(bbox=utm_bbox, dim=dim, res=res, inSR=utm_epsg, noData=-9999)
        ## apply a smoothing filter to mitigate stitching/edge artifacts
        dem = filters.gaussian(dem, 3)

        # ---
        # TODO: use multiprocessing to speed up this step
        # TODO: generate maps with projected dem
        topo_metrics = [
            slope(dem),
            aspect(dem),
            flow_accumulation(dem),
            tpi(dem, irad=15, orad=30, res=res),
            tpi(dem, irad=180, orad=200, res=res),
        ]
        # ---

        band_info = {
            'dem': 'Digital Elevation Model',
            'slope': 'Slope',
            'aspect': 'Aspect',
            'flowac': 'Flow Accumulation',
            'tpi': 'Topographic Position Index',
        }

        ## Reproject, generate cog, and write the data to disk
        crs = CRS.from_epsg(4326)
        PROFILE.update(crs=crs, transform=trf, width=width, height=height, count=len(topo_metrics)+1)
        cog_profile = cog_profiles.get("deflate")
        # with rasterio.open(outfile, 'w', **PROFILE) as dst:
        with MemoryFile() as memfile:
            with memfile.open(mode='w', **PROFILE) as dst:
                dst_idx = 1
                for band, data in zip(band_info.keys(), *topo_metrics):
                    output = np.zeros(dst.shape, rasterio.float32)
                    reproject(
                        source=data,
                        destination=output,
                        src_transform=p_trf,
                        src_crs=crs,
                        dst_transform=trf,
                        dst_crs=CRS.from_epsg(utm_epsg),
                        resampling=Resampling.nearest
                    )
                    dst.write(output, dst_idx)
                    dst.set_band_description(dst_idx, band_info[band])
                    dst_idx += 1
                
                cog_translate(
                    dst,
                    outfile,
                    cog_profile,
                    in_memory=True,
                    quiet=True
                )

        return True

    else:
        print(f'File {outfile.name} already exists, skipping.')
        return False


# %%
# def fetch_dems(path_to_tiles, out_dir,  res=1):
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


def fetch_dems(path_to_tiles, out_dir, res=1, overwrite=False):
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
        'count': 5 # set number of bands
    }

    ## loop through all the geometries in the geodataframe and fetch the DEM
    with tqdm(total=len(gdf), desc=desc, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
        for idx in range(len(gdf)):
            ## don't bother fetching data if we already have processed this tile
            cell_id = gdf.loc[idx, 'CELL_ID']
            outname = f'{cell_id}_3DEP_{res}mDEM-cog.tif'
            outfile = root / out_dir / outname.replace('-cog.tif', '') / outname
            
            if os.path.exists(outfile) and not overwrite:
                pbar.write(f"File {outname} already exists, skipping...")
                pbar.update(1)
                continue

            pbar.write(f"Processing file {outname}")
            create_directory_tree(out_dir, outname.replace('-cog.tif', ''))

            # We want to request the data in a planar coordinate system
            # so we can calculate topographic metrics. This is to avoid
            # distortions due to the curvature of the earth.
            # See discussion https://gis.stackexchange.com/q/7906/72937
            p_crs = infer_utm(gdf.loc[idx, 'geometry'].bounds)
            geom = gdf[gdf.CELL_ID == gdf.CELL_ID[idx]].geometry.to_crs(p_crs)
            p_bbox = geom[idx].bounds
            p_width = np.ceil((p_bbox[2]-p_bbox[0])/res) 
            p_height = np.ceil((p_bbox[-1]-p_bbox[1])/res)
            p_trf = transform.from_bounds(*p_bbox, p_width, p_height)  # type: ignore

            # We'll need this to transform the data back to the original CRS
            bbox = gdf.loc[idx, 'geometry'].bounds
            width = np.ceil(degrees_to_meters(bbox[2]-bbox[0])/res)
            height = np.ceil(degrees_to_meters(bbox[-1]-bbox[1])/res)
            trf = transform.from_bounds(*bbox, width, height)  # type: ignore

            dem = quad_fetch(bbox=p_bbox, dim=3, res=res, inSR=p_crs.to_epsg(), noData=-9999)
            ## apply a smoothing filter to mitigate stitching/edge artifacts
            dem = filters.gaussian(dem, 3)
            
            # ---
            # TODO: use multiprocessing to speed up this step
            # TODO: generate maps with projected dem
            _slope = slope(dem)
            _aspect = aspect(dem)
            _flowac = flow_accumulation(dem)
            _tpi = tpi(dem, res=res)
            # ---

            band_info = {
                'dem': 'Digital Elevation Model',
                'slope': 'Slope',
                'aspect': 'Aspect',
                'flowac': 'Flow Accumulation',
                'tpi': 'Topographic Position Index',
            }

            ## Reproject, generate cog, and write the data to disk
            crs = CRS.from_epsg(4326)
            PROFILE.update(crs=crs, transform=trf, width=width, height=height)
            cog_profile = cog_profiles.get("deflate")
            # with rasterio.open(outfile, 'w', **PROFILE) as dst:
            with MemoryFile() as memfile:
                with memfile.open(**PROFILE) as dst:
                    dst_idx = 1
                    for band, data in zip(band_info.keys(), [dem, _slope, _aspect, _flowac, _tpi]):
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
                        dst.write(output, dst_idx)
                        dst.set_band_description(dst_idx, band_info[band])
                        dst_idx += 1
                   
                    cog_translate(
                        dst,
                        outfile,
                        cog_profile,
                        nodata=-9999,
                        in_memory=True,
                        quiet=True
                    )

            pbar.update(1)

def fetch_metadata(path_to_tiles, out_dir, overwrite=False):
    # id
    # name
    # datetime start
    # datetime end
    # crs
    # transform
    # bounds
    # bands
    # license for each collection
    pass


if __name__ == '__main__':
    WORK_DIR = 'data/external/usfs_stands'
    OUT_DIR = 'data/raw'
    OR_QUADS = 'oregon_quarter_quads_sample.geojson'
    out_path = create_directory_tree(OUT_DIR, '3DEP')

    fetch_dems(path_to_tiles=os.path.join(WORK_DIR, OR_QUADS), res=10,
               out_dir=out_path, overwrite=True)

# %%

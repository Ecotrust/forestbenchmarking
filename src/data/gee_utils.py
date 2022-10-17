"""
"""
import os
import json

from zipfile import ZipFile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
from geetools import composite
import tqdm


class GEEImageLoader:
    """Class to hold additional methods and parameters to fetch Google Earth Engine (GEE) images.

    Parameters
    ----------
    image : ee.Image
    """

    def __init__(self, image: ee.Image):

        self.image = image
        self.metadata = image.getInfo()

        if self.metadata.get("id"):
            self.id = self.metadata["id"].split("/")[-1]
        else:
            self.id = "image"
        if self.metadata.get("type"):
            self.type = self.metadata["type"]
        else:
            self.type = None

        self.params = {
            "name": self.id,
            "crs": image.projection().crs().getInfo(),
            "region": image.geometry().getInfo(),
            "filePerBand": False,
            "formatOptions": {"cloudOptimized": True},
        }

        self.viz_params = {}

    @property
    def id(self):
        return self.metadata.get("id")

    @id.setter
    def id(self, value):
        assert value, "Image ID cannot be empty"
        self.metadata["id"] = value

    @property
    def type(self):
        return self.metadata.get("type")

    @type.setter
    def type(self, value):
        self.metadata["type"] = value

    def get_property(self, property):
        """Get image metadata property."""
        if self.metadata.get("properties"):
            return self.metadata["properties"].get(property)
        else:
            return None

    def set_property(self, property, value):
        """Set image metadata property."""
        if not self.metadata.get("properties"):
            self.metadata["properties"] = {}

        self.metadata["properties"][property] = value

    def get_params(self, parameter):
        """Get GEE parameters."""
        return self.params.get(parameter)

    def set_params(self, parameter, value):
        """Set GEE parameters.

        TODO: validate params
        """
        self.params[parameter] = value

    def get_viz_params(self, parameter):
        """Get GEE visualization parameters."""
        return self.viz_params.get(parameter)

    def set_viz_params(self, parameter, value):
        """Set GEE visualization parameters.

        TODO: validate viz_params
        """
        self.viz_params[parameter] = value

    def get_url(
        self,
        params = None,
        viz_params = None,
        preview: bool = False,
        prev_format="png",
    ):
        """Get GEE URL to download the image.

        Parameters
        ----------
        params : dict or None (default None)
            Parameters to pass to the GEE API. If None, will use the default parameters. Options include:
            name, scale, crs, crs_transform, region, format, dimensions, filePerBand and others. See more
            information see https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl
        viz_params : dict or None (default None)
            Parameters to pass to ee.Image.visualize. Required if preview = True. For more information see
            https://developers.google.com/earth-engine/apidocs/ee-image-visualize
        """
        from copy import copy

        if params:
            for key, value in params.items():
                self.set_params(key, value)

        if viz_params:
            for key, value in viz_params.items():
                self.set_viz_params(key, value)

        if preview:
            params = copy(self.params)
            params["format"] = prev_format
            return self.image.visualize(**self.viz_params).getThumbURL(params)

        else:
            return self.image.getDownloadURL(self.params)

    def save_metadata(self, path):
        """Save metadata as a JSON file."""
        with open(os.path.join(path, f"{self.id}-metadata.json"), "w") as f:
            f.write(json.dumps(self.metadata, indent=4))

    def save_preview(
        self,
        path: str,
        viz_params: dict or None = None,
        format: str = "png",
        **kargs,
    ):
        """Save a preview of the image.

        Parameters
        ----------
        path : str
            Directory to save the downloaded image.
        viz_params : dict
            Parameters to pass to the GEE API. If None, will use the default parameters.
            For more information see https://developers.google.com/earth-engine/apidocs/ee-image-visualize
        format : str
            Format of the image to download. Default is png.
        """
        url = self.get_url(self.params, viz_params, preview=True, prev_format=format)
        download_from_url(
            url, f"{self.id}-preview.{format}", path, preview=True, **kargs
        )

    def to_geotif(self, path: str, **kargs):
        """Download image as GeoTIF.

        Parameters
        ----------
        path : str
            Directory to save the downloaded image.
        """
        url = self.get_url(self.params)
        if self.params.get("formatOptions")["cloudOptimized"]:
            filename = f"{self.id}-cog.tif"
        else:
            filename = f"{self.id}.tif"

        download_from_url(url, filename, path, **kargs)

    def metadata_from_collection(self, collection: ee.ImageCollection):
        """Get metadata from an image collection.

        Parameters
        ----------
        collection : ee.ImageCollection
            Image collection to get metadata from.
        """
        # emulates T-SQL COALESCE function
        def coalesce(*arg):
            return next((a for a in arg if a), None)

        # safe method for indexing lists
        def get_item(_list, index):
            try:
                return _list[index]
            except (IndexError, TypeError):
                return None

        collection_info = collection.sort("system:time_start", False).getInfo()

        if collection_info.get("properties"):
            properties = collection_info.get("properties")
        else:
            assert (
                len(collection_info.get("features")) > 1
            ), "Collection has only one feature or is empty."
            properties = collection_info.get("features")[0].get("properties")

        features = collection_info.get("features")
        properties_end = features[-1].get("properties")

        description = coalesce(
            properties.get("description"),
            properties.get("system:description"),
            properties_end.get("description"),
        )

        date_start, date_end = [
            coalesce(
                get_item(properties.get("date_range"), 0),
                properties.get("system:time_start"),
            ),
            coalesce(
                get_item(properties.get("date_range"), 1),
                properties.get("system:time_end"),
                properties_end.get("system:time_end"),
                properties_end.get("system:time_start"),
            ),
        ]

        self.type = coalesce(
            collection_info.get("type"), collection_info.get("type_name")
        )
        self.set_property("system:time_start", date_start)
        self.set_property("system:time_end", date_end)
        self.set_property("description", description)

        print("Image metadata updated successfully.")


def harmonize_to_oli(image):
    """Applies linear adjustments to transform earlier sensors to more closely
    match LANDSAT 8 OLI as described in:

        Roy et al. (2016). "Characterization of Landsat-7 to Landsat-8
        reflective wavelength and normalized difference vegetation index
        continuity." Remote Sensing of Environment (185): 57–70.
        https://doi.org/10.1016/j.rse.2015.12.024
    """

    ROY_COEFS = {  # B, G, R, NIR, SWIR1, SWIR2
        "intercepts": ee.Image.constant(
            [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]
        ).multiply(
            10000
        ),  # this scales LS7ETM to match LS8OLI scaling
        "slopes": ee.Image.constant([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]),
    }

    harmonized = (
        image.select(["B", "G", "R", "NIR", "SWIR1", "SWIR2"])
        .multiply(ROY_COEFS["slopes"])
        .add(ROY_COEFS["intercepts"])
        .round()
        .toShort()
    )

    return harmonized


def mask_stuff(image):
    """Masks pixels likely to be cloud, shadow, water, or snow in a LANDSAT
    image based on the `pixel_qa` band."""
    qa = image.select("pixel_qa")

    shadow = qa.bitwiseAnd(8).eq(0)
    snow = qa.bitwiseAnd(16).eq(0)
    cloud = qa.bitwiseAnd(32).eq(0)
    water = qa.bitwiseAnd(4).eq(0)

    masked = (
        image.updateMask(shadow).updateMask(cloud).updateMask(snow).updateMask(water)
    )

    return masked


def get_landsat_collection(aoi, start_year, end_year, band="SWIR1"):
    """Builds a time series of summertime LANDSAT imagery within an Area of
    Interest, returning a single composite image for a single band each year.

    Parameters
    ----------
    aoi :
        An ee.Geometry object representing the Area of Interest.
    start_year : int
        The first year to include in the time series.
    end_year : int
        The last year to include in the time series.
    band : str, optional
        The band to include in the time series. Defaults to 'SWIR1'.

    Returns
    -------
    ee.ImageCollection
    """
    years = range(start_year, end_year + 1)
    images = []

    for i, year in enumerate(years):
        if year >= 1984 and year <= 2011:
            sensor, bands = "LT05", ["B1", "B2", "B3", "B4", "B5", "B7"]
        elif year == 2012:
            continue
        elif year >= 2013:
            sensor, bands = "LC08", ["B2", "B3", "B4", "B5", "B6", "B7"]

        landsat = ee.ImageCollection(f"LANDSAT/{sensor}/C01/T1_SR")

        coll = landsat.filterBounds(aoi).filterDate(f"{year}-06-15", f"{year}-09-15")
        masked = coll.map(mask_stuff).select(
            bands, ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]
        )
        medoid = composite.medoid(masked, discard_zeros=True)

        if sensor != "LC08":
            img = harmonize_to_oli(medoid)
        else:
            img = medoid

        if band == "NBR":
            nbr = (
                img.normalizedDifference(["NIR", "SWIR2"]).rename("NBR").multiply(1000)
            )
            img = img.addBands(nbr)

        images.append(
            img.select([band]).set(
                "system:time_start", coll.first().get("system:time_start")
            )
        )

    return ee.ImageCollection(images)


def download_from_url(
    url, filename=None, path=".", preview=False, retry=True, overwrite=False
    ):
    """Given a download URL, downloads the zip file and writes it to disk.

    Parameters
    ----------
    url : str
        URL from which the raster will be downloaded.
    save_as : str
        The raster will be saved as this filename. If None, the filename will be the zipped file name.
    path : str
        Path to which the raster will be saved.
    """
    import requests.adapters

    out_path = os.path.join(path, filename)

    if os.path.exists(out_path) and overwrite is False:
        print(
            f"File already exists: {filename}. Set overwrite to True to bypass this message."
        )
        return

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("http://", adapter)
    response = session.get(url)

    with requests.get(url) as response:
        if preview:
            imgfile = "thumbnail.png"
            if filename:
                imgfile = filename
            with open(os.path.join(path, imgfile), "wb") as f:
                f.write(response.content)

        else:
            try:
                zip = ZipFile(BytesIO(response.content))
            except Exception as e:  # downloaded zip is corrupt/failed
                raise e

            imgfile = zip.infolist()[0]
            if filename:
                imgfile.filename = filename
            zip.extract(imgfile, path=path)

    # Verify that the file was downloaded.
    if not os.path.exists(out_path):
        print(f"Download failed")
        if retry:
            print("Retring to download from {url} ...")
            return download_from_url(url, filename, path, retry=False)

    print(f"GEE image saved as {out_path}")


def multithreaded_download(to_download, function, threads=12):
    num_downloads = len(to_download)
    assert num_downloads > 0, "Empty list of parameters passed."
    print("\n", "Attempting download of {:,d} images".format(num_downloads))

    with ThreadPoolExecutor(threads) as executor:
        print("Starting to download files.")
        jobs = [executor.submit(function, *params) for params in to_download]
        results = []

        try:
            for job in tqdm(as_completed(jobs), total=len(jobs)):
                results.append(job.result())
        except Exception as e:
            print(f'Exception "{e}" raised while downloading files.')
            pass

    return results

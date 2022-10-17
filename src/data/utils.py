# %%
import numpy as np

def degrees_to_meters(deg, angle='lon'):
    """Convert degrees to meters.

    From https://earthscience.stackexchange.com/questions/7350/
    converting-grid-resolution-from-degrees-to-kilometers

    Parameters
    ----------
    deg : float
        Degrees to convert.
    plane : str, optional
        Indicates whether deg are degrees latitude or degrees longitude. 
        The default is 'lon'.

    Returns
    -------
    float : Distance in meters.
    """
    import math

    R = 6378137.0  # Earth radius in meters
    rad = math.radians(deg)

    if angle == 'lon':
        return R * rad * math.cos(rad)

    elif angle == 'lat':
        return R * rad


def split_bbox(dim, bbox_to_split):
    """Split a bounding box into n = dim x dim bounding boxes.

    Parameters
    ----------
    dim : int
        Number of splits per dimension of bbox with shape (dim, dim). The number
        of parts (n) to split the original bounding box will be dim x dim.
    bbox_to_split : list-like
        The bounding box to split, with format [xmin, ymin, xmax, ymax].

    Returns
    -------
    np.array
        An array of n bounding boxes.
    """
    xmin, ymin, xmax, ymax = bbox_to_split

    w = (xmax - xmin) / dim
    h = (ymax - ymin) / dim

    # For testing
    # cols = ['xmin', *[f'xmin + w*{dim + 1}' for dim in range(dim - 1)], 'xmax']
    # rows = ['ymin', *[f'ymin + l*{dim + 1}' for dim in range(dim - 1)], 'ymax']

    cols = [xmin, *[xmin + w * (dim + 1) for dim in range(dim - 1)], xmax]
    rows = [ymin, *[ymin + h * (dim + 1) for dim in range(dim - 1)], ymax]

    coords = np.array(np.meshgrid(cols, rows)).T

    bbox_splitted = []
    for i in range(dim):
        bbox_splitted.append(
            [
                np.array([coords[i][j], coords[i + 1][k]]).flatten()
                for j, k in zip(range(dim), range(1, dim + 1))
            ]
        )

    return np.array([x for sbl in bbox_splitted for x in sbl])


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
    from pathlib import Path
    import os

    filepath = Path(*args)

    if not filepath.exists():
        os.makedirs(filepath)

    return filepath

# %%

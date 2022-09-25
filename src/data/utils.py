import numpy as np

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
    
    w = (xmax-xmin)/dim
    h = (ymax-ymin)/dim
    
    # For testing
    #cols = ['xmin', *[f'xmin + w*{dim + 1}' for dim in range(dim - 1)], 'xmax']
    #rows = ['ymin', *[f'ymin + l*{dim + 1}' for dim in range(dim - 1)], 'ymax']
    
    cols = [xmin, *[xmin + w*(dim + 1) for dim in range(dim - 1)], xmax]
    rows = [ymin, *[ymin + h*(dim + 1) for dim in range(dim - 1)], ymax]
    
    coords = np.array(np.meshgrid(cols, rows)).T
    
    bbox_splitted = []
    for i in range(dim):
        bbox_splitted.append([
            np.array(
                [coords[i][j], coords[i + 1][k]]
            ).flatten()
            for j,k in zip(range(dim), range(1, dim + 1))
        ])
    
    return np.array([x for sbl in bbox_splitted for x in sbl])

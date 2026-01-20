import numpy as np
from Vectorization.ImageGraphs.binary_image_graph3 import binary_image_graph3
from Vectorization.ImageGraphs.private.check_connectivity import check_connectivity

def image_graph3(sz, conn=26):
    """
    Computes the pixel neighbor graph for a 3-D image.
    """
    if not isinstance(sz, (tuple, list)) or len(sz) != 3:
        raise ValueError("Input `sz` must be a tuple or list of 3 integers.")

    bw = np.ones(sz, dtype=bool)
    
    conn = check_connectivity(conn)

    g, _ = binary_image_graph3(bw, conn)

    return g

import numpy as np
from Vectorization.ImageGraphs.binary_image_graph import binary_image_graph
from Vectorization.ImageGraphs.private.check_connectivity import check_connectivity

def image_graph(sz, conn=8):
    """
    Computes the pixel neighbor graph for a 2-D image.
    """
    if not isinstance(sz, (tuple, list)) or len(sz) != 2:
        raise ValueError("Input `sz` must be a tuple or list of 2 integers.")

    bw = np.ones(sz, dtype=bool)
    
    conn = check_connectivity(conn)

    g, _ = binary_image_graph(bw, conn)

    return g

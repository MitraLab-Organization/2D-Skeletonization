import numpy as np
from Vectorization.ImageGraphs.binary_image_graph3 import binary_image_graph3
from Vectorization.ImageGraphs.private.check_connectivity import check_connectivity

def binary_image_graph(bw, conn=8):
    """
    Computes the pixel neighbor graph for a 2-D binary image.
    """
    if not isinstance(bw, (np.ndarray, list)) or np.array(bw).ndim != 2:
        raise ValueError("Input `bw` must be a 2D numpy array or list.")

    bw = np.asarray(bw, dtype=bool)
    
    # Expand dimensions to make it a 3D image with a single slice
    bw_3d = np.expand_dims(bw, axis=2)
    
    conn_3d = check_connectivity(conn)
    if conn_3d.ndim == 2:
        conn_3d = np.expand_dims(conn_3d, axis=2)

    g, nodenums = binary_image_graph3(bw_3d, conn_3d)
    
    # Remove the z attribute from the nodes
    for node in g.nodes():
        del g.nodes[node]['z']
        
    nodenums = np.squeeze(nodenums, axis=2)

    return g, nodenums

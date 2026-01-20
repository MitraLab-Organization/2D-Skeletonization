import numpy as np
import networkx as nx
from Vectorization.ImageGraphs.private.check_connectivity import check_connectivity

def binary_image_graph3(bw, conn=26):
    """
    Computes the pixel neighbor graph for a 3-D binary image.
    """
    if not isinstance(bw, (np.ndarray, list)) or np.array(bw).ndim != 3:
        raise ValueError("Input `bw` must be a 3D numpy array or list.")
    
    bw = np.asarray(bw, dtype=bool)
    
    conn = check_connectivity(conn)
    
    # To avoid duplicate edges, we only consider half of the connectivity.
    conn_half = conn.copy()
    conn_half.flat[0:len(conn_half.flat)//2] = 0

    conn_offsets_y, conn_offsets_x, conn_offsets_z = np.nonzero(conn_half)
    conn_offsets_x -= int((conn.shape[1] - 1) / 2)
    conn_offsets_y -= int((conn.shape[0] - 1) / 2)
    conn_offsets_z -= int((conn.shape[2] - 1) / 2)

    foreground_pixel_indices = np.flatnonzero(bw)
    nodenums = np.zeros(bw.shape, dtype=int) - 1 # Use -1 for background
    nodenums.flat[foreground_pixel_indices] = np.arange(len(foreground_pixel_indices))

    foreground_y, foreground_x, foreground_z = np.unravel_index(foreground_pixel_indices, bw.shape)

    g = nx.Graph()
    for i, (y, x, z) in enumerate(zip(foreground_y, foreground_x, foreground_z)):
        g.add_node(i, x=x, y=y, z=z, PixelIndex=foreground_pixel_indices[i])

    for dx, dy, dz in zip(conn_offsets_x, conn_offsets_y, conn_offsets_z):
        neighbor_x = foreground_x + dx
        neighbor_y = foreground_y + dy
        neighbor_z = foreground_z + dz

        valid_indices = (
            (neighbor_x >= 0) & (neighbor_x < bw.shape[1]) &
            (neighbor_y >= 0) & (neighbor_y < bw.shape[0]) &
            (neighbor_z >= 0) & (neighbor_z < bw.shape[2])
        )

        neighbor_indices_flat = np.ravel_multi_index(
            (neighbor_y[valid_indices], neighbor_x[valid_indices], neighbor_z[valid_indices]),
            bw.shape
        )
        
        is_foreground = bw.flat[neighbor_indices_flat]

        start_nodes = nodenums.flat[foreground_pixel_indices[valid_indices][is_foreground]]
        end_nodes = nodenums.flat[neighbor_indices_flat[is_foreground]]
        
        weight = np.sqrt(dx**2 + dy**2 + dz**2)
        
        for start, end in zip(start_nodes, end_nodes):
            if start != -1 and end != -1:
                g.add_edge(start, end, weight=weight)

    return g, nodenums
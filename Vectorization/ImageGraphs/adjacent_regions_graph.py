import numpy as np
import networkx as nx
from scipy.ndimage import grey_dilation, grey_erosion
from Vectorization.ImageGraphs.binary_image_graph import binary_image_graph
from Vectorization.ImageGraphs.private.check_connectivity import check_connectivity

def adjacent_regions_graph(L, conn=None):
    """
    Computes a graph of adjacent regions from a label matrix.
    """
    if not isinstance(L, (np.ndarray, list)):
        raise ValueError("Input `L` must be a numpy array or list.")
    
    L = np.asarray(L)
    
    if conn is None:
        if L.ndim == 2:
            conn = 8
        else:
            conn = 26
            
    conn = check_connectivity(conn)

    # We can improve efficiency by restricting our attention to pixels
    # that are known to be different from at least one of their
    # neighbors.
    dilated = grey_dilation(L, footprint=conn)
    eroded = grey_erosion(L, footprint=conn)
    mask = (L != dilated) | (L != eroded)
    
    mask_graph, _ = binary_image_graph(mask, conn)
    
    # Find pairs of pixel values corresponding to neighbor pairs.
    pixel_indices = nx.get_node_attributes(mask_graph, 'PixelIndex')
    end_nodes = np.array(mask_graph.edges())
    
    if end_nodes.size == 0:
        return nx.Graph() # No adjacencies found

    # Get the pixel indices for the start and end nodes of each edge
    start_node_indices = L.flat[np.array([pixel_indices[node] for node in end_nodes[:, 0]])]
    end_node_indices = L.flat[np.array([pixel_indices[node] for node in end_nodes[:, 1]])]

    connected_values = np.vstack([start_node_indices, end_node_indices]).T

    # Remove pairs of the same value.
    connected_values = connected_values[connected_values[:, 0] != connected_values[:, 1]]

    # Remove duplicate value pairs.
    connected_values = np.unique(np.sort(connected_values, axis=1), axis=0)

    # Use unique to compute a node numbering for all the different
    # values in the list of value pairs.
    labels = np.unique(connected_values)
    label_map = {label: i for i, label in enumerate(labels)}
    
    edges = np.array([[label_map[v] for v in row] for row in connected_values])

    # Make the graph.
    g = nx.Graph()
    for i, label in enumerate(labels):
        g.add_node(i, Label=label)
    
    g.add_edges_from(edges)
    
    # Add Labels attribute to edges
    for u, v in g.edges():
        g.edges[u, v]['Labels'] = (g.nodes[u]['Label'], g.nodes[v]['Label'])

    return g

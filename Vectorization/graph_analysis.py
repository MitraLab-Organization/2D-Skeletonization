import numpy as np
from skimage.morphology import remove_small_objects
from skimage.measure import label
from Vectorization.ImageGraphs.binary_image_graph import binary_image_graph
from Vectorization.get_arcs_properties_skel import get_arcs_properties_skel

from scipy.ndimage import find_objects

def graph_analysis(dm_bw, debug=False, min_size=40):
    """
    Performs graph analysis on a binary skeleton image.
    """
    # Remove small objects
    if debug:
        print("Removing small objects...")
    dm_bw_cleaned = remove_small_objects(dm_bw, min_size=min_size, connectivity=2)

    # Find connected components
    if debug:
        print("Labeling connected components...")
    labeled_image, num_labels = label(dm_bw_cleaned, return_num=True, connectivity=2)
    if debug:
        print(f"Found {num_labels} connected components.")
    
    if debug:
        print("Finding objects slices...")
    slices = find_objects(labeled_image)

    cc = {
        'Connectivity': 8,
        'ImageSize': dm_bw.shape,
        'NumObjects': num_labels,
        'PixelIdxList': [],
        'Node_graph': [],
        'arcProperties': []
    }

    if debug:
        print("Processing components...")
    width = dm_bw.shape[1]
    
    for i in range(num_labels):
        if debug and (i + 1) % 1000 == 0:
            print(f"Component {i + 1} / {num_labels}")

        sl = slices[i]
        if sl is None:
            cc['PixelIdxList'].append(np.array([]))
            cc['Node_graph'].append(None)
            cc['arcProperties'].append([])
            continue

        y_offset = sl[0].start
        x_offset = sl[1].start
        
        # Extract small mask
        masked_crop = (labeled_image[sl] == (i + 1))
        
        # PixelIdxList optimization
        local_indices = np.flatnonzero(masked_crop)
        local_y, local_x = np.unravel_index(local_indices, masked_crop.shape)
        global_indices = (local_y + y_offset) * width + (local_x + x_offset)
        cc['PixelIdxList'].append(global_indices)

        node_graph, _ = binary_image_graph(masked_crop)
        
        arc_properties = get_arcs_properties_skel(node_graph, masked_crop)
        
        # Adjust arc_properties to global
        for prop in arc_properties:
            prop['x'] += x_offset
            prop['y'] += y_offset
            prop['Pos'][:, 0] += x_offset
            prop['Pos'][:, 1] += y_offset
            # Recalculate global arc indices
            prop['arc'] = (prop['y'].astype(np.int64) * width + prop['x'].astype(np.int64)).tolist()

        cc['arcProperties'].append(arc_properties)

        # Adjust graph to global
        for node in node_graph.nodes():
            node_graph.nodes[node]['x'] += x_offset
            node_graph.nodes[node]['y'] += y_offset
            gx = node_graph.nodes[node]['x']
            gy = node_graph.nodes[node]['y']
            node_graph.nodes[node]['PixelIndex'] = gy * width + gx
            
        cc['Node_graph'].append(node_graph)

    return cc

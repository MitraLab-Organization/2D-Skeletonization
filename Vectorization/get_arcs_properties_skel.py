import networkx as nx
import numpy as np

def get_arcs_properties_skel(node_graph, masked_image):
    """
    Finds arc properties from a skeleton graph.
    This function decomposes the graph into segments (arcs) that connect
    endpoints and branch points.
    """
    if not isinstance(node_graph, nx.Graph):
        raise TypeError("Input `node_graph` must be a networkx Graph object.")

    g = node_graph.copy()
    if g.number_of_nodes() == 0:
        return [{'length': 0}]

    degrees = dict(g.degree())
    branch_points = {n for n, d in degrees.items() if d > 2}
    end_points = {n for n, d in degrees.items() if d == 1}
    special_points = branch_points.union(end_points)

    line_paths = []
    visited_edges = set()

    for u, v in g.edges():
        edge = tuple(sorted((u, v)))
        if edge in visited_edges:
            continue

        # Start a new path from this edge
        path = [u, v]
        visited_edges.add(edge)

        # Extend from u
        curr = u
        prev = v
        while curr not in special_points:
            neighbors = list(g.neighbors(curr))
            if len(neighbors) != 2: break
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            
            next_edge = tuple(sorted((curr, next_node)))
            if next_edge in visited_edges: break

            path.insert(0, next_node)
            visited_edges.add(next_edge)
            prev = curr
            curr = next_node

        # Extend from v
        curr = v
        prev = u
        while curr not in special_points:
            neighbors = list(g.neighbors(curr))
            if len(neighbors) != 2: break
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]

            next_edge = tuple(sorted((curr, next_node)))
            if next_edge in visited_edges: break

            path.append(next_node)
            visited_edges.add(next_edge)
            prev = curr
            curr = next_node
            
        line_paths.append(path)

    # Add single-edge paths between special points that might have been missed
    for u, v in g.edges():
        if u in special_points and v in special_points:
            edge = tuple(sorted((u, v)))
            if edge not in visited_edges:
                line_paths.append([u, v])
                visited_edges.add(edge)

    arc_properties = []
    pixel_indices = nx.get_node_attributes(node_graph, 'PixelIndex')
    if not pixel_indices and node_graph.number_of_nodes() > 0:
        pixel_indices = {n: n for n in node_graph.nodes()}

    if not pixel_indices:
        return [{'length': 0}]

    for path in line_paths:
        if len(path) < 2:
            continue
            
        arc = [pixel_indices[node] for node in path]
        y, x = np.unravel_index(arc, masked_image.shape)
        pos = np.column_stack([x, y])
        
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        length = np.sum(segment_lengths)
        
        arc_properties.append({
            'arc': arc,
            'x': x,
            'y': y,
            'Pos': pos,
            'length': length
        })

    if not arc_properties:
        return [{'length': 0}]

    return arc_properties